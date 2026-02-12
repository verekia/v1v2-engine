import { buildBVH, raycastBVH } from './bvh.ts'
import { createRenderer as createRendererInternal } from './gpu.ts'
import {
  m4FromTRS,
  m4Invert,
  m4LookAt,
  m4Multiply,
  m4TransformDirection,
  m4TransformNormal,
  m4TransformPoint,
  v3Length,
  v3Normalize,
} from './math.ts'
import { findBoneNodeIndex } from './skin.ts'

import type { BVH } from './bvh.ts'
import type { BackendType, IRenderer, RenderScene } from './renderer.ts'
import type { SkinInstance } from './skin.ts'

// ── Camera ──────────────────────────────────────────────────────────────────

export class Camera {
  fov = Math.PI / 3
  near = 0.1
  far = 1000
  readonly eye = new Float32Array([0, -10, 5])
  readonly target = new Float32Array([0, 0, 0])
  readonly up = new Float32Array([0, 0, 1])
}

// ── Bloom config ────────────────────────────────────────────────────────────

export interface BloomConfig {
  enabled: boolean
  intensity: number
  threshold: number
  /** Spread multiplier for blur kernel — higher = wider glow, same cost. Default 1 */
  radius: number
  /** How much bloom-emitting meshes are pushed toward white (0 = no change, 1 = fully white). Default 0 */
  whiten: number
}

function createDefaultBloom(): BloomConfig {
  return { enabled: false, intensity: 1, threshold: 0, radius: 1, whiten: 0 }
}

// ── Outline config ──────────────────────────────────────────────────────────

export interface OutlineConfig {
  enabled: boolean
  thickness: number
  color: [number, number, number]
  /** Distance at which outline is full thickness. Beyond this, outline shrinks proportionally. 0 = no scaling. */
  distanceFactor: number
}

function createDefaultOutline(): OutlineConfig {
  return { enabled: false, thickness: 3, color: [0, 0, 0], distanceFactor: 0 }
}

// ── Shadow config ───────────────────────────────────────────────────────────

export interface ShadowConfig {
  enabled: boolean
  /** Number of shadow cascades (1–4). Default 4. */
  cascades: number
  /** Maximum shadow distance from camera. Default 200. */
  far: number
  /** Depth bias to reduce shadow acne. Default 0.0001. */
  bias: number
  /** Cascade split blend: 0 = uniform, 1 = logarithmic. Default 0.5. */
  lambda: number
}

function createDefaultShadow(): ShadowConfig {
  return {
    enabled: false,
    cascades: 4,
    far: 200,
    bias: 0.0001,
    lambda: 0.5,
  }
}

// ── Mesh ────────────────────────────────────────────────────────────────────

export interface MeshOptions {
  geometry: number
  position?: [number, number, number]
  rotation?: [number, number, number]
  scale?: [number, number, number]
  color?: [number, number, number]
  alpha?: number
  unlit?: boolean
  skinned?: boolean
  skinInstanceId?: number
  aoMap?: number
  bloom?: number
  /** Outline group (0 = no outline, positive = group ID; meshes in the same group share one outline) */
  outline?: number
}

export class Mesh {
  readonly position: Float32Array
  readonly rotation: Float32Array
  readonly scale: Float32Array
  readonly color: Float32Array
  alpha: number
  geometry: number
  visible = true
  unlit: boolean
  skinned: boolean
  skinInstanceId: number
  aoMap: number
  bloom: number
  outline: number
  boneParent: Mesh | null = null
  boneSkinInstance: SkinInstance | null = null
  boneNodeIndex = -1

  constructor(opts: MeshOptions) {
    this.position = new Float32Array(opts.position ?? [0, 0, 0])
    this.rotation = new Float32Array(opts.rotation ?? [0, 0, 0])
    this.scale = new Float32Array(opts.scale ?? [1, 1, 1])
    this.color = new Float32Array(opts.color ?? [1, 1, 1])
    this.alpha = opts.alpha ?? 1
    this.geometry = opts.geometry
    this.unlit = opts.unlit ?? false
    this.skinned = opts.skinned ?? false
    this.skinInstanceId = opts.skinInstanceId ?? -1
    this.aoMap = opts.aoMap ?? -1
    this.bloom = opts.bloom ?? 0
    this.outline = opts.outline ?? 0
  }
}

// ── Geometry registration (internal) ────────────────────────────────────────

type GeoReg =
  | { type: 'static'; vertices: Float32Array; indices: Uint16Array | Uint32Array; skinned: false }
  | {
      type: 'skinned'
      vertices: Float32Array
      indices: Uint16Array | Uint32Array
      joints: Uint8Array
      weights: Float32Array
      skinned: true
    }
  | {
      type: 'textured'
      vertices: Float32Array
      indices: Uint16Array | Uint32Array
      uvs: Float32Array
      skinned: false
    }

interface TexReg {
  data: Uint8Array
  width: number
  height: number
}

// ── Raycast result ─────────────────────────────────────────────────────────

export interface RaycastHit {
  hit: boolean
  distance: number
  pointX: number
  pointY: number
  pointZ: number
  normalX: number
  normalY: number
  normalZ: number
  faceIndex: number
  mesh: Mesh | null
}

export function createRaycastHit(): RaycastHit {
  return {
    hit: false,
    distance: Infinity,
    pointX: 0,
    pointY: 0,
    pointZ: 0,
    normalX: 0,
    normalY: 0,
    normalZ: 0,
    faceIndex: -1,
    mesh: null,
  }
}

// ── Scene ───────────────────────────────────────────────────────────────────

export class Scene {
  readonly camera = new Camera()
  readonly shadow: ShadowConfig = createDefaultShadow()
  readonly bloom: BloomConfig = createDefaultBloom()
  readonly outline: OutlineConfig = createDefaultOutline()
  readonly lightDirection = new Float32Array(3)
  readonly lightDirColor = new Float32Array(3)
  readonly lightAmbientColor = new Float32Array(3)
  readonly skinInstances: SkinInstance[] = []

  private _meshes: Mesh[] = []
  private _renderer: IRenderer
  private _canvas: HTMLCanvasElement
  private _maxEntities: number
  private _maxSkinnedEntities: number | undefined
  private _geoRegs = new Map<number, GeoReg>()
  private _texRegs = new Map<number, TexReg>()
  private _bvhCache = new Map<number, BVH>()
  private _nextGeoId = 0
  private _nextTexId = 0

  // SoA arrays synced before each render
  private _positions: Float32Array
  private _scales: Float32Array
  private _worldMatrices: Float32Array
  private _colors: Float32Array
  private _alphas: Float32Array
  private _geometryIds: Uint8Array
  private _skinInstanceIds: Int16Array
  private _renderMask: Uint8Array
  private _skinnedMask: Uint8Array
  private _unlitMask: Uint8Array
  private _texturedMask: Uint8Array
  private _aoMapIds: Int16Array
  private _bloomValues: Float32Array
  private _outlineMask: Uint8Array

  // View/projection matrices (written during render)
  private _viewMatrix = new Float32Array(16)
  private _projMatrix = new Float32Array(16)

  // Bone attachment scratch buffers
  private _boneScratch = new Float32Array(16)
  private _meshIndexMap = new Map<Mesh, number>()

  // Shadow scratch buffers (CSM)
  private _lightDirNorm = new Float32Array(3)
  private _lightView = new Float32Array(16)
  private _lightProj = new Float32Array(16)
  private _cascadeVPs = new Float32Array(64) // 4 × mat4
  private _cascadeSplits = new Float32Array(4)
  private _frustumCorners = new Float32Array(24) // 8 × vec3
  private _lightCorner = new Float32Array(3)
  private _cascadeCenter = new Float32Array(3)
  private _lightUp = new Float32Array(3)

  // Cached RenderScene (avoids per-frame object literal allocation)
  private _renderScene: RenderScene | null = null

  // Raycast scratch buffers
  private _rayWorldMat = new Float32Array(16)
  private _rayInvMat = new Float32Array(16)
  private _rayLocalOrigin = new Float32Array(3)
  private _rayLocalDir = new Float32Array(3)
  private _rayFaceOut = new Uint32Array(1)
  private _rayNormalOut = new Float32Array(3)
  private _rayWorldNormal = new Float32Array(3)

  /** @internal — use createScene() instead */
  constructor(renderer: IRenderer, canvas: HTMLCanvasElement, maxEntities: number, maxSkinnedEntities?: number) {
    this._renderer = renderer
    this._canvas = canvas
    this._maxEntities = maxEntities
    this._maxSkinnedEntities = maxSkinnedEntities
    this._positions = new Float32Array(maxEntities * 3)
    this._scales = new Float32Array(maxEntities * 3)
    this._worldMatrices = new Float32Array(maxEntities * 16)
    this._colors = new Float32Array(maxEntities * 3)
    this._alphas = new Float32Array(maxEntities)
    this._geometryIds = new Uint8Array(maxEntities)
    this._skinInstanceIds = new Int16Array(maxEntities).fill(-1)
    this._renderMask = new Uint8Array(maxEntities)
    this._skinnedMask = new Uint8Array(maxEntities)
    this._unlitMask = new Uint8Array(maxEntities)
    this._texturedMask = new Uint8Array(maxEntities)
    this._aoMapIds = new Int16Array(maxEntities).fill(-1)
    this._bloomValues = new Float32Array(maxEntities)
    this._outlineMask = new Uint8Array(maxEntities)
  }

  get backendType(): BackendType {
    return this._renderer.backendType
  }

  get drawCalls(): number {
    return this._renderer.drawCalls
  }

  get meshes(): readonly Mesh[] {
    return this._meshes
  }

  get viewMatrix(): Float32Array {
    return this._viewMatrix
  }

  get projMatrix(): Float32Array {
    return this._projMatrix
  }

  get canvas(): HTMLCanvasElement {
    return this._canvas
  }

  // ── Geometry management ─────────────────────────────────────────────────

  registerGeometry(vertices: Float32Array, indices: Uint16Array | Uint32Array): number {
    const id = this._nextGeoId++
    this._renderer.registerGeometry(id, vertices, indices)
    this._geoRegs.set(id, { type: 'static', vertices, indices, skinned: false })
    return id
  }

  registerSkinnedGeometry(
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    joints: Uint8Array,
    weights: Float32Array,
  ): number {
    const id = this._nextGeoId++
    this._renderer.registerSkinnedGeometry(id, vertices, indices, joints, weights)
    this._geoRegs.set(id, { type: 'skinned', vertices, indices, joints, weights, skinned: true })
    return id
  }

  registerTexturedGeometry(vertices: Float32Array, indices: Uint16Array | Uint32Array, uvs: Float32Array): number {
    const id = this._nextGeoId++
    this._renderer.registerTexturedGeometry(id, vertices, indices, uvs)
    this._geoRegs.set(id, { type: 'textured', vertices, indices, uvs, skinned: false })
    return id
  }

  registerTexture(data: Uint8Array, width: number, height: number): number {
    const id = this._nextTexId++
    this._renderer.registerTexture(id, data, width, height)
    this._texRegs.set(id, { data, width, height })
    return id
  }

  // ── Mesh management ─────────────────────────────────────────────────────

  add(mesh: Mesh): Mesh {
    this._meshes.push(mesh)
    return mesh
  }

  remove(mesh: Mesh): void {
    const idx = this._meshes.indexOf(mesh)
    if (idx >= 0) this._meshes.splice(idx, 1)
  }

  // ── Bone attachment ────────────────────────────────────────────────────

  attachToBone(mesh: Mesh, parentMesh: Mesh, boneName: string): void {
    if (parentMesh.skinInstanceId < 0) throw new Error('Parent mesh has no skin instance')
    const skinInstance = this.skinInstances[parentMesh.skinInstanceId]!
    const nodeIndex = findBoneNodeIndex(skinInstance.skeleton, boneName)
    mesh.boneParent = parentMesh
    mesh.boneSkinInstance = skinInstance
    mesh.boneNodeIndex = nodeIndex
  }

  detachFromBone(mesh: Mesh): void {
    mesh.boneParent = null
    mesh.boneSkinInstance = null
    mesh.boneNodeIndex = -1
  }

  // ── Lighting ────────────────────────────────────────────────────────────

  setDirectionalLight(dir: [number, number, number], color: [number, number, number]): void {
    this.lightDirection.set(dir)
    this.lightDirColor.set(color)
  }

  setAmbientLight(color: [number, number, number]): void {
    this.lightAmbientColor.set(color)
  }

  // ── Resize ──────────────────────────────────────────────────────────────

  resize(w: number, h: number): void {
    this._renderer.resize(w, h)
  }

  // ── Raycasting ─────────────────────────────────────────────────────

  /** Pre-build a BVH for a registered geometry. Optional — raycast() lazily builds on first use. */
  buildBVH(geometryId: number): void {
    if (this._bvhCache.has(geometryId)) return
    const geo = this._geoRegs.get(geometryId)
    if (!geo) return
    this._bvhCache.set(geometryId, buildBVH(geo.vertices, geo.indices))
  }

  /**
   * Cast a ray and write the closest hit into `result`.
   * Direction (dx,dy,dz) must be normalized.
   * Returns true if any hit was found.
   * Pass `meshFilter` to limit the test to a subset of meshes.
   */
  raycast(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    result: RaycastHit,
    meshFilter?: readonly Mesh[],
  ): boolean {
    result.hit = false
    result.distance = Infinity
    result.mesh = null

    const meshes = meshFilter ?? this._meshes
    for (let mi = 0; mi < meshes.length; mi++) {
      const mesh = meshes[mi]!
      if (!mesh.visible || mesh.skinned) continue

      // Lazy BVH build
      let bvh = this._bvhCache.get(mesh.geometry)
      if (!bvh) {
        const geo = this._geoRegs.get(mesh.geometry)
        if (!geo) continue
        bvh = buildBVH(geo.vertices, geo.indices)
        this._bvhCache.set(mesh.geometry, bvh)
      }

      // Compute world matrix & its inverse
      m4FromTRS(this._rayWorldMat, 0, mesh.position, 0, mesh.rotation, 0, mesh.scale, 0)
      if (!m4Invert(this._rayInvMat, 0, this._rayWorldMat, 0)) continue

      // Transform ray to local space
      m4TransformPoint(this._rayLocalOrigin, 0, this._rayInvMat, 0, ox, oy, oz)
      m4TransformDirection(this._rayLocalDir, 0, this._rayInvMat, 0, dx, dy, dz)

      // The t parameter from the BVH is in local-ray-parameter space.
      // Since worldDir is unit, world distance = t * |localDir| … but we keep t
      // in the same parameter space for the entire loop by scaling maxT.
      const localDirLen = v3Length(this._rayLocalDir, 0)
      if (localDirLen < 1e-10) continue
      const maxTLocal = result.distance * localDirLen

      const t = raycastBVH(
        bvh,
        this._rayLocalOrigin[0]!,
        this._rayLocalOrigin[1]!,
        this._rayLocalOrigin[2]!,
        this._rayLocalDir[0]!,
        this._rayLocalDir[1]!,
        this._rayLocalDir[2]!,
        maxTLocal,
        this._rayFaceOut,
        this._rayNormalOut,
      )

      if (t >= 0) {
        const worldDist = t / localDirLen
        if (worldDist < result.distance) {
          result.hit = true
          result.distance = worldDist
          result.mesh = mesh
          result.faceIndex = this._rayFaceOut[0]!
          result.pointX = ox + dx * worldDist
          result.pointY = oy + dy * worldDist
          result.pointZ = oz + dz * worldDist

          // Normal: local→world via (M⁻¹)ᵀ
          m4TransformNormal(
            this._rayWorldNormal,
            0,
            this._rayInvMat,
            0,
            this._rayNormalOut[0]!,
            this._rayNormalOut[1]!,
            this._rayNormalOut[2]!,
          )
          const nlen = v3Length(this._rayWorldNormal, 0) || 1
          result.normalX = this._rayWorldNormal[0]! / nlen
          result.normalY = this._rayWorldNormal[1]! / nlen
          result.normalZ = this._rayWorldNormal[2]! / nlen
        }
      }
    }

    return result.hit
  }

  // ── Backend switching ───────────────────────────────────────────────────

  async switchBackend(canvas: HTMLCanvasElement, type: BackendType): Promise<void> {
    this._renderer.destroy()
    this._renderer = await createRendererInternal(canvas, this._maxEntities, type, this._maxSkinnedEntities)
    this._canvas = canvas
    this._renderScene = null

    // Re-register all geometries on the new renderer
    for (const [id, reg] of this._geoRegs) {
      if (reg.type === 'skinned') {
        this._renderer.registerSkinnedGeometry(id, reg.vertices, reg.indices, reg.joints, reg.weights)
      } else if (reg.type === 'textured') {
        this._renderer.registerTexturedGeometry(id, reg.vertices, reg.indices, reg.uvs)
      } else {
        this._renderer.registerGeometry(id, reg.vertices, reg.indices)
      }
    }

    // Re-register all textures on the new renderer
    for (const [id, tex] of this._texRegs) {
      this._renderer.registerTexture(id, tex.data, tex.width, tex.height)
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────

  render(): void {
    const aspect = this._canvas.width / this._canvas.height
    const cam = this.camera

    // Compute view matrix from camera eye/target/up
    m4LookAt(this._viewMatrix, 0, cam.eye, 0, cam.target, 0, cam.up, 0)

    // Compute projection matrix (delegates to renderer for correct Z range)
    this._renderer.perspective(this._projMatrix, 0, cam.fov, aspect, cam.near, cam.far)

    // Sync meshes to SoA arrays
    const meshes = this._meshes
    const count = meshes.length

    for (let i = 0; i < count; i++) {
      const m = meshes[i]!

      if (!m.visible) {
        this._renderMask[i] = 0
        // Still compute world matrix — invisible meshes may be bone parents
        m4FromTRS(this._worldMatrices, i * 16, m.position, 0, m.rotation, 0, m.scale, 0)
        continue
      }

      const i3 = i * 3

      this._positions[i3] = m.position[0]!
      this._positions[i3 + 1] = m.position[1]!
      this._positions[i3 + 2] = m.position[2]!
      this._scales[i3] = m.scale[0]!
      this._scales[i3 + 1] = m.scale[1]!
      this._scales[i3 + 2] = m.scale[2]!

      m4FromTRS(this._worldMatrices, i * 16, m.position, 0, m.rotation, 0, m.scale, 0)

      this._colors[i3] = m.color[0]!
      this._colors[i3 + 1] = m.color[1]!
      this._colors[i3 + 2] = m.color[2]!
      this._alphas[i] = m.alpha

      this._geometryIds[i] = m.geometry
      this._skinInstanceIds[i] = m.skinInstanceId

      this._renderMask[i] = 1
      this._skinnedMask[i] = m.skinned ? 1 : 0
      this._unlitMask[i] = m.unlit ? 1 : 0
      this._texturedMask[i] = m.aoMap >= 0 ? 1 : 0
      this._aoMapIds[i] = m.aoMap
      this._bloomValues[i] = m.bloom
      this._outlineMask[i] = m.outline
    }

    // Bone attachment pass: override world matrices for bone-attached meshes
    // Build mesh→index map once to avoid O(n) indexOf per bone-attached mesh
    const meshIndexMap = this._meshIndexMap
    meshIndexMap.clear()
    for (let i = 0; i < count; i++) meshIndexMap.set(meshes[i]!, i)

    for (let i = 0; i < count; i++) {
      const m = meshes[i]!
      if (!m.boneParent || !m.boneSkinInstance || m.boneNodeIndex < 0) continue
      const parentIdx = meshIndexMap.get(m.boneParent)
      if (parentIdx === undefined) continue
      const boneGlobal = m.boneSkinInstance.globalMatrices
      const boneOff = m.boneNodeIndex * 16
      // temp = boneGlobalMatrix × meshLocalTRS
      m4Multiply(this._boneScratch, 0, boneGlobal, boneOff, this._worldMatrices, i * 16)
      // worldMatrices[i] = parentWorldMatrix × temp
      m4Multiply(this._worldMatrices, i * 16, this._worldMatrices, parentIdx * 16, this._boneScratch, 0)
      // Update positions from final world matrix so frustum culling uses the actual position
      const wo = i * 16
      this._positions[i * 3] = this._worldMatrices[wo + 12]!
      this._positions[i * 3 + 1] = this._worldMatrices[wo + 13]!
      this._positions[i * 3 + 2] = this._worldMatrices[wo + 14]!
    }

    // Build RenderScene (reuse cached object to avoid per-frame allocation)
    let rs = this._renderScene
    if (!rs) {
      rs = {
        cameraView: this._viewMatrix,
        cameraProj: this._projMatrix,
        lightDirection: this.lightDirection,
        lightDirColor: this.lightDirColor,
        lightAmbientColor: this.lightAmbientColor,
        entityCount: count,
        renderMask: this._renderMask,
        skinnedMask: this._skinnedMask,
        unlitMask: this._unlitMask,
        positions: this._positions,
        scales: this._scales,
        worldMatrices: this._worldMatrices,
        colors: this._colors,
        alphas: this._alphas,
        geometryIds: this._geometryIds,
        skinInstanceIds: this._skinInstanceIds,
        skinInstances: this.skinInstances,
        texturedMask: this._texturedMask,
        aoMapIds: this._aoMapIds,
        bloomEnabled: this.bloom.enabled,
        bloomIntensity: this.bloom.intensity,
        bloomThreshold: this.bloom.threshold,
        bloomRadius: this.bloom.radius,
        bloomWhiten: this.bloom.whiten,
        bloomValues: this._bloomValues,
        outlineEnabled: this.outline.enabled,
        outlineThickness: this.outline.thickness,
        outlineColor: this.outline.color,
        outlineDistanceFactor: this.outline.distanceFactor,
        outlineMask: this._outlineMask,
      }
      this._renderScene = rs
    }
    // Update mutable fields
    rs.entityCount = count
    rs.skinInstances = this.skinInstances
    rs.bloomEnabled = this.bloom.enabled
    rs.bloomIntensity = this.bloom.intensity
    rs.bloomThreshold = this.bloom.threshold
    rs.bloomRadius = this.bloom.radius
    rs.bloomWhiten = this.bloom.whiten
    rs.outlineEnabled = this.outline.enabled
    rs.outlineThickness = this.outline.thickness
    rs.outlineColor = this.outline.color
    rs.outlineDistanceFactor = this.outline.distanceFactor

    // Compute cascaded shadow VPs if enabled
    if (this.shadow.enabled) {
      const s = this.shadow
      const numCascades = Math.max(1, Math.min(4, s.cascades | 0))
      const camNear = cam.near
      const shadowFar = Math.min(s.far, cam.far)

      // Compute cascade split distances (PSSM: blend of logarithmic + uniform)
      const splits = this._cascadeSplits
      for (let i = 0; i < numCascades; i++) {
        const p = (i + 1) / numCascades
        const log = camNear * Math.pow(shadowFar / camNear, p)
        const uni = camNear + (shadowFar - camNear) * p
        splits[i] = s.lambda * log + (1 - s.lambda) * uni
      }

      // Compute inverse VP to unproject frustum corners
      const invVP = this._lightProj // reuse scratch
      m4Multiply(invVP, 0, this._projMatrix, 0, this._viewMatrix, 0)
      m4Invert(invVP, 0, invVP, 0)

      // Light view: look along light direction, centered at origin
      v3Normalize(this._lightDirNorm, 0, this.lightDirection, 0)

      // Build a stable light view basis from the light direction
      const ld = this._lightDirNorm
      // Pick an up vector that isn't parallel to light direction
      const absX = Math.abs(ld[0]!),
        absY = Math.abs(ld[1]!),
        absZ = Math.abs(ld[2]!)
      const lightUp = this._lightUp
      if (absZ > absX && absZ > absY) {
        lightUp[0] = 0
        lightUp[1] = 1
        lightUp[2] = 0
      } else {
        lightUp[0] = 0
        lightUp[1] = 0
        lightUp[2] = 1
      }

      for (let c = 0; c < numCascades; c++) {
        const nearDist = c === 0 ? camNear : splits[c - 1]!
        const farDist = splits[c]!

        // Compute 8 frustum corners in NDC then unproject to world space
        // NDC corners: x,y ∈ {-1,+1}, z ∈ {nearNDC, farNDC}
        // For WebGPU: z ∈ [0,1], for WebGL: z ∈ [-1,1]
        // Instead of dealing with NDC, compute frustum corners directly from camera params
        const tanHalfFov = Math.tan(cam.fov / 2)
        const nearH = tanHalfFov * nearDist
        const nearW = nearH * aspect
        const farH = tanHalfFov * farDist
        const farW = farH * aspect

        // Camera basis vectors from view matrix (column-major)
        // View matrix: rows are right, up, -forward; columns store these transposed
        const vm = this._viewMatrix
        // Right = first row of view matrix = [vm[0], vm[4], vm[8]]
        const rx = vm[0]!,
          ry = vm[4]!,
          rz = vm[8]!
        // Up = second row = [vm[1], vm[5], vm[9]]
        const ux = vm[1]!,
          uy = vm[5]!,
          uz = vm[9]!
        // Forward = -third row = -[vm[2], vm[6], vm[10]]
        const fx = -vm[2]!,
          fy = -vm[6]!,
          fz = -vm[10]!

        const ex = cam.eye[0]!,
          ey = cam.eye[1]!,
          ez = cam.eye[2]!

        // Near plane center & far plane center
        const ncx = ex + fx * nearDist,
          ncy = ey + fy * nearDist,
          ncz = ez + fz * nearDist
        const fcx = ex + fx * farDist,
          fcy = ey + fy * farDist,
          fcz = ez + fz * farDist

        // 8 corners: near TL, TR, BL, BR, far TL, TR, BL, BR
        const fc = this._frustumCorners
        // Near top-left
        fc[0] = ncx - rx * nearW + ux * nearH
        fc[1] = ncy - ry * nearW + uy * nearH
        fc[2] = ncz - rz * nearW + uz * nearH
        // Near top-right
        fc[3] = ncx + rx * nearW + ux * nearH
        fc[4] = ncy + ry * nearW + uy * nearH
        fc[5] = ncz + rz * nearW + uz * nearH
        // Near bottom-left
        fc[6] = ncx - rx * nearW - ux * nearH
        fc[7] = ncy - ry * nearW - uy * nearH
        fc[8] = ncz - rz * nearW - uz * nearH
        // Near bottom-right
        fc[9] = ncx + rx * nearW - ux * nearH
        fc[10] = ncy + ry * nearW - uy * nearH
        fc[11] = ncz + rz * nearW - uz * nearH
        // Far top-left
        fc[12] = fcx - rx * farW + ux * farH
        fc[13] = fcy - ry * farW + uy * farH
        fc[14] = fcz - rz * farW + uz * farH
        // Far top-right
        fc[15] = fcx + rx * farW + ux * farH
        fc[16] = fcy + ry * farW + uy * farH
        fc[17] = fcz + rz * farW + uz * farH
        // Far bottom-left
        fc[18] = fcx - rx * farW - ux * farH
        fc[19] = fcy - ry * farW - uy * farH
        fc[20] = fcz - rz * farW - uz * farH
        // Far bottom-right
        fc[21] = fcx + rx * farW - ux * farH
        fc[22] = fcy + ry * farW - uy * farH
        fc[23] = fcz + rz * farW - uz * farH

        // Compute center of frustum slice
        let cx = 0,
          cy = 0,
          cz = 0
        for (let i = 0; i < 8; i++) {
          cx += fc[i * 3]!
          cy += fc[i * 3 + 1]!
          cz += fc[i * 3 + 2]!
        }
        cx /= 8
        cy /= 8
        cz /= 8

        // Light view: look from center - lightDir * backoff toward center
        const backoff = farDist - nearDist + 200
        const lEx = cx - this._lightDirNorm[0]! * backoff
        const lEy = cy - this._lightDirNorm[1]! * backoff
        const lEz = cz - this._lightDirNorm[2]! * backoff
        const lightEyeTmp = this._lightCorner
        lightEyeTmp[0] = lEx
        lightEyeTmp[1] = lEy
        lightEyeTmp[2] = lEz
        const centerTmp = this._cascadeCenter
        centerTmp[0] = cx
        centerTmp[1] = cy
        centerTmp[2] = cz
        m4LookAt(this._lightView, 0, lightEyeTmp, 0, centerTmp, 0, lightUp, 0)

        // Transform all 8 corners to light space and find AABB
        let minX = Infinity,
          maxX = -Infinity
        let minY = Infinity,
          maxY = -Infinity
        let minZ = Infinity,
          maxZ = -Infinity
        for (let i = 0; i < 8; i++) {
          const wx = fc[i * 3]!,
            wy = fc[i * 3 + 1]!,
            wz = fc[i * 3 + 2]!
          m4TransformPoint(this._lightCorner, 0, this._lightView, 0, wx, wy, wz)
          const lx = this._lightCorner[0]!,
            ly = this._lightCorner[1]!,
            lz = this._lightCorner[2]!
          if (lx < minX) minX = lx
          if (lx > maxX) maxX = lx
          if (ly < minY) minY = ly
          if (ly > maxY) maxY = ly
          if (lz < minZ) minZ = lz
          if (lz > maxZ) maxZ = lz
        }

        // Extend Z range to capture shadow casters behind the frustum
        minZ -= 200

        // Build ortho projection for this cascade
        // ortho() expects positive near/far distances; light-space Z is negative for objects
        // in front of the camera, so negate and swap: near = -maxZ, far = -minZ
        this._renderer.ortho(this._lightProj, 0, minX, maxX, minY, maxY, -maxZ, -minZ)

        // VP = proj * view
        m4Multiply(this._cascadeVPs, c * 16, this._lightProj, 0, this._lightView, 0)
      }

      rs.shadowCascadeVPs = this._cascadeVPs
      rs.shadowCascadeSplits = splits
      rs.shadowCascadeCount = numCascades
      rs.shadowBias = s.bias
    } else {
      rs.shadowCascadeCount = 0
    }

    this._renderer.render(rs)
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────

  destroy(): void {
    this._renderer.destroy()
  }
}

// ── Factory ─────────────────────────────────────────────────────────────────

export async function createScene(
  canvas: HTMLCanvasElement,
  opts?: { maxEntities?: number; maxSkinnedEntities?: number; backend?: BackendType },
): Promise<Scene> {
  const maxEntities = opts?.maxEntities ?? 10_000
  const renderer = await createRendererInternal(canvas, maxEntities, opts?.backend, opts?.maxSkinnedEntities)
  return new Scene(renderer, canvas, maxEntities, opts?.maxSkinnedEntities)
}
