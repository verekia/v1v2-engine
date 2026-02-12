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
  v3Scale,
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
  /** Center of the shadow volume */
  target: Float32Array
  /** How far the virtual light source is from the target */
  distance: number
  /** Half-size of the orthographic shadow frustum */
  extent: number
  near: number
  far: number
  bias: number
}

function createDefaultShadow(): ShadowConfig {
  return {
    enabled: false,
    target: new Float32Array([0, 0, 0]),
    distance: 400,
    extent: 150,
    near: 1,
    far: 800,
    bias: 0.0001,
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

  // Shadow scratch buffers
  private _lightDirNorm = new Float32Array(3)
  private _lightEye = new Float32Array(3)
  private _lightView = new Float32Array(16)
  private _lightProj = new Float32Array(16)
  private _lightVP = new Float32Array(16)

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

    // Compute shadow VP if enabled
    if (this.shadow.enabled) {
      const s = this.shadow
      v3Normalize(this._lightDirNorm, 0, this.lightDirection, 0)
      v3Scale(this._lightEye, 0, this._lightDirNorm, 0, -s.distance)
      this._lightEye[0]! += s.target[0]!
      this._lightEye[1]! += s.target[1]!
      this._lightEye[2]! += s.target[2]!
      m4LookAt(this._lightView, 0, this._lightEye, 0, s.target, 0, this.camera.up, 0)
      this._renderer.ortho(this._lightProj, 0, -s.extent, s.extent, -s.extent, s.extent, s.near, s.far)
      m4Multiply(this._lightVP, 0, this._lightProj, 0, this._lightView, 0)
      rs.shadowLightViewProj = this._lightVP
      rs.shadowBias = s.bias
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
