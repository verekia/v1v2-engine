import { createRenderer as createRendererInternal } from './gpu.ts'
import { m4FromTRS, m4LookAt, m4Multiply, v3Normalize, v3Scale } from './math.ts'

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
  }
}

// ── Geometry registration (internal) ────────────────────────────────────────

type GeoReg =
  | { vertices: Float32Array; indices: Uint16Array | Uint32Array; skinned: false }
  | {
      vertices: Float32Array
      indices: Uint16Array | Uint32Array
      joints: Uint8Array
      weights: Float32Array
      skinned: true
    }

// ── Scene ───────────────────────────────────────────────────────────────────

export class Scene {
  readonly camera = new Camera()
  readonly shadow: ShadowConfig = createDefaultShadow()
  readonly lightDirection = new Float32Array(3)
  readonly lightDirColor = new Float32Array(3)
  readonly lightAmbientColor = new Float32Array(3)
  readonly skinInstances: SkinInstance[] = []

  private _meshes: Mesh[] = []
  private _renderer: IRenderer
  private _canvas: HTMLCanvasElement
  private _maxEntities: number
  private _geoRegs = new Map<number, GeoReg>()
  private _nextGeoId = 0

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

  // View/projection matrices (written during render)
  private _viewMatrix = new Float32Array(16)
  private _projMatrix = new Float32Array(16)

  // Shadow scratch buffers
  private _lightDirNorm = new Float32Array(3)
  private _lightEye = new Float32Array(3)
  private _lightView = new Float32Array(16)
  private _lightProj = new Float32Array(16)
  private _lightVP = new Float32Array(16)

  /** @internal — use createScene() instead */
  constructor(renderer: IRenderer, canvas: HTMLCanvasElement, maxEntities: number) {
    this._renderer = renderer
    this._canvas = canvas
    this._maxEntities = maxEntities
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

  // ── Geometry management ─────────────────────────────────────────────────

  registerGeometry(vertices: Float32Array, indices: Uint16Array | Uint32Array): number {
    const id = this._nextGeoId++
    this._renderer.registerGeometry(id, vertices, indices)
    this._geoRegs.set(id, { vertices, indices, skinned: false })
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
    this._geoRegs.set(id, { vertices, indices, joints, weights, skinned: true })
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

  // ── Backend switching ───────────────────────────────────────────────────

  async switchBackend(canvas: HTMLCanvasElement, type: BackendType): Promise<void> {
    this._renderer.destroy()
    this._renderer = await createRendererInternal(canvas, this._maxEntities, type)
    this._canvas = canvas

    // Re-register all geometries on the new renderer
    for (const [id, reg] of this._geoRegs) {
      if (reg.skinned) {
        this._renderer.registerSkinnedGeometry(id, reg.vertices, reg.indices, reg.joints, reg.weights)
      } else {
        this._renderer.registerGeometry(id, reg.vertices, reg.indices)
      }
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

      this._renderMask[i] = m.visible ? 1 : 0
      this._skinnedMask[i] = m.skinned ? 1 : 0
      this._unlitMask[i] = m.unlit ? 1 : 0
    }

    // Build RenderScene
    const rs: RenderScene = {
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
    }

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
  opts?: { maxEntities?: number; backend?: BackendType },
): Promise<Scene> {
  const maxEntities = opts?.maxEntities ?? 10_000
  const renderer = await createRendererInternal(canvas, maxEntities, opts?.backend)
  return new Scene(renderer, canvas, maxEntities)
}
