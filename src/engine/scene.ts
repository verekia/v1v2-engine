import { m4FromTRS } from './math.ts'

import type { RenderScene } from './renderer.ts'
import type { SkinInstance } from './skin.ts'

export interface MeshOptions {
  geometryId: number
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
  geometryId: number
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
    this.geometryId = opts.geometryId
    this.unlit = opts.unlit ?? false
    this.skinned = opts.skinned ?? false
    this.skinInstanceId = opts.skinInstanceId ?? -1
  }
}

export interface ShadowOptions {
  lightViewProj: Float32Array
  bias?: number
  normalBias?: number
  mapSize?: number
}

export class Scene {
  readonly viewMatrix = new Float32Array(16)
  readonly projMatrix = new Float32Array(16)
  readonly lightDirection = new Float32Array(3)
  readonly lightDirColor = new Float32Array(3)
  readonly lightAmbientColor = new Float32Array(3)
  readonly skinInstances: SkinInstance[] = []

  private _meshes: Mesh[] = []

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

  constructor(maxEntities = 10_000) {
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

  get meshes(): readonly Mesh[] {
    return this._meshes
  }

  add(mesh: Mesh): Mesh {
    this._meshes.push(mesh)
    return mesh
  }

  remove(mesh: Mesh): void {
    const idx = this._meshes.indexOf(mesh)
    if (idx >= 0) this._meshes.splice(idx, 1)
  }

  setDirectionalLight(dir: [number, number, number], color: [number, number, number]): void {
    this.lightDirection.set(dir)
    this.lightDirColor.set(color)
  }

  setAmbientLight(color: [number, number, number]): void {
    this.lightAmbientColor.set(color)
  }

  /** Sync mesh data to SoA arrays, compute world matrices, and return a RenderScene. */
  buildRenderScene(shadow?: ShadowOptions): RenderScene {
    const meshes = this._meshes
    const count = meshes.length

    for (let i = 0; i < count; i++) {
      const m = meshes[i]!
      const i3 = i * 3

      // Position + scale (used by frustum culling)
      this._positions[i3] = m.position[0]!
      this._positions[i3 + 1] = m.position[1]!
      this._positions[i3 + 2] = m.position[2]!
      this._scales[i3] = m.scale[0]!
      this._scales[i3 + 1] = m.scale[1]!
      this._scales[i3 + 2] = m.scale[2]!

      // World matrix from TRS
      m4FromTRS(this._worldMatrices, i * 16, m.position, 0, m.rotation, 0, m.scale, 0)

      // Color + alpha
      this._colors[i3] = m.color[0]!
      this._colors[i3 + 1] = m.color[1]!
      this._colors[i3 + 2] = m.color[2]!
      this._alphas[i] = m.alpha

      // Geometry + skin IDs
      this._geometryIds[i] = m.geometryId
      this._skinInstanceIds[i] = m.skinInstanceId

      // Masks
      this._renderMask[i] = m.visible ? 1 : 0
      this._skinnedMask[i] = m.skinned ? 1 : 0
      this._unlitMask[i] = m.unlit ? 1 : 0
    }

    const rs: RenderScene = {
      cameraView: this.viewMatrix,
      cameraViewOffset: 0,
      cameraProj: this.projMatrix,
      cameraProjOffset: 0,
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

    if (shadow) {
      rs.shadowLightViewProj = shadow.lightViewProj
      rs.shadowBias = shadow.bias
      rs.shadowNormalBias = shadow.normalBias
      rs.shadowMapSize = shadow.mapSize
    }

    return rs
  }
}
