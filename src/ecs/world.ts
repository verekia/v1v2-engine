const DEFAULT_MAX_ENTITIES = 10_000

// Component bitmask IDs
export const TRANSFORM = 1 << 0
export const MESH_INSTANCE = 1 << 1
export const CAMERA = 1 << 2
export const INPUT_RECEIVER = 1 << 3
export const SKINNED = 1 << 4
export const UNLIT = 1 << 5

export class World {
  readonly maxEntities: number
  entityCount = 0

  componentMask: Uint32Array

  // Transform (SoA)
  positions: Float32Array
  rotations: Float32Array
  scales: Float32Array
  worldMatrices: Float32Array

  // MeshInstance
  geometryIds: Uint8Array
  colors: Float32Array
  alphas: Float32Array

  // Skinned
  skinInstanceIds: Int16Array

  // Camera
  fovs: Float32Array
  nears: Float32Array
  fars: Float32Array
  viewMatrices: Float32Array
  projMatrices: Float32Array

  // Lighting (global)
  directionalLightDir = new Float32Array(3)
  directionalLightColor = new Float32Array(3)
  ambientLightColor = new Float32Array(3)

  activeCamera = -1

  // Free list for entity recycling (LIFO stack)
  private freeList: number[] = []

  constructor(maxEntities = DEFAULT_MAX_ENTITIES) {
    this.maxEntities = maxEntities

    this.componentMask = new Uint32Array(maxEntities)

    this.positions = new Float32Array(maxEntities * 3)
    this.rotations = new Float32Array(maxEntities * 3)
    this.scales = new Float32Array(maxEntities * 3)
    this.worldMatrices = new Float32Array(maxEntities * 16)

    this.geometryIds = new Uint8Array(maxEntities)
    this.colors = new Float32Array(maxEntities * 3)
    this.alphas = new Float32Array(maxEntities).fill(1)

    this.skinInstanceIds = new Int16Array(maxEntities).fill(-1)

    this.fovs = new Float32Array(maxEntities)
    this.nears = new Float32Array(maxEntities)
    this.fars = new Float32Array(maxEntities)
    this.viewMatrices = new Float32Array(maxEntities * 16)
    this.projMatrices = new Float32Array(maxEntities * 16)
  }

  createEntity(): number {
    if (this.freeList.length > 0) {
      return this.freeList.pop()!
    }
    return this.entityCount++
  }

  destroyEntity(e: number): void {
    this.componentMask[e] = 0
    this.skinInstanceIds[e] = -1
    this.alphas[e] = 1
    this.freeList.push(e)
  }

  addTransform(
    e: number,
    opts?: {
      position?: [number, number, number]
      rotation?: [number, number, number]
      scale?: [number, number, number]
    },
  ): void {
    this.componentMask[e]! |= TRANSFORM
    const p = opts?.position ?? [0, 0, 0]
    const r = opts?.rotation ?? [0, 0, 0]
    const s = opts?.scale ?? [1, 1, 1]
    this.positions.set(p, e * 3)
    this.rotations.set(r, e * 3)
    this.scales.set(s, e * 3)
  }

  addMeshInstance(e: number, opts: { geometryId: number; color: [number, number, number]; alpha?: number }): void {
    this.componentMask[e]! |= MESH_INSTANCE
    this.geometryIds[e] = opts.geometryId
    this.colors.set(opts.color, e * 3)
    this.alphas[e] = opts.alpha ?? 1
  }

  addCamera(e: number, opts: { fov: number; near: number; far: number }): void {
    this.componentMask[e]! |= CAMERA
    this.fovs[e] = opts.fov
    this.nears[e] = opts.near
    this.fars[e] = opts.far
  }

  addInputReceiver(e: number): void {
    this.componentMask[e]! |= INPUT_RECEIVER
  }

  addSkinned(e: number, skinInstanceId: number): void {
    this.componentMask[e]! |= SKINNED
    this.skinInstanceIds[e] = skinInstanceId
  }

  addUnlit(e: number): void {
    this.componentMask[e]! |= UNLIT
  }

  setDirectionalLight(dir: [number, number, number], color: [number, number, number]): void {
    this.directionalLightDir.set(dir)
    this.directionalLightColor.set(color)
  }

  setAmbientLight(color: [number, number, number]): void {
    this.ambientLightColor.set(color)
  }
}
