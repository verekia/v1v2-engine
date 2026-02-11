const MAX_ENTITIES = 10_000

// Component bitmask IDs
export const TRANSFORM = 1 << 0
export const MESH_INSTANCE = 1 << 1
export const CAMERA = 1 << 2
export const INPUT_RECEIVER = 1 << 3
export const SKINNED = 1 << 4
export const UNLIT = 1 << 5

export class World {
  entityCount = 0
  componentMask = new Uint32Array(MAX_ENTITIES)

  // Transform (SoA)
  positions = new Float32Array(MAX_ENTITIES * 3)
  rotations = new Float32Array(MAX_ENTITIES * 3)
  scales = new Float32Array(MAX_ENTITIES * 3)
  worldMatrices = new Float32Array(MAX_ENTITIES * 16)

  // MeshInstance
  geometryIds = new Uint8Array(MAX_ENTITIES)
  colors = new Float32Array(MAX_ENTITIES * 3)
  alphas = new Float32Array(MAX_ENTITIES).fill(1)

  // Skinned
  skinInstanceIds = new Int16Array(MAX_ENTITIES).fill(-1)

  // Camera
  fovs = new Float32Array(MAX_ENTITIES)
  nears = new Float32Array(MAX_ENTITIES)
  fars = new Float32Array(MAX_ENTITIES)
  viewMatrices = new Float32Array(MAX_ENTITIES * 16)
  projMatrices = new Float32Array(MAX_ENTITIES * 16)

  // Lighting (global)
  directionalLightDir = new Float32Array(3)
  directionalLightColor = new Float32Array(3)
  ambientLightColor = new Float32Array(3)

  activeCamera = -1

  createEntity(): number {
    return this.entityCount++
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
