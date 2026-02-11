import { m4FromQuatTRS, m4Multiply, v3Lerp, quatSlerp } from './math.ts'

import type { GltfSkin, GltfAnimation, GltfNodeTransform } from './gltf.ts'

const MAX_JOINTS = 128

export interface Skeleton {
  jointCount: number
  jointNodeIndices: number[]
  inverseBindMatrices: Float32Array
  nodeTransforms: GltfNodeTransform[]
}

export interface SkinInstance {
  skeleton: Skeleton
  clipIndex: number
  time: number
  speed: number
  loop: boolean
  // Pre-allocated scratch buffers
  localTranslations: Float32Array // nodeCount * 3
  localRotations: Float32Array // nodeCount * 4
  localScales: Float32Array // nodeCount * 3
  globalMatrices: Float32Array // nodeCount * 16
  jointMatrices: Float32Array // jointCount * 16
}

// Module-level scratch mat4 to avoid aliasing during multiply
const _tempMat = new Float32Array(16)
const _localMat = new Float32Array(16)

export function createSkeleton(skin: GltfSkin, nodeTransforms: GltfNodeTransform[]): Skeleton {
  return {
    jointCount: skin.jointNodeIndices.length,
    jointNodeIndices: skin.jointNodeIndices,
    inverseBindMatrices: skin.inverseBindMatrices,
    nodeTransforms,
  }
}

export function createSkinInstance(skeleton: Skeleton, clipIndex: number): SkinInstance {
  const nodeCount = skeleton.nodeTransforms.length
  const localTranslations = new Float32Array(nodeCount * 3)
  const localRotations = new Float32Array(nodeCount * 4)
  const localScales = new Float32Array(nodeCount * 3)
  const globalMatrices = new Float32Array(nodeCount * 16)
  const jointMatrices = new Float32Array(MAX_JOINTS * 16)

  // Copy rest poses
  for (let i = 0; i < nodeCount; i++) {
    const nt = skeleton.nodeTransforms[i]!
    localTranslations[i * 3] = nt.translation[0]!
    localTranslations[i * 3 + 1] = nt.translation[1]!
    localTranslations[i * 3 + 2] = nt.translation[2]!
    localRotations[i * 4] = nt.rotation[0]!
    localRotations[i * 4 + 1] = nt.rotation[1]!
    localRotations[i * 4 + 2] = nt.rotation[2]!
    localRotations[i * 4 + 3] = nt.rotation[3]!
    localScales[i * 3] = nt.scale[0]!
    localScales[i * 3 + 1] = nt.scale[1]!
    localScales[i * 3 + 2] = nt.scale[2]!
  }

  return {
    skeleton,
    clipIndex,
    time: 0,
    speed: 1,
    loop: true,
    localTranslations,
    localRotations,
    localScales,
    globalMatrices,
    jointMatrices,
  }
}

function binarySearchKeyframe(times: Float32Array, t: number): number {
  let lo = 0
  let hi = times.length - 1
  if (t <= times[lo]!) return 0
  if (t >= times[hi]!) return hi - 1
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1
    if (times[mid]! <= t) lo = mid
    else hi = mid
  }
  return lo
}

export function updateSkinInstance(inst: SkinInstance, clips: GltfAnimation[], dt: number): void {
  const clip = clips[inst.clipIndex]!
  const skeleton = inst.skeleton

  // Advance time
  inst.time += dt * inst.speed
  if (inst.loop) {
    if (clip.duration > 0) {
      inst.time = inst.time % clip.duration
    }
  } else {
    if (inst.time > clip.duration) inst.time = clip.duration
  }
  const t = inst.time

  // Reset local transforms to rest pose
  const nodeCount = skeleton.nodeTransforms.length
  for (let i = 0; i < nodeCount; i++) {
    const nt = skeleton.nodeTransforms[i]!
    inst.localTranslations[i * 3] = nt.translation[0]!
    inst.localTranslations[i * 3 + 1] = nt.translation[1]!
    inst.localTranslations[i * 3 + 2] = nt.translation[2]!
    inst.localRotations[i * 4] = nt.rotation[0]!
    inst.localRotations[i * 4 + 1] = nt.rotation[1]!
    inst.localRotations[i * 4 + 2] = nt.rotation[2]!
    inst.localRotations[i * 4 + 3] = nt.rotation[3]!
    inst.localScales[i * 3] = nt.scale[0]!
    inst.localScales[i * 3 + 1] = nt.scale[1]!
    inst.localScales[i * 3 + 2] = nt.scale[2]!
  }

  // Sample each animation channel
  for (const ch of clip.channels) {
    const ni = ch.targetNodeIndex
    const times = ch.inputTimes
    const vals = ch.outputValues
    if (times.length === 0) continue

    const ki = binarySearchKeyframe(times, t)
    const t0 = times[ki]!
    const t1 = times[ki + 1]
    const frac = t1 !== undefined && t1 > t0 ? (t - t0) / (t1 - t0) : 0

    if (ch.path === 'translation') {
      const o0 = ki * 3
      const o1 = (ki + 1) * 3
      if (t1 !== undefined) {
        v3Lerp(inst.localTranslations, ni * 3, vals, o0, vals, o1, frac)
      } else {
        inst.localTranslations[ni * 3] = vals[o0]!
        inst.localTranslations[ni * 3 + 1] = vals[o0 + 1]!
        inst.localTranslations[ni * 3 + 2] = vals[o0 + 2]!
      }
    } else if (ch.path === 'rotation') {
      const o0 = ki * 4
      const o1 = (ki + 1) * 4
      if (t1 !== undefined) {
        quatSlerp(inst.localRotations, ni * 4, vals, o0, vals, o1, frac)
      } else {
        inst.localRotations[ni * 4] = vals[o0]!
        inst.localRotations[ni * 4 + 1] = vals[o0 + 1]!
        inst.localRotations[ni * 4 + 2] = vals[o0 + 2]!
        inst.localRotations[ni * 4 + 3] = vals[o0 + 3]!
      }
    } else if (ch.path === 'scale') {
      const o0 = ki * 3
      const o1 = (ki + 1) * 3
      if (t1 !== undefined) {
        v3Lerp(inst.localScales, ni * 3, vals, o0, vals, o1, frac)
      } else {
        inst.localScales[ni * 3] = vals[o0]!
        inst.localScales[ni * 3 + 1] = vals[o0 + 1]!
        inst.localScales[ni * 3 + 2] = vals[o0 + 2]!
      }
    }
  }

  // Compute global matrices parent-first (nodes are already in order from glTF)
  for (let i = 0; i < nodeCount; i++) {
    const nt = skeleton.nodeTransforms[i]!
    m4FromQuatTRS(_localMat, 0, inst.localTranslations, i * 3, inst.localRotations, i * 4, inst.localScales, i * 3)
    if (nt.parentIndex >= 0) {
      // Copy parent global to temp, then multiply: global = parent * local
      const po = nt.parentIndex * 16
      for (let j = 0; j < 16; j++) _tempMat[j] = inst.globalMatrices[po + j]!
      m4Multiply(inst.globalMatrices, i * 16, _tempMat, 0, _localMat, 0)
    } else {
      // Root node: global = local
      for (let j = 0; j < 16; j++) inst.globalMatrices[i * 16 + j] = _localMat[j]!
    }
  }

  // Compute joint matrices: jointMatrix[j] = globalMatrix[jointNode[j]] * inverseBindMatrix[j]
  const jointCount = skeleton.jointCount
  for (let j = 0; j < jointCount; j++) {
    const nodeIdx = skeleton.jointNodeIndices[j]!
    m4Multiply(inst.jointMatrices, j * 16, inst.globalMatrices, nodeIdx * 16, skeleton.inverseBindMatrices, j * 16)
  }
}
