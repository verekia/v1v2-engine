import { m4FromQuatTRS, m4Multiply, v3Lerp, quatSlerp } from './math.ts'

import type { GltfSkin, GltfAnimation, GltfNodeTransform } from './gltf.ts'

const MAX_JOINTS = 128

export interface Skeleton {
  jointCount: number
  jointNodeIndices: number[]
  inverseBindMatrices: Float32Array
  nodeTransforms: GltfNodeTransform[]
  traversalOrder: number[] // topological order: parents before children
  // Pre-packed rest-pose arrays for fast bulk copy in sampleClip
  restTranslations: Float32Array // nodeCount * 3
  restRotations: Float32Array // nodeCount * 4
  restScales: Float32Array // nodeCount * 3
}

export interface SkinInstance {
  skeleton: Skeleton
  clipIndex: number
  time: number
  speed: number
  loop: boolean
  // Crossfade blending
  prevClipIndex: number
  prevTime: number
  prevSpeed: number
  prevLoop: boolean
  blendDuration: number
  blendElapsed: number
  // Pre-allocated scratch buffers
  localTranslations: Float32Array // nodeCount * 3
  localRotations: Float32Array // nodeCount * 4
  localScales: Float32Array // nodeCount * 3
  prevLocalTranslations: Float32Array // nodeCount * 3 (blend source)
  prevLocalRotations: Float32Array // nodeCount * 4 (blend source)
  prevLocalScales: Float32Array // nodeCount * 3 (blend source)
  globalMatrices: Float32Array // nodeCount * 16
  jointMatrices: Float32Array // jointCount * 16
}

// Module-level scratch mat4 to avoid aliasing during multiply
const _tempMat = new Float32Array(16)
const _localMat = new Float32Array(16)

export function findBoneNodeIndex(skeleton: Skeleton, boneName: string): number {
  for (const nt of skeleton.nodeTransforms) {
    if (nt.name === boneName) return nt.nodeIndex
  }
  throw new Error(`Bone "${boneName}" not found in skeleton`)
}

export function createSkeleton(skin: GltfSkin, nodeTransforms: GltfNodeTransform[]): Skeleton {
  // Compute topological traversal order (parents before children)
  // glTF doesn't guarantee parent nodes have lower indices than children
  const nodeCount = nodeTransforms.length
  const traversalOrder: number[] = []
  const visited = new Uint8Array(nodeCount)
  function visit(i: number) {
    if (visited[i]) return
    const parent = nodeTransforms[i]!.parentIndex
    if (parent >= 0) visit(parent)
    visited[i] = 1
    traversalOrder.push(i)
  }
  for (let i = 0; i < nodeCount; i++) visit(i)

  // Pre-pack rest-pose arrays for fast bulk copy in sampleClip
  const restTranslations = new Float32Array(nodeCount * 3)
  const restRotations = new Float32Array(nodeCount * 4)
  const restScales = new Float32Array(nodeCount * 3)
  for (let i = 0; i < nodeCount; i++) {
    const nt = nodeTransforms[i]!
    restTranslations[i * 3] = nt.translation[0]!
    restTranslations[i * 3 + 1] = nt.translation[1]!
    restTranslations[i * 3 + 2] = nt.translation[2]!
    restRotations[i * 4] = nt.rotation[0]!
    restRotations[i * 4 + 1] = nt.rotation[1]!
    restRotations[i * 4 + 2] = nt.rotation[2]!
    restRotations[i * 4 + 3] = nt.rotation[3]!
    restScales[i * 3] = nt.scale[0]!
    restScales[i * 3 + 1] = nt.scale[1]!
    restScales[i * 3 + 2] = nt.scale[2]!
  }

  return {
    jointCount: skin.jointNodeIndices.length,
    jointNodeIndices: skin.jointNodeIndices,
    inverseBindMatrices: skin.inverseBindMatrices,
    nodeTransforms,
    traversalOrder,
    restTranslations,
    restRotations,
    restScales,
  }
}

export function createSkinInstance(skeleton: Skeleton, clipIndex: number): SkinInstance {
  const nodeCount = skeleton.nodeTransforms.length
  const localTranslations = new Float32Array(nodeCount * 3)
  const localRotations = new Float32Array(nodeCount * 4)
  const localScales = new Float32Array(nodeCount * 3)
  const prevLocalTranslations = new Float32Array(nodeCount * 3)
  const prevLocalRotations = new Float32Array(nodeCount * 4)
  const prevLocalScales = new Float32Array(nodeCount * 3)
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
    prevClipIndex: clipIndex,
    prevTime: 0,
    prevSpeed: 1,
    prevLoop: true,
    blendDuration: 0,
    blendElapsed: 0,
    localTranslations,
    localRotations,
    localScales,
    prevLocalTranslations,
    prevLocalRotations,
    prevLocalScales,
    globalMatrices,
    jointMatrices,
  }
}

export function transitionTo(inst: SkinInstance, newClipIndex: number, duration: number, loop = true): void {
  if (inst.clipIndex === newClipIndex) return
  // Save current clip as the "from" state
  inst.prevClipIndex = inst.clipIndex
  inst.prevTime = inst.time
  inst.prevSpeed = inst.speed
  inst.prevLoop = inst.loop
  // Set up new clip
  inst.clipIndex = newClipIndex
  inst.time = 0
  inst.loop = loop
  // Start blend
  inst.blendDuration = duration
  inst.blendElapsed = 0
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

function sampleClip(
  clip: GltfAnimation,
  t: number,
  skeleton: Skeleton,
  outTranslations: Float32Array,
  outRotations: Float32Array,
  outScales: Float32Array,
): void {
  // Reset to rest pose (bulk copy from pre-packed arrays)
  outTranslations.set(skeleton.restTranslations)
  outRotations.set(skeleton.restRotations)
  outScales.set(skeleton.restScales)

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
        v3Lerp(outTranslations, ni * 3, vals, o0, vals, o1, frac)
      } else {
        outTranslations[ni * 3] = vals[o0]!
        outTranslations[ni * 3 + 1] = vals[o0 + 1]!
        outTranslations[ni * 3 + 2] = vals[o0 + 2]!
      }
    } else if (ch.path === 'rotation') {
      const o0 = ki * 4
      const o1 = (ki + 1) * 4
      if (t1 !== undefined) {
        quatSlerp(outRotations, ni * 4, vals, o0, vals, o1, frac)
      } else {
        outRotations[ni * 4] = vals[o0]!
        outRotations[ni * 4 + 1] = vals[o0 + 1]!
        outRotations[ni * 4 + 2] = vals[o0 + 2]!
        outRotations[ni * 4 + 3] = vals[o0 + 3]!
      }
    } else if (ch.path === 'scale') {
      const o0 = ki * 3
      const o1 = (ki + 1) * 3
      if (t1 !== undefined) {
        v3Lerp(outScales, ni * 3, vals, o0, vals, o1, frac)
      } else {
        outScales[ni * 3] = vals[o0]!
        outScales[ni * 3 + 1] = vals[o0 + 1]!
        outScales[ni * 3 + 2] = vals[o0 + 2]!
      }
    }
  }
}

function advanceTime(time: number, dt: number, speed: number, loop: boolean, duration: number): number {
  time += dt * speed
  if (loop) {
    if (duration > 0) time = time % duration
  } else {
    if (time > duration) time = duration
  }
  return time
}

export function updateSkinInstance(inst: SkinInstance, clips: GltfAnimation[], dt: number): void {
  const clip = clips[inst.clipIndex]!
  const skeleton = inst.skeleton
  const nodeCount = skeleton.nodeTransforms.length
  const blending = inst.blendDuration > 0 && inst.blendElapsed < inst.blendDuration

  // Advance current clip time
  inst.time = advanceTime(inst.time, dt, inst.speed, inst.loop, clip.duration)

  // Sample current clip into localTranslations/Rotations/Scales
  sampleClip(clip, inst.time, skeleton, inst.localTranslations, inst.localRotations, inst.localScales)

  if (blending) {
    const prevClip = clips[inst.prevClipIndex]!

    // Advance previous clip time
    inst.prevTime = advanceTime(inst.prevTime, dt, inst.prevSpeed, inst.prevLoop, prevClip.duration)

    // Sample previous clip into prev buffers
    sampleClip(
      prevClip,
      inst.prevTime,
      skeleton,
      inst.prevLocalTranslations,
      inst.prevLocalRotations,
      inst.prevLocalScales,
    )

    // Advance blend
    inst.blendElapsed += dt
    let alpha = inst.blendElapsed / inst.blendDuration
    if (alpha >= 1) {
      alpha = 1
      inst.blendDuration = 0
    }

    // Blend: lerp translations and scales, slerp rotations (prev → current)
    for (let i = 0; i < nodeCount; i++) {
      v3Lerp(inst.localTranslations, i * 3, inst.prevLocalTranslations, i * 3, inst.localTranslations, i * 3, alpha)
      quatSlerp(inst.localRotations, i * 4, inst.prevLocalRotations, i * 4, inst.localRotations, i * 4, alpha)
      v3Lerp(inst.localScales, i * 3, inst.prevLocalScales, i * 3, inst.localScales, i * 3, alpha)
    }
  }

  // Compute global matrices in topological order (parents before children)
  const order = skeleton.traversalOrder
  for (let k = 0; k < order.length; k++) {
    const i = order[k]!
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
