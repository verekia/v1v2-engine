export const MODEL_SLOT_SIZE = 256 // minUniformBufferOffsetAlignment
export const MAX_JOINTS = 128
export const JOINT_SLOT_SIZE = MAX_JOINTS * 64 // 128 mat4 * 64 bytes = 8192 (already 256-aligned)
export const MAX_SKINNED_ENTITIES = 64
export const SHADOW_MAP_SIZE = 2048
export const MSAA_SAMPLES = 4
