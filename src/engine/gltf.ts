// GLB/glTF loader with Draco mesh compression support

import { quatToEulerZXY } from './math.ts'

export interface GltfMesh {
  name: string
  vertices: Float32Array // interleaved [px,py,pz, nx,ny,nz, cr,cg,cb]
  indices: Uint16Array | Uint32Array
  position: [number, number, number]
  rotation: [number, number, number] // euler ZXY
  scale: [number, number, number]
  color: [number, number, number]
  skinJoints?: Uint8Array
  skinWeights?: Float32Array
  skinIndex?: number
}

export interface GltfSkin {
  jointNodeIndices: number[]
  inverseBindMatrices: Float32Array
}

export interface GltfAnimationChannel {
  targetNodeIndex: number
  path: 'translation' | 'rotation' | 'scale'
  interpolation: string
  inputTimes: Float32Array
  outputValues: Float32Array
}

export interface GltfAnimation {
  name: string
  channels: GltfAnimationChannel[]
  duration: number
}

export interface GltfNodeTransform {
  nodeIndex: number
  parentIndex: number // -1 for root
  translation: Float32Array // vec3
  rotation: Float32Array // quat xyzw
  scale: Float32Array // vec3
}

export interface GlbResult {
  meshes: GltfMesh[]
  skins: GltfSkin[]
  animations: GltfAnimation[]
  nodeTransforms: GltfNodeTransform[]
}

type ColorMap = Map<number, [number, number, number]>

// Draco decoder singleton
let dracoPromise: Promise<any> | null = null

function loadDracoDecoder(decoderPath: string): Promise<any> {
  if (dracoPromise) return dracoPromise
  dracoPromise = new Promise<any>((resolve, reject) => {
    const script = document.createElement('script')
    script.src = decoderPath + 'draco_decoder_gltf.js'
    script.onload = () => {
      const factory = (globalThis as any).DracoDecoderModule
      factory({ locateFile: (f: string) => decoderPath + f })
        .then(resolve)
        .catch(reject)
    }
    script.onerror = () => reject(new Error('Failed to load Draco decoder'))
    document.head.appendChild(script)
  })
  return dracoPromise
}

export async function loadGlb(
  url: string,
  dracoDecoderPath = '/draco-1.5.7/',
  materialColors?: ColorMap,
): Promise<GlbResult> {
  const buf = await (await fetch(url)).arrayBuffer()
  const view = new DataView(buf)

  // Parse GLB header
  const magic = view.getUint32(0, true)
  if (magic !== 0x46546c67) throw new Error('Not a GLB file')

  // Read chunks
  let offset = 12
  let gltf: any = null
  let binChunk: ArrayBuffer | null = null

  while (offset < buf.byteLength) {
    const chunkLength = view.getUint32(offset, true)
    const chunkType = view.getUint32(offset + 4, true)
    const chunkData = buf.slice(offset + 8, offset + 8 + chunkLength)

    if (chunkType === 0x4e4f534a) {
      // JSON
      gltf = JSON.parse(new TextDecoder().decode(chunkData))
    } else if (chunkType === 0x004e4942) {
      // BIN
      binChunk = chunkData
    }

    offset += 8 + chunkLength
  }

  if (!gltf || !binChunk) throw new Error('Missing GLB chunks')

  const bin = new Uint8Array(binChunk)

  // Load Draco decoder if needed
  const usesDraco = gltf.extensionsUsed?.includes('KHR_draco_mesh_compression')
  let draco: any = null
  if (usesDraco) {
    draco = await loadDracoDecoder(dracoDecoderPath)
  }

  // ── Build node hierarchy ──────────────────────────────────────────
  const nodes = gltf.nodes ?? []
  const nodeTransforms: GltfNodeTransform[] = []
  const childToParent = new Map<number, number>()

  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni]
    if (node.children) {
      for (const childIdx of node.children) {
        childToParent.set(childIdx, ni)
      }
    }
  }

  for (let ni = 0; ni < nodes.length; ni++) {
    const node = nodes[ni]
    const t = new Float32Array(3)
    const r = new Float32Array([0, 0, 0, 1])
    const s = new Float32Array([1, 1, 1])
    if (node.translation) {
      t[0] = node.translation[0]
      t[1] = node.translation[1]
      t[2] = node.translation[2]
    }
    if (node.rotation) {
      r[0] = node.rotation[0]
      r[1] = node.rotation[1]
      r[2] = node.rotation[2]
      r[3] = node.rotation[3]
    }
    if (node.scale) {
      s[0] = node.scale[0]
      s[1] = node.scale[1]
      s[2] = node.scale[2]
    }
    nodeTransforms.push({
      nodeIndex: ni,
      parentIndex: childToParent.get(ni) ?? -1,
      translation: t,
      rotation: r,
      scale: s,
    })
  }

  // ── Parse skins ───────────────────────────────────────────────────
  const skins: GltfSkin[] = []
  if (gltf.skins) {
    for (const skin of gltf.skins) {
      const jointNodeIndices: number[] = skin.joints
      let inverseBindMatrices = new Float32Array(jointNodeIndices.length * 16)
      if (skin.inverseBindMatrices !== undefined) {
        const accessor = gltf.accessors[skin.inverseBindMatrices]
        inverseBindMatrices = new Float32Array(readAccessorFloat32(accessor, gltf, bin))
      } else {
        // Identity matrices
        for (let j = 0; j < jointNodeIndices.length; j++) {
          const o = j * 16
          inverseBindMatrices[o] = 1
          inverseBindMatrices[o + 5] = 1
          inverseBindMatrices[o + 10] = 1
          inverseBindMatrices[o + 15] = 1
        }
      }
      skins.push({ jointNodeIndices, inverseBindMatrices })
    }
  }

  // ── Parse animations ──────────────────────────────────────────────
  const animations: GltfAnimation[] = []
  if (gltf.animations) {
    for (const anim of gltf.animations) {
      const channels: GltfAnimationChannel[] = []
      let duration = 0
      for (const ch of anim.channels) {
        const sampler = anim.samplers[ch.sampler]
        const inputAccessor = gltf.accessors[sampler.input]
        const outputAccessor = gltf.accessors[sampler.output]
        const inputTimes = readAccessorFloat32(inputAccessor, gltf, bin)
        const outputValues = readAccessorFloat32(outputAccessor, gltf, bin)
        const maxTime = inputTimes[inputTimes.length - 1]!
        if (maxTime > duration) duration = maxTime
        channels.push({
          targetNodeIndex: ch.target.node,
          path: ch.target.path,
          interpolation: sampler.interpolation ?? 'LINEAR',
          inputTimes,
          outputValues,
        })
      }
      animations.push({ name: anim.name ?? '', channels, duration })
    }
  }

  // ── Traverse scene and collect meshes ─────────────────────────────
  const meshes: GltfMesh[] = []

  // Scratch buffers for quaternion conversion
  const qBuf = new Float32Array(4)
  const rBuf = new Float32Array(3)

  function traverseNode(nodeIdx: number): void {
    const node = gltf.nodes[nodeIdx]

    // Get node transform (TRS)
    let position: [number, number, number] = [0, 0, 0]
    let rotation: [number, number, number] = [0, 0, 0]
    let scale: [number, number, number] = [1, 1, 1]

    if (node.translation) {
      position = [node.translation[0], node.translation[1], node.translation[2]]
    }
    if (node.rotation) {
      qBuf[0] = node.rotation[0]
      qBuf[1] = node.rotation[1]
      qBuf[2] = node.rotation[2]
      qBuf[3] = node.rotation[3]
      quatToEulerZXY(rBuf, 0, qBuf, 0)
      rotation = [rBuf[0]!, rBuf[1]!, rBuf[2]!]
    }
    if (node.scale) {
      scale = [node.scale[0], node.scale[1], node.scale[2]]
    }

    // Process mesh
    if (node.mesh !== undefined) {
      const mesh = gltf.meshes[node.mesh]
      const baseName = mesh.name ?? `mesh_${node.mesh}`
      const prims = mesh.primitives
      const hasSkin = node.skin !== undefined
      for (let pi = 0; pi < prims.length; pi++) {
        const primitive = prims[pi]
        const result = decodePrimitive(primitive, gltf, bin, draco, materialColors)
        if (!result) continue

        // Get material color
        let color: [number, number, number] = [1, 1, 1]
        if (primitive.material !== undefined) {
          const mat = gltf.materials?.[primitive.material]
          const bcf = mat?.pbrMetallicRoughness?.baseColorFactor
          if (bcf) {
            color = [bcf[0], bcf[1], bcf[2]]
          }
        }

        const name = prims.length > 1 ? `${baseName}_${pi + 1}` : baseName
        const entry: GltfMesh = {
          name,
          vertices: result.vertices,
          indices: result.indices,
          position,
          rotation,
          scale,
          color,
        }

        if (hasSkin && result.skinJoints && result.skinWeights) {
          entry.skinJoints = result.skinJoints
          entry.skinWeights = result.skinWeights
          entry.skinIndex = node.skin
        }

        meshes.push(entry)
      }
    }

    // Recurse children
    if (node.children) {
      for (const childIdx of node.children) {
        traverseNode(childIdx)
      }
    }
  }

  const sceneIdx = gltf.scene ?? 0
  const scene = gltf.scenes[sceneIdx]
  for (const nodeIdx of scene.nodes) {
    traverseNode(nodeIdx)
  }

  return { meshes, skins, animations, nodeTransforms }
}

interface PrimitiveResult {
  vertices: Float32Array
  indices: Uint16Array | Uint32Array
  skinJoints?: Uint8Array
  skinWeights?: Float32Array
}

function decodePrimitive(
  primitive: any,
  gltf: any,
  bin: Uint8Array,
  draco: any,
  materialColors?: ColorMap,
): PrimitiveResult | null {
  const dracoExt = primitive.extensions?.KHR_draco_mesh_compression
  if (dracoExt && draco) {
    return decodeDracoPrimitive(dracoExt, primitive, gltf, bin, draco, materialColors)
  }
  return decodeStandardPrimitive(primitive, gltf, bin)
}

// Find the _materialindex key in the Draco attributes map (case-insensitive)
function findMaterialIndexKey(attributes: Record<string, number>): string | undefined {
  return Object.keys(attributes).find(k => k.toLowerCase() === '_materialindex')
}

function decodeDracoPrimitive(
  dracoExt: any,
  primitive: any,
  gltf: any,
  bin: Uint8Array,
  draco: any,
  materialColors?: ColorMap,
): PrimitiveResult | null {
  const bv = gltf.bufferViews[dracoExt.bufferView]
  const byteOffset = bv.byteOffset ?? 0
  const compressedData = bin.subarray(byteOffset, byteOffset + bv.byteLength)

  const decoder = new draco.Decoder()
  const buffer = new draco.DecoderBuffer()
  buffer.Init(compressedData, compressedData.length)

  const geometryType = decoder.GetEncodedGeometryType(buffer)
  if (geometryType !== draco.TRIANGULAR_MESH) {
    draco.destroy(buffer)
    draco.destroy(decoder)
    return null
  }

  const dracoMesh = new draco.Mesh()
  const status = decoder.DecodeBufferToMesh(buffer, dracoMesh)
  if (!status.ok()) {
    draco.destroy(dracoMesh)
    draco.destroy(buffer)
    draco.destroy(decoder)
    return null
  }

  const numVertices = dracoMesh.num_points()
  const numFaces = dracoMesh.num_faces()

  // Read positions
  const posAttr = decoder.GetAttributeByUniqueId(dracoMesh, dracoExt.attributes.POSITION)
  const posArray = new draco.DracoFloat32Array()
  decoder.GetAttributeFloatForAllPoints(dracoMesh, posAttr, posArray)

  // Read normals (if available)
  let normArray: any = null
  if (dracoExt.attributes.NORMAL !== undefined) {
    const normAttr = decoder.GetAttributeByUniqueId(dracoMesh, dracoExt.attributes.NORMAL)
    normArray = new draco.DracoFloat32Array()
    decoder.GetAttributeFloatForAllPoints(dracoMesh, normAttr, normArray)
  }

  // Read _materialindex (if available and color map provided)
  let matIdxArray: any = null
  const matIdxKey = findMaterialIndexKey(dracoExt.attributes)
  if (matIdxKey && materialColors) {
    const matIdxAttr = decoder.GetAttributeByUniqueId(dracoMesh, dracoExt.attributes[matIdxKey])
    matIdxArray = new draco.DracoFloat32Array()
    decoder.GetAttributeFloatForAllPoints(dracoMesh, matIdxAttr, matIdxArray)
  }

  // Read JOINTS_0 / WEIGHTS_0 from Draco (if present)
  let skinJoints: Uint8Array | undefined
  let skinWeights: Float32Array | undefined
  if (dracoExt.attributes.JOINTS_0 !== undefined && dracoExt.attributes.WEIGHTS_0 !== undefined) {
    const jointsAttr = decoder.GetAttributeByUniqueId(dracoMesh, dracoExt.attributes.JOINTS_0)
    const jointsArray = new draco.DracoFloat32Array()
    decoder.GetAttributeFloatForAllPoints(dracoMesh, jointsAttr, jointsArray)

    const weightsAttr = decoder.GetAttributeByUniqueId(dracoMesh, dracoExt.attributes.WEIGHTS_0)
    const weightsArray = new draco.DracoFloat32Array()
    decoder.GetAttributeFloatForAllPoints(dracoMesh, weightsAttr, weightsArray)

    skinJoints = new Uint8Array(numVertices * 4)
    skinWeights = new Float32Array(numVertices * 4)
    for (let i = 0; i < numVertices; i++) {
      skinJoints[i * 4] = jointsArray.GetValue(i * 4)
      skinJoints[i * 4 + 1] = jointsArray.GetValue(i * 4 + 1)
      skinJoints[i * 4 + 2] = jointsArray.GetValue(i * 4 + 2)
      skinJoints[i * 4 + 3] = jointsArray.GetValue(i * 4 + 3)
      skinWeights[i * 4] = weightsArray.GetValue(i * 4)
      skinWeights[i * 4 + 1] = weightsArray.GetValue(i * 4 + 1)
      skinWeights[i * 4 + 2] = weightsArray.GetValue(i * 4 + 2)
      skinWeights[i * 4 + 3] = weightsArray.GetValue(i * 4 + 3)
    }

    draco.destroy(jointsArray)
    draco.destroy(weightsArray)
  }

  // Interleave [px, py, pz, nx, ny, nz, cr, cg, cb]
  const vertices = new Float32Array(numVertices * 9)
  for (let i = 0; i < numVertices; i++) {
    const vi = i * 9
    vertices[vi] = posArray.GetValue(i * 3)
    vertices[vi + 1] = posArray.GetValue(i * 3 + 1)
    vertices[vi + 2] = posArray.GetValue(i * 3 + 2)
    if (normArray) {
      vertices[vi + 3] = normArray.GetValue(i * 3)
      vertices[vi + 4] = normArray.GetValue(i * 3 + 1)
      vertices[vi + 5] = normArray.GetValue(i * 3 + 2)
    } else {
      vertices[vi + 5] = 1 // default up normal (+Z)
    }
    // Vertex color from material index
    if (matIdxArray && materialColors) {
      const idx = Math.round(matIdxArray.GetValue(i))
      const c = materialColors.get(idx) ?? [1, 1, 1]
      vertices[vi + 6] = c[0]
      vertices[vi + 7] = c[1]
      vertices[vi + 8] = c[2]
    } else {
      vertices[vi + 6] = 1
      vertices[vi + 7] = 1
      vertices[vi + 8] = 1
    }
  }

  // Read indices
  const numIndices = numFaces * 3
  const face = new draco.DracoInt32Array()
  let indices: Uint16Array | Uint32Array
  if (numVertices <= 65535) {
    indices = new Uint16Array(numIndices)
  } else {
    indices = new Uint32Array(numIndices)
  }
  for (let i = 0; i < numFaces; i++) {
    decoder.GetFaceFromMesh(dracoMesh, i, face)
    indices[i * 3] = face.GetValue(0)
    indices[i * 3 + 1] = face.GetValue(1)
    indices[i * 3 + 2] = face.GetValue(2)
  }

  // Cleanup
  draco.destroy(face)
  draco.destroy(posArray)
  if (normArray) draco.destroy(normArray)
  if (matIdxArray) draco.destroy(matIdxArray)
  draco.destroy(dracoMesh)
  draco.destroy(buffer)
  draco.destroy(decoder)

  return { vertices, indices, skinJoints, skinWeights }
}

function decodeStandardPrimitive(primitive: any, gltf: any, bin: Uint8Array): PrimitiveResult | null {
  if (primitive.attributes.POSITION === undefined || primitive.indices === undefined) return null

  const posAccessor = gltf.accessors[primitive.attributes.POSITION]
  const positions = readAccessorFloat32(posAccessor, gltf, bin)

  let normals: Float32Array | null = null
  if (primitive.attributes.NORMAL !== undefined) {
    const normAccessor = gltf.accessors[primitive.attributes.NORMAL]
    normals = readAccessorFloat32(normAccessor, gltf, bin)
  }

  const numVertices = posAccessor.count as number
  const vertices = new Float32Array(numVertices * 9)
  for (let i = 0; i < numVertices; i++) {
    const vi = i * 9
    vertices[vi] = positions[i * 3]!
    vertices[vi + 1] = positions[i * 3 + 1]!
    vertices[vi + 2] = positions[i * 3 + 2]!
    if (normals) {
      vertices[vi + 3] = normals[i * 3]!
      vertices[vi + 4] = normals[i * 3 + 1]!
      vertices[vi + 5] = normals[i * 3 + 2]!
    } else {
      vertices[vi + 5] = 1 // default up normal (+Z)
    }
    // Default white vertex color
    vertices[vi + 6] = 1
    vertices[vi + 7] = 1
    vertices[vi + 8] = 1
  }

  // Read skin data (JOINTS_0 + WEIGHTS_0) if present
  let skinJoints: Uint8Array | undefined
  let skinWeights: Float32Array | undefined
  if (primitive.attributes.JOINTS_0 !== undefined && primitive.attributes.WEIGHTS_0 !== undefined) {
    const jointsAccessor = gltf.accessors[primitive.attributes.JOINTS_0]
    skinJoints = readAccessorUint8(jointsAccessor, gltf, bin)
    const weightsAccessor = gltf.accessors[primitive.attributes.WEIGHTS_0]
    skinWeights = readAccessorFloat32(weightsAccessor, gltf, bin)
  }

  const idxAccessor = gltf.accessors[primitive.indices]
  const indices = readAccessorIndices(idxAccessor, gltf, bin)

  return { vertices, indices, skinJoints, skinWeights }
}

function readAccessorFloat32(accessor: any, gltf: any, bin: Uint8Array): Float32Array {
  const bv = gltf.bufferViews[accessor.bufferView]
  const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const count = accessor.count as number
  const componentCount = accessorTypeSize(accessor.type as string)
  const total = count * componentCount
  const absOffset = bin.byteOffset + byteOffset
  // Float32Array requires 4-byte alignment; copy if the offset is misaligned
  if (absOffset % 4 !== 0) {
    const out = new Float32Array(total)
    const src = new DataView(bin.buffer, absOffset, total * 4)
    for (let i = 0; i < total; i++) out[i] = src.getFloat32(i * 4, true)
    return out
  }
  return new Float32Array(bin.buffer, absOffset, total)
}

function readAccessorUint8(accessor: any, gltf: any, bin: Uint8Array): Uint8Array {
  const bv = gltf.bufferViews[accessor.bufferView]
  const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const count = accessor.count as number
  const componentCount = accessorTypeSize(accessor.type as string)
  const total = count * componentCount

  // componentType: 5121=UNSIGNED_BYTE, 5123=UNSIGNED_SHORT
  if (accessor.componentType === 5121) {
    return new Uint8Array(bin.buffer, bin.byteOffset + byteOffset, total)
  }
  // Unsigned short joints → downconvert to uint8
  if (accessor.componentType === 5123) {
    const absOffset = bin.byteOffset + byteOffset
    const out = new Uint8Array(total)
    if (absOffset % 2 !== 0) {
      const src = new DataView(bin.buffer, absOffset, total * 2)
      for (let i = 0; i < total; i++) out[i] = src.getUint16(i * 2, true)
    } else {
      const u16 = new Uint16Array(bin.buffer, absOffset, total)
      for (let i = 0; i < total; i++) out[i] = u16[i]!
    }
    return out
  }
  return new Uint8Array(bin.buffer, bin.byteOffset + byteOffset, total)
}

function readAccessorIndices(accessor: any, gltf: any, bin: Uint8Array): Uint16Array | Uint32Array {
  const bv = gltf.bufferViews[accessor.bufferView]
  const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const count = accessor.count as number
  const absOffset = bin.byteOffset + byteOffset

  // componentType: 5121=UNSIGNED_BYTE, 5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT
  if (accessor.componentType === 5125) {
    if (absOffset % 4 !== 0) {
      const out = new Uint32Array(count)
      const src = new DataView(bin.buffer, absOffset, count * 4)
      for (let i = 0; i < count; i++) out[i] = src.getUint32(i * 4, true)
      return out
    }
    return new Uint32Array(bin.buffer, absOffset, count)
  }
  if (accessor.componentType === 5123) {
    if (absOffset % 2 !== 0) {
      const out = new Uint16Array(count)
      const src = new DataView(bin.buffer, absOffset, count * 2)
      for (let i = 0; i < count; i++) out[i] = src.getUint16(i * 2, true)
      return out
    }
    return new Uint16Array(bin.buffer, absOffset, count)
  }
  // Unsigned byte — upconvert to uint16
  const bytes = bin.subarray(byteOffset, byteOffset + count)
  const out = new Uint16Array(count)
  for (let i = 0; i < count; i++) out[i] = bytes[i]!
  return out
}

function accessorTypeSize(type: string): number {
  switch (type) {
    case 'SCALAR':
      return 1
    case 'VEC2':
      return 2
    case 'VEC3':
      return 3
    case 'VEC4':
      return 4
    case 'MAT2':
      return 4
    case 'MAT3':
      return 9
    case 'MAT4':
      return 16
    default:
      return 1
  }
}
