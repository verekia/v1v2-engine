// GLB/glTF loader with Draco mesh compression support

import { quatToEulerYXZ } from './math.ts'

export interface GltfMesh {
  name: string
  vertices: Float32Array // interleaved [px,py,pz, nx,ny,nz, cr,cg,cb]
  indices: Uint16Array | Uint32Array
  position: [number, number, number]
  rotation: [number, number, number] // euler YXZ
  scale: [number, number, number]
  color: [number, number, number]
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
): Promise<GltfMesh[]> {
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
      quatToEulerYXZ(rBuf, 0, qBuf, 0)
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
        meshes.push({
          name,
          vertices: result.vertices,
          indices: result.indices,
          position,
          rotation,
          scale,
          color,
        })
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

  return meshes
}

function decodePrimitive(
  primitive: any,
  gltf: any,
  bin: Uint8Array,
  draco: any,
  materialColors?: ColorMap,
): { vertices: Float32Array; indices: Uint16Array | Uint32Array } | null {
  const dracoExt = primitive.extensions?.KHR_draco_mesh_compression
  if (dracoExt && draco) {
    return decodeDracoPrimitive(dracoExt, gltf, bin, draco, materialColors)
  }
  return decodeStandardPrimitive(primitive, gltf, bin)
}

// Find the _materialindex key in the Draco attributes map (case-insensitive)
function findMaterialIndexKey(attributes: Record<string, number>): string | undefined {
  return Object.keys(attributes).find(k => k.toLowerCase() === '_materialindex')
}

function decodeDracoPrimitive(
  dracoExt: any,
  gltf: any,
  bin: Uint8Array,
  draco: any,
  materialColors?: ColorMap,
): { vertices: Float32Array; indices: Uint16Array | Uint32Array } | null {
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
      vertices[vi + 4] = 1 // default up normal
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

  return { vertices, indices }
}

function decodeStandardPrimitive(
  primitive: any,
  gltf: any,
  bin: Uint8Array,
): { vertices: Float32Array; indices: Uint16Array | Uint32Array } | null {
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
      vertices[vi + 4] = 1
    }
    // Default white vertex color
    vertices[vi + 6] = 1
    vertices[vi + 7] = 1
    vertices[vi + 8] = 1
  }

  const idxAccessor = gltf.accessors[primitive.indices]
  const indices = readAccessorIndices(idxAccessor, gltf, bin)

  return { vertices, indices }
}

function readAccessorFloat32(accessor: any, gltf: any, bin: Uint8Array): Float32Array {
  const bv = gltf.bufferViews[accessor.bufferView]
  const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const count = accessor.count as number
  const componentCount = accessorTypeSize(accessor.type as string)
  return new Float32Array(bin.buffer, bin.byteOffset + byteOffset, count * componentCount)
}

function readAccessorIndices(accessor: any, gltf: any, bin: Uint8Array): Uint16Array | Uint32Array {
  const bv = gltf.bufferViews[accessor.bufferView]
  const byteOffset = (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
  const count = accessor.count as number

  // componentType: 5121=UNSIGNED_BYTE, 5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT
  if (accessor.componentType === 5125) {
    return new Uint32Array(bin.buffer, bin.byteOffset + byteOffset, count)
  }
  if (accessor.componentType === 5123) {
    return new Uint16Array(bin.buffer, bin.byteOffset + byteOffset, count)
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
