// Interleaved: [px, py, pz, nx, ny, nz, cr, cg, cb] per vertex (stride 36 bytes)

// ── Cube ──────────────────────────────────────────────────────────────
// 24 vertices (4 per face × 6 faces), 36 indices

// prettier-ignore
export const cubeVertices = new Float32Array([
  // +Z face (top)
  -1, -1,  1,   0,  0,  1,   1, 1, 1,
   1, -1,  1,   0,  0,  1,   1, 1, 1,
   1,  1,  1,   0,  0,  1,   1, 1, 1,
  -1,  1,  1,   0,  0,  1,   1, 1, 1,
  // -Z face (bottom)
   1, -1, -1,   0,  0, -1,   1, 1, 1,
  -1, -1, -1,   0,  0, -1,   1, 1, 1,
  -1,  1, -1,   0,  0, -1,   1, 1, 1,
   1,  1, -1,   0,  0, -1,   1, 1, 1,
  // +Y face (front)
  -1,  1,  1,   0,  1,  0,   1, 1, 1,
   1,  1,  1,   0,  1,  0,   1, 1, 1,
   1,  1, -1,   0,  1,  0,   1, 1, 1,
  -1,  1, -1,   0,  1,  0,   1, 1, 1,
  // -Y face (back)
  -1, -1, -1,   0, -1,  0,   1, 1, 1,
   1, -1, -1,   0, -1,  0,   1, 1, 1,
   1, -1,  1,   0, -1,  0,   1, 1, 1,
  -1, -1,  1,   0, -1,  0,   1, 1, 1,
  // +X face (right)
   1, -1,  1,   1,  0,  0,   1, 1, 1,
   1, -1, -1,   1,  0,  0,   1, 1, 1,
   1,  1, -1,   1,  0,  0,   1, 1, 1,
   1,  1,  1,   1,  0,  0,   1, 1, 1,
  // -X face (left)
  -1, -1, -1,  -1,  0,  0,   1, 1, 1,
  -1, -1,  1,  -1,  0,  0,   1, 1, 1,
  -1,  1,  1,  -1,  0,  0,   1, 1, 1,
  -1,  1, -1,  -1,  0,  0,   1, 1, 1,
])

// prettier-ignore
export const cubeIndices = new Uint16Array([
   0,  1,  2,   0,  2,  3,   // +Z (top)
   4,  5,  6,   4,  6,  7,   // -Z (bottom)
   8,  9, 10,   8, 10, 11,   // +Y (front)
  12, 13, 14,  12, 14, 15,   // -Y (back)
  16, 17, 18,  16, 18, 19,   // +X
  20, 21, 22,  20, 22, 23,   // -X
])

// ── Sphere (UV sphere) ───────────────────────────────────────────────

export function createSphereGeometry(stacks = 16, slices = 24): { vertices: Float32Array; indices: Uint16Array } {
  const vertCount = (stacks + 1) * (slices + 1)
  const vertices = new Float32Array(vertCount * 9)
  let vi = 0

  for (let st = 0; st <= stacks; st++) {
    const phi = (st / stacks) * Math.PI
    const sinP = Math.sin(phi)
    const cosP = Math.cos(phi)
    for (let sl = 0; sl <= slices; sl++) {
      const theta = (sl / slices) * Math.PI * 2
      const nx = sinP * Math.cos(theta)
      const ny = sinP * Math.sin(theta)
      const nz = cosP
      // position = normal (unit sphere)
      vertices[vi++] = nx
      vertices[vi++] = ny
      vertices[vi++] = nz
      // normal
      vertices[vi++] = nx
      vertices[vi++] = ny
      vertices[vi++] = nz
      // color (white)
      vertices[vi++] = 1
      vertices[vi++] = 1
      vertices[vi++] = 1
    }
  }

  const triCount = stacks * slices * 2
  const indices = new Uint16Array(triCount * 3)
  let ii = 0
  const row = slices + 1

  for (let st = 0; st < stacks; st++) {
    for (let sl = 0; sl < slices; sl++) {
      const a = st * row + sl
      const b = a + row
      indices[ii++] = a
      indices[ii++] = b
      indices[ii++] = a + 1
      indices[ii++] = a + 1
      indices[ii++] = b
      indices[ii++] = b + 1
    }
  }

  return { vertices, indices }
}

// ── Merge primitives ────────────────────────────────────────────────

export function mergeGeometries(
  primitives: {
    vertices: Float32Array
    indices: Uint16Array | Uint32Array
    color: [number, number, number]
  }[],
): { vertices: Float32Array; indices: Uint32Array } {
  let totalVerts = 0
  let totalIdxs = 0
  for (const p of primitives) {
    totalVerts += p.vertices.length / 9
    totalIdxs += p.indices.length
  }
  const vertices = new Float32Array(totalVerts * 9)
  const indices = new Uint32Array(totalIdxs)
  let vertOff = 0
  let idxOff = 0
  for (const p of primitives) {
    const vCount = p.vertices.length / 9
    for (let v = 0; v < vCount; v++) {
      const src = v * 9
      const dst = (vertOff + v) * 9
      vertices[dst]! = p.vertices[src]!
      vertices[dst + 1]! = p.vertices[src + 1]!
      vertices[dst + 2]! = p.vertices[src + 2]!
      vertices[dst + 3]! = p.vertices[src + 3]!
      vertices[dst + 4]! = p.vertices[src + 4]!
      vertices[dst + 5]! = p.vertices[src + 5]!
      vertices[dst + 6]! = p.color[0]
      vertices[dst + 7]! = p.color[1]
      vertices[dst + 8]! = p.color[2]
    }
    for (let j = 0; j < p.indices.length; j++) {
      indices[idxOff + j] = p.indices[j]! + vertOff
    }
    vertOff += vCount
    idxOff += p.indices.length
  }
  return { vertices, indices }
}
