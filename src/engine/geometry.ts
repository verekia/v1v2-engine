// Interleaved: [px, py, pz, nx, ny, nz, cr, cg, cb, bloom] per vertex (stride 40 bytes)
// Z-up coordinate system: width=X, height=Z(up), depth=Y(forward)

// ── Box ─────────────────────────────────────────────────────────────

export function createBoxGeometry(
  width = 1,
  height = 1,
  depth = 1,
  widthSegments = 1,
  heightSegments = 1,
  depthSegments = 1,
): { vertices: Float32Array; indices: Uint16Array } {
  const ws = Math.max(1, widthSegments | 0)
  const hs = Math.max(1, heightSegments | 0)
  const ds = Math.max(1, depthSegments | 0)

  const hw = width / 2
  const hh = height / 2
  const hd = depth / 2

  // Face +Z/-Z: grid ws × ds, Face +Y/-Y: grid ws × hs, Face +X/-X: grid hs × ds
  const totalVerts = 2 * (ws + 1) * (ds + 1) + 2 * (ws + 1) * (hs + 1) + 2 * (hs + 1) * (ds + 1)
  const totalIdx = 2 * ws * ds * 6 + 2 * ws * hs * 6 + 2 * hs * ds * 6

  const vertices = new Float32Array(totalVerts * 10)
  const indices = new Uint16Array(totalIdx)
  let vi = 0
  let ii = 0
  let vertexOffset = 0

  function buildFace(
    nx: number,
    ny: number,
    nz: number,
    uSegs: number,
    vSegs: number,
    pos: (u: number, v: number) => [number, number, number],
  ) {
    const start = vertexOffset
    for (let iv = 0; iv <= vSegs; iv++) {
      for (let iu = 0; iu <= uSegs; iu++) {
        const [px, py, pz] = pos(iu / uSegs, iv / vSegs)
        vertices[vi++] = px
        vertices[vi++] = py
        vertices[vi++] = pz
        vertices[vi++] = nx
        vertices[vi++] = ny
        vertices[vi++] = nz
        vertices[vi++] = 1
        vertices[vi++] = 1
        vertices[vi++] = 1
        vertices[vi++] = 0 // bloom
      }
    }
    const row = uSegs + 1
    for (let iv = 0; iv < vSegs; iv++) {
      for (let iu = 0; iu < uSegs; iu++) {
        const a = start + iv * row + iu
        const b = a + row
        indices[ii++] = a
        indices[ii++] = a + 1
        indices[ii++] = b + 1
        indices[ii++] = a
        indices[ii++] = b + 1
        indices[ii++] = b
      }
    }
    vertexOffset += (uSegs + 1) * (vSegs + 1)
  }

  // +Z (top)
  buildFace(0, 0, 1, ws, ds, (u, v) => [-hw + u * width, -hd + v * depth, hh])
  // -Z (bottom)
  buildFace(0, 0, -1, ws, ds, (u, v) => [hw - u * width, -hd + v * depth, -hh])
  // +Y (front)
  buildFace(0, 1, 0, ws, hs, (u, v) => [-hw + u * width, hd, hh - v * height])
  // -Y (back)
  buildFace(0, -1, 0, ws, hs, (u, v) => [-hw + u * width, -hd, -hh + v * height])
  // +X (right)
  buildFace(1, 0, 0, hs, ds, (u, v) => [hw, -hd + v * depth, hh - u * height])
  // -X (left)
  buildFace(-1, 0, 0, hs, ds, (u, v) => [-hw, -hd + v * depth, -hh + u * height])

  return { vertices, indices }
}

// ── Sphere (UV sphere) ──────────────────────────────────────────────

export function createSphereGeometry(
  radius = 1,
  widthSegments = 24,
  heightSegments = 16,
  inverted = false,
): { vertices: Float32Array; indices: Uint16Array } {
  const ws = Math.max(3, widthSegments | 0)
  const hs = Math.max(2, heightSegments | 0)
  const vertCount = (hs + 1) * (ws + 1)
  const vertices = new Float32Array(vertCount * 10)
  const sign = inverted ? -1 : 1
  let vi = 0

  for (let st = 0; st <= hs; st++) {
    const phi = (st / hs) * Math.PI
    const sinP = Math.sin(phi)
    const cosP = Math.cos(phi)
    for (let sl = 0; sl <= ws; sl++) {
      const theta = (sl / ws) * Math.PI * 2
      const nx = sinP * Math.cos(theta)
      const ny = sinP * Math.sin(theta)
      const nz = cosP
      vertices[vi++] = nx * radius
      vertices[vi++] = ny * radius
      vertices[vi++] = nz * radius
      vertices[vi++] = nx * sign
      vertices[vi++] = ny * sign
      vertices[vi++] = nz * sign
      vertices[vi++] = 1
      vertices[vi++] = 1
      vertices[vi++] = 1
      vertices[vi++] = 0 // bloom
    }
  }

  const triCount = hs * ws * 2
  const indices = new Uint16Array(triCount * 3)
  let ii = 0
  const row = ws + 1

  for (let st = 0; st < hs; st++) {
    for (let sl = 0; sl < ws; sl++) {
      const a = st * row + sl
      const b = a + row
      if (inverted) {
        indices[ii++] = a
        indices[ii++] = a + 1
        indices[ii++] = b
        indices[ii++] = a + 1
        indices[ii++] = b + 1
        indices[ii++] = b
      } else {
        indices[ii++] = a
        indices[ii++] = b
        indices[ii++] = a + 1
        indices[ii++] = a + 1
        indices[ii++] = b
        indices[ii++] = b + 1
      }
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
    bloom?: number
    uvs?: Float32Array
  }[],
): { vertices: Float32Array; indices: Uint32Array; uvs?: Float32Array } {
  let totalVerts = 0
  let totalIdxs = 0
  let hasAnyUVs = false
  for (const p of primitives) {
    totalVerts += p.vertices.length / 10
    totalIdxs += p.indices.length
    if (p.uvs) hasAnyUVs = true
  }
  const vertices = new Float32Array(totalVerts * 10)
  const indices = new Uint32Array(totalIdxs)
  const uvs = hasAnyUVs ? new Float32Array(totalVerts * 2) : undefined
  let vertOff = 0
  let idxOff = 0
  for (const p of primitives) {
    const vCount = p.vertices.length / 10
    for (let v = 0; v < vCount; v++) {
      const src = v * 10
      const dst = (vertOff + v) * 10
      vertices[dst]! = p.vertices[src]!
      vertices[dst + 1]! = p.vertices[src + 1]!
      vertices[dst + 2]! = p.vertices[src + 2]!
      vertices[dst + 3]! = p.vertices[src + 3]!
      vertices[dst + 4]! = p.vertices[src + 4]!
      vertices[dst + 5]! = p.vertices[src + 5]!
      vertices[dst + 6]! = p.color[0]
      vertices[dst + 7]! = p.color[1]
      vertices[dst + 8]! = p.color[2]
      vertices[dst + 9]! = p.bloom ?? 0
      if (uvs) {
        const uvDst = (vertOff + v) * 2
        if (p.uvs) {
          uvs[uvDst] = p.uvs[v * 2]!
          uvs[uvDst + 1] = p.uvs[v * 2 + 1]!
        }
        // else zeros (Float32Array default)
      }
    }
    for (let j = 0; j < p.indices.length; j++) {
      indices[idxOff + j] = p.indices[j]! + vertOff
    }
    vertOff += vCount
    idxOff += p.indices.length
  }
  return uvs ? { vertices, indices, uvs } : { vertices, indices }
}
