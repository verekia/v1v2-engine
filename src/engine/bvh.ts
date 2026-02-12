// BVH acceleration structure for raycasting.
// SAH-binned build, flat-array storage, stack-based traversal — zero allocation in the hot path.

const STRIDE = 10 // interleaved vertex stride: px,py,pz, nx,ny,nz, cr,cg,cb, bloom
const NODE_FLOATS = 8
// Node layout (8 floats):
//   [0–2] AABB min (x,y,z)
//   [3–5] AABB max (x,y,z)
//   [6]   inner → right-child node index | leaf → first triangle offset in triIndices
//   [7]   triangle count: 0 = inner node, >0 = leaf
// For inner nodes, left child is always currentIndex + 1 (depth-first build order).

const MAX_LEAF_TRIS = 4
const SAH_BINS = 12

export interface BVH {
  nodes: Float32Array
  nodeCount: number
  triIndices: Uint32Array
  vertices: Float32Array
  indices: Uint16Array | Uint32Array
}

// ── Build ────────────────────────────────────────────────────────────────────

export function buildBVH(vertices: Float32Array, indices: Uint16Array | Uint32Array): BVH {
  const triCount = (indices.length / 3) | 0
  if (triCount === 0) {
    const empty = new Float32Array(NODE_FLOATS)
    return { nodes: empty, nodeCount: 0, triIndices: new Uint32Array(0), vertices, indices }
  }

  // Per-triangle centroids and AABBs
  const centroids = new Float32Array(triCount * 3)
  const triMins = new Float32Array(triCount * 3)
  const triMaxs = new Float32Array(triCount * 3)
  const triOrder = new Uint32Array(triCount)

  for (let t = 0; t < triCount; t++) {
    triOrder[t] = t
    const i0 = indices[t * 3]! * STRIDE
    const i1 = indices[t * 3 + 1]! * STRIDE
    const i2 = indices[t * 3 + 2]! * STRIDE
    const ax = vertices[i0]!,
      ay = vertices[i0 + 1]!,
      az = vertices[i0 + 2]!
    const bx = vertices[i1]!,
      by = vertices[i1 + 1]!,
      bz = vertices[i1 + 2]!
    const cx = vertices[i2]!,
      cy = vertices[i2 + 1]!,
      cz = vertices[i2 + 2]!
    const t3 = t * 3
    centroids[t3] = (ax + bx + cx) / 3
    centroids[t3 + 1] = (ay + by + cy) / 3
    centroids[t3 + 2] = (az + bz + cz) / 3
    triMins[t3] = Math.min(ax, bx, cx)
    triMins[t3 + 1] = Math.min(ay, by, cy)
    triMins[t3 + 2] = Math.min(az, bz, cz)
    triMaxs[t3] = Math.max(ax, bx, cx)
    triMaxs[t3 + 1] = Math.max(ay, by, cy)
    triMaxs[t3 + 2] = Math.max(az, bz, cz)
  }

  const maxNodes = triCount * 2
  const nodes = new Float32Array(maxNodes * NODE_FLOATS)
  let nodeCount = 0

  // Scratch arrays for SAH sweep (reused across build calls)
  const binCounts = new Uint32Array(SAH_BINS)
  const binMins = new Float32Array(SAH_BINS * 3)
  const binMaxs = new Float32Array(SAH_BINS * 3)
  const leftAreas = new Float32Array(SAH_BINS)
  const leftCounts = new Uint32Array(SAH_BINS)
  const lMin = new Float32Array(3)
  const lMax = new Float32Array(3)
  const rMin = new Float32Array(3)
  const rMax = new Float32Array(3)

  function halfArea(minA: Float32Array, mo: number, maxA: Float32Array, xo: number): number {
    const dx = maxA[xo]! - minA[mo]!
    const dy = maxA[xo + 1]! - minA[mo + 1]!
    const dz = maxA[xo + 2]! - minA[mo + 2]!
    return dx * dy + dy * dz + dz * dx
  }

  function build(start: number, count: number): number {
    const ni = nodeCount++
    const no = ni * NODE_FLOATS

    // Compute AABB
    let mnX = Infinity,
      mnY = Infinity,
      mnZ = Infinity
    let mxX = -Infinity,
      mxY = -Infinity,
      mxZ = -Infinity
    for (let i = start; i < start + count; i++) {
      const t3 = triOrder[i]! * 3
      if (triMins[t3]! < mnX) mnX = triMins[t3]!
      if (triMins[t3 + 1]! < mnY) mnY = triMins[t3 + 1]!
      if (triMins[t3 + 2]! < mnZ) mnZ = triMins[t3 + 2]!
      if (triMaxs[t3]! > mxX) mxX = triMaxs[t3]!
      if (triMaxs[t3 + 1]! > mxY) mxY = triMaxs[t3 + 1]!
      if (triMaxs[t3 + 2]! > mxZ) mxZ = triMaxs[t3 + 2]!
    }
    nodes[no] = mnX
    nodes[no + 1] = mnY
    nodes[no + 2] = mnZ
    nodes[no + 3] = mxX
    nodes[no + 4] = mxY
    nodes[no + 5] = mxZ

    if (count <= MAX_LEAF_TRIS) {
      nodes[no + 6] = start
      nodes[no + 7] = count
      return ni
    }

    const parentSA = halfArea(nodes, no, nodes, no + 3)
    if (parentSA < 1e-12) {
      nodes[no + 6] = start
      nodes[no + 7] = count
      return ni
    }

    let bestCost = Infinity
    let bestAxis = -1
    let bestSplit = 0

    for (let axis = 0; axis < 3; axis++) {
      let cMin = Infinity,
        cMax = -Infinity
      for (let i = start; i < start + count; i++) {
        const c = centroids[triOrder[i]! * 3 + axis]!
        if (c < cMin) cMin = c
        if (c > cMax) cMax = c
      }
      if (cMax - cMin < 1e-10) continue

      // Bin triangles
      binCounts.fill(0)
      binMins.fill(Infinity)
      binMaxs.fill(-Infinity)
      const scale = SAH_BINS / (cMax - cMin)
      for (let i = start; i < start + count; i++) {
        const t = triOrder[i]!
        const bin = Math.min(SAH_BINS - 1, ((centroids[t * 3 + axis]! - cMin) * scale) | 0)
        binCounts[bin]!++
        const t3 = t * 3
        const b3 = bin * 3
        if (triMins[t3]! < binMins[b3]!) binMins[b3] = triMins[t3]!
        if (triMins[t3 + 1]! < binMins[b3 + 1]!) binMins[b3 + 1] = triMins[t3 + 1]!
        if (triMins[t3 + 2]! < binMins[b3 + 2]!) binMins[b3 + 2] = triMins[t3 + 2]!
        if (triMaxs[t3]! > binMaxs[b3]!) binMaxs[b3] = triMaxs[t3]!
        if (triMaxs[t3 + 1]! > binMaxs[b3 + 1]!) binMaxs[b3 + 1] = triMaxs[t3 + 1]!
        if (triMaxs[t3 + 2]! > binMaxs[b3 + 2]!) binMaxs[b3 + 2] = triMaxs[t3 + 2]!
      }

      // Left sweep
      lMin[0] = Infinity
      lMin[1] = Infinity
      lMin[2] = Infinity
      lMax[0] = -Infinity
      lMax[1] = -Infinity
      lMax[2] = -Infinity
      let lc = 0
      for (let i = 0; i < SAH_BINS - 1; i++) {
        lc += binCounts[i]!
        const b3 = i * 3
        if (binCounts[i]! > 0) {
          if (binMins[b3]! < lMin[0]!) lMin[0] = binMins[b3]!
          if (binMins[b3 + 1]! < lMin[1]!) lMin[1] = binMins[b3 + 1]!
          if (binMins[b3 + 2]! < lMin[2]!) lMin[2] = binMins[b3 + 2]!
          if (binMaxs[b3]! > lMax[0]!) lMax[0] = binMaxs[b3]!
          if (binMaxs[b3 + 1]! > lMax[1]!) lMax[1] = binMaxs[b3 + 1]!
          if (binMaxs[b3 + 2]! > lMax[2]!) lMax[2] = binMaxs[b3 + 2]!
        }
        leftCounts[i] = lc
        leftAreas[i] = lc > 0 ? halfArea(lMin, 0, lMax, 0) : 0
      }

      // Right sweep + evaluate SAH splits
      rMin[0] = Infinity
      rMin[1] = Infinity
      rMin[2] = Infinity
      rMax[0] = -Infinity
      rMax[1] = -Infinity
      rMax[2] = -Infinity
      let rc = 0
      for (let i = SAH_BINS - 1; i > 0; i--) {
        rc += binCounts[i]!
        const b3 = i * 3
        if (binCounts[i]! > 0) {
          if (binMins[b3]! < rMin[0]!) rMin[0] = binMins[b3]!
          if (binMins[b3 + 1]! < rMin[1]!) rMin[1] = binMins[b3 + 1]!
          if (binMins[b3 + 2]! < rMin[2]!) rMin[2] = binMins[b3 + 2]!
          if (binMaxs[b3]! > rMax[0]!) rMax[0] = binMaxs[b3]!
          if (binMaxs[b3 + 1]! > rMax[1]!) rMax[1] = binMaxs[b3 + 1]!
          if (binMaxs[b3 + 2]! > rMax[2]!) rMax[2] = binMaxs[b3 + 2]!
        }
        const li = leftCounts[i - 1]!
        if (li === 0 || rc === 0) continue
        const cost = li * leftAreas[i - 1]! + rc * halfArea(rMin, 0, rMax, 0)
        if (cost < bestCost) {
          bestCost = cost
          bestAxis = axis
          bestSplit = i
        }
      }
    }

    // If no good split, make leaf
    if (bestAxis < 0 || bestCost >= count * parentSA) {
      nodes[no + 6] = start
      nodes[no + 7] = count
      return ni
    }

    // Partition triOrder[start..start+count) around bestSplit on bestAxis
    let cMin = Infinity,
      cMax = -Infinity
    for (let i = start; i < start + count; i++) {
      const c = centroids[triOrder[i]! * 3 + bestAxis]!
      if (c < cMin) cMin = c
      if (c > cMax) cMax = c
    }
    const scale = SAH_BINS / (cMax - cMin)
    let lo = start,
      hi = start + count - 1
    while (lo <= hi) {
      const bin = Math.min(SAH_BINS - 1, ((centroids[triOrder[lo]! * 3 + bestAxis]! - cMin) * scale) | 0)
      if (bin < bestSplit) {
        lo++
      } else {
        const tmp = triOrder[lo]!
        triOrder[lo] = triOrder[hi]!
        triOrder[hi] = tmp
        hi--
      }
    }
    const leftCount = lo - start
    if (leftCount === 0 || leftCount === count) {
      // Partition degenerate — make leaf
      nodes[no + 6] = start
      nodes[no + 7] = count
      return ni
    }

    // Build children (left first → left child = ni + 1)
    build(start, leftCount)
    const rightChild = build(start + leftCount, count - leftCount)

    nodes[no + 6] = rightChild
    nodes[no + 7] = 0 // count=0 → inner node
    return ni
  }

  build(0, triCount)

  return {
    nodes: nodes.slice(0, nodeCount * NODE_FLOATS),
    nodeCount,
    triIndices: triOrder,
    vertices,
    indices,
  }
}

// ── Raycast ──────────────────────────────────────────────────────────────────

// Pre-allocated traversal stack (single-threaded, shared across all raycasts)
const _stack = new Uint32Array(64)

/**
 * Raycast against a BVH in local space.
 * Direction does NOT need to be normalized — the returned t is the raw parameter
 * such that hitPoint = origin + t * direction.
 * Returns t of nearest hit (< maxT), or -1 if no hit.
 * On hit, writes triangle index to faceOut[0] and local-space face normal to normalOut[0..2].
 */
export function raycastBVH(
  bvh: BVH,
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  maxT: number,
  faceOut: Uint32Array,
  normalOut: Float32Array,
): number {
  const { nodes, triIndices, vertices, indices } = bvh
  if (bvh.nodeCount === 0) return -1

  const idx = 1 / dx,
    idy = 1 / dy,
    idz = 1 / dz
  let closest = maxT
  let hitFace = -1
  let hitNx = 0,
    hitNy = 0,
    hitNz = 0

  let sp = 0
  _stack[sp++] = 0 // start at root

  while (sp > 0) {
    const ni = _stack[--sp]!
    const no = ni * NODE_FLOATS

    // Ray-AABB slab test
    const mnX = nodes[no]!,
      mnY = nodes[no + 1]!,
      mnZ = nodes[no + 2]!
    const mxX = nodes[no + 3]!,
      mxY = nodes[no + 4]!,
      mxZ = nodes[no + 5]!

    let tmin: number, tmax: number
    if (idx >= 0) {
      tmin = (mnX - ox) * idx
      tmax = (mxX - ox) * idx
    } else {
      tmin = (mxX - ox) * idx
      tmax = (mnX - ox) * idx
    }
    let tymin: number, tymax: number
    if (idy >= 0) {
      tymin = (mnY - oy) * idy
      tymax = (mxY - oy) * idy
    } else {
      tymin = (mxY - oy) * idy
      tymax = (mnY - oy) * idy
    }
    if (tmin > tymax || tymin > tmax) continue
    if (tymin > tmin) tmin = tymin
    if (tymax < tmax) tmax = tymax

    let tzmin: number, tzmax: number
    if (idz >= 0) {
      tzmin = (mnZ - oz) * idz
      tzmax = (mxZ - oz) * idz
    } else {
      tzmin = (mxZ - oz) * idz
      tzmax = (mnZ - oz) * idz
    }
    if (tmin > tzmax || tzmin > tmax) continue
    if (tzmin > tmin) tmin = tzmin
    if (tzmax < tmax) tmax = tzmax

    if (tmax < 0 || tmin > closest) continue

    const cnt = nodes[no + 7]!
    if (cnt > 0) {
      // Leaf — test triangles (Möller–Trumbore)
      const first = nodes[no + 6]! | 0
      for (let i = first; i < first + cnt; i++) {
        const tri = triIndices[i]!
        const ti = tri * 3
        const vi0 = indices[ti]! * STRIDE
        const vi1 = indices[ti + 1]! * STRIDE
        const vi2 = indices[ti + 2]! * STRIDE

        const v0x = vertices[vi0]!,
          v0y = vertices[vi0 + 1]!,
          v0z = vertices[vi0 + 2]!
        const e1x = vertices[vi1]! - v0x,
          e1y = vertices[vi1 + 1]! - v0y,
          e1z = vertices[vi1 + 2]! - v0z
        const e2x = vertices[vi2]! - v0x,
          e2y = vertices[vi2 + 1]! - v0y,
          e2z = vertices[vi2 + 2]! - v0z

        // p = cross(dir, e2)
        const px = dy * e2z - dz * e2y,
          py = dz * e2x - dx * e2z,
          pz = dx * e2y - dy * e2x
        const det = e1x * px + e1y * py + e1z * pz
        if (det > -1e-8 && det < 1e-8) continue
        const idet = 1 / det

        const tx = ox - v0x,
          ty = oy - v0y,
          tz = oz - v0z
        const u = (tx * px + ty * py + tz * pz) * idet
        if (u < 0 || u > 1) continue

        const qx = ty * e1z - tz * e1y,
          qy = tz * e1x - tx * e1z,
          qz = tx * e1y - ty * e1x
        const v = (dx * qx + dy * qy + dz * qz) * idet
        if (v < 0 || u + v > 1) continue

        const t = (e2x * qx + e2y * qy + e2z * qz) * idet
        if (t > 0 && t < closest) {
          closest = t
          hitFace = tri
          hitNx = e1y * e2z - e1z * e2y
          hitNy = e1z * e2x - e1x * e2z
          hitNz = e1x * e2y - e1y * e2x
        }
      }
    } else {
      // Inner — push both children; left = ni+1, right = nodes[no+6]
      const right = nodes[no + 6]! | 0
      _stack[sp++] = right
      _stack[sp++] = ni + 1 // left on top → processed first
    }
  }

  if (hitFace >= 0) {
    faceOut[0] = hitFace
    const len = Math.sqrt(hitNx * hitNx + hitNy * hitNy + hitNz * hitNz) || 1
    normalOut[0] = hitNx / len
    normalOut[1] = hitNy / len
    normalOut[2] = hitNz / len
    return closest
  }
  return -1
}
