import { describe, test, expect } from 'bun:test'

import { buildBVH, raycastBVH } from './bvh.ts'
import { createBoxGeometry } from './geometry.ts'

const faceOut = new Uint32Array(1)
const normalOut = new Float32Array(3)

describe('buildBVH', () => {
  test('builds from box geometry', () => {
    const box = createBoxGeometry(2, 2, 2)
    const bvh = buildBVH(box.vertices, box.indices)
    expect(bvh.nodeCount).toBeGreaterThan(0)
    expect(bvh.triIndices.length).toBe(box.indices.length / 3)
  })

  test('handles empty geometry', () => {
    const bvh = buildBVH(new Float32Array(0), new Uint16Array(0))
    expect(bvh.nodeCount).toBe(0)
  })
})

describe('raycastBVH', () => {
  test('hits a box from above (Z-down)', () => {
    const box = createBoxGeometry(2, 2, 2) // centered at origin, extends ±1
    const bvh = buildBVH(box.vertices, box.indices)

    // Ray from above, pointing down
    const t = raycastBVH(bvh, 0, 0, 5, 0, 0, -1, 100, faceOut, normalOut)
    expect(t).toBeGreaterThan(0)
    expect(t).toBeCloseTo(4, 3) // origin at z=5, box top at z=1 → distance = 4
    // Normal should point up (+Z)
    expect(normalOut[2]!).toBeCloseTo(1, 3)
  })

  test('hits a box from the side (+X)', () => {
    const box = createBoxGeometry(2, 2, 2)
    const bvh = buildBVH(box.vertices, box.indices)

    const t = raycastBVH(bvh, 5, 0, 0, -1, 0, 0, 100, faceOut, normalOut)
    expect(t).toBeGreaterThan(0)
    expect(t).toBeCloseTo(4, 3) // box face at x=1, origin at x=5 → distance = 4
    expect(normalOut[0]!).toBeCloseTo(1, 3) // normal points +X
  })

  test('misses when ray points away', () => {
    const box = createBoxGeometry(2, 2, 2)
    const bvh = buildBVH(box.vertices, box.indices)

    // Ray pointing away from box
    const t = raycastBVH(bvh, 5, 0, 0, 1, 0, 0, 100, faceOut, normalOut)
    expect(t).toBe(-1)
  })

  test('misses when ray is parallel and outside', () => {
    const box = createBoxGeometry(2, 2, 2)
    const bvh = buildBVH(box.vertices, box.indices)

    // Ray parallel to box face but outside
    const t = raycastBVH(bvh, 5, 0, 0, 0, 1, 0, 100, faceOut, normalOut)
    expect(t).toBe(-1)
  })

  test('respects maxT', () => {
    const box = createBoxGeometry(2, 2, 2)
    const bvh = buildBVH(box.vertices, box.indices)

    // Ray hits at t=4 but maxT=3
    const t = raycastBVH(bvh, 0, 0, 5, 0, 0, -1, 3, faceOut, normalOut)
    expect(t).toBe(-1)
  })

  test('ray from inside box hits near face', () => {
    const box = createBoxGeometry(2, 2, 2)
    const bvh = buildBVH(box.vertices, box.indices)

    // From origin (inside box), pointing +X
    const t = raycastBVH(bvh, 0, 0, 0, 1, 0, 0, 100, faceOut, normalOut)
    expect(t).toBeGreaterThan(0)
    expect(t).toBeCloseTo(1, 3) // box extends to x=1
  })

  test('works with a ground plane (two triangles)', () => {
    // 9 floats per vertex: px,py,pz, nx,ny,nz, cr,cg,cb
    // A 10×10 quad on the XY plane at Z=0
    const vertices = new Float32Array([
      // v0: (-5, -5, 0) normal (0,0,1) color (1,1,1)
      -5, -5, 0, 0, 0, 1, 1, 1, 1,
      // v1: (5, -5, 0)
      5, -5, 0, 0, 0, 1, 1, 1, 1,
      // v2: (5, 5, 0)
      5, 5, 0, 0, 0, 1, 1, 1, 1,
      // v3: (-5, 5, 0)
      -5, 5, 0, 0, 0, 1, 1, 1, 1,
    ])
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3])
    const bvh = buildBVH(vertices, indices)

    // Downward ray from above the plane center
    const t = raycastBVH(bvh, 0, 0, 10, 0, 0, -1, 100, faceOut, normalOut)
    expect(t).toBeCloseTo(10, 3)
    expect(normalOut[2]!).toBeCloseTo(1, 3)

    // Downward ray from above a corner
    const t2 = raycastBVH(bvh, 3, 3, 5, 0, 0, -1, 100, faceOut, normalOut)
    expect(t2).toBeCloseTo(5, 3)

    // Miss outside the plane
    const t3 = raycastBVH(bvh, 10, 0, 5, 0, 0, -1, 100, faceOut, normalOut)
    expect(t3).toBe(-1)
  })
})
