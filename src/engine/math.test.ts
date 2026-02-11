import { describe, test, expect } from 'bun:test'

import { quatToEulerZXY, m4FromTRS, m4FromQuatTRS } from './math.ts'

const EPSILON = 1e-6

function approxEqual(a: number, b: number, eps = EPSILON): boolean {
  return Math.abs(a - b) < eps
}

// Build a quaternion from axis-angle (stored in Float32Array like the engine)
function quatFromAxisAngle(ax: number, ay: number, az: number, angle: number): Float32Array {
  const half = angle / 2
  const s = Math.sin(half)
  return new Float32Array([ax * s, ay * s, az * s, Math.cos(half)])
}

describe('quatToEulerZXY', () => {
  test('identity quaternion → zero euler', () => {
    const q = new Float32Array([0, 0, 0, 1])
    const out = new Float32Array(3)
    quatToEulerZXY(out, 0, q, 0)
    expect(approxEqual(out[0]!, 0)).toBe(true)
    expect(approxEqual(out[1]!, 0)).toBe(true)
    expect(approxEqual(out[2]!, 0)).toBe(true)
  })

  test('90° around Z → rz = π/2', () => {
    const q = quatFromAxisAngle(0, 0, 1, Math.PI / 2)
    const out = new Float32Array(3)
    quatToEulerZXY(out, 0, q, 0)
    expect(approxEqual(out[0]!, 0)).toBe(true) // rx
    expect(approxEqual(out[1]!, 0)).toBe(true) // ry
    expect(approxEqual(out[2]!, Math.PI / 2)).toBe(true) // rz
  })

  test('45° around X → rx = π/4', () => {
    const q = quatFromAxisAngle(1, 0, 0, Math.PI / 4)
    const out = new Float32Array(3)
    quatToEulerZXY(out, 0, q, 0)
    expect(approxEqual(out[0]!, Math.PI / 4)).toBe(true) // rx
    expect(approxEqual(out[1]!, 0)).toBe(true) // ry
    expect(approxEqual(out[2]!, 0)).toBe(true) // rz
  })

  test('90° around Y → ry = π/2', () => {
    const q = quatFromAxisAngle(0, 1, 0, Math.PI / 2)
    const out = new Float32Array(3)
    quatToEulerZXY(out, 0, q, 0)
    expect(approxEqual(out[0]!, 0)).toBe(true) // rx
    expect(approxEqual(out[1]!, Math.PI / 2)).toBe(true) // ry
    expect(approxEqual(out[2]!, 0)).toBe(true) // rz
  })
})

describe('quatToEulerZXY ↔ m4FromTRS roundtrip', () => {
  // For any quaternion (away from gimbal lock at rx=±π/2), converting to euler
  // then building a matrix via m4FromTRS should match m4FromQuatTRS.
  function testRoundtrip(label: string, q: Float32Array) {
    test(label, () => {
      const euler = new Float32Array(3)
      quatToEulerZXY(euler, 0, q, 0)

      const pos = new Float32Array([0, 0, 0])
      const scl = new Float32Array([1, 1, 1])

      const matEuler = new Float32Array(16)
      m4FromTRS(matEuler, 0, pos, 0, euler, 0, scl, 0)

      const matQuat = new Float32Array(16)
      m4FromQuatTRS(matQuat, 0, pos, 0, q, 0, scl, 0)

      // Compare the 3×3 rotation block (first 3 elements of each column)
      for (let col = 0; col < 3; col++) {
        for (let row = 0; row < 3; row++) {
          const i = col * 4 + row
          if (!approxEqual(matEuler[i]!, matQuat[i]!)) {
            throw new Error(`Mismatch at [${col}][${row}]: euler=${matEuler[i]} quat=${matQuat[i]}`)
          }
        }
      }
    })
  }

  testRoundtrip('identity', new Float32Array([0, 0, 0, 1]))
  testRoundtrip('90° around Z', quatFromAxisAngle(0, 0, 1, Math.PI / 2))
  testRoundtrip('90° around Y', quatFromAxisAngle(0, 1, 0, Math.PI / 2))
  testRoundtrip('45° around X', quatFromAxisAngle(1, 0, 0, Math.PI / 4))
  testRoundtrip('45° around Z', quatFromAxisAngle(0, 0, 1, Math.PI / 4))
  testRoundtrip('30° around (1,1,0)', quatFromAxisAngle(1 / Math.SQRT2, 1 / Math.SQRT2, 0, Math.PI / 6))
  testRoundtrip(
    'arbitrary rotation',
    quatFromAxisAngle(0.2672612419124244, 0.5345224838248488, 0.8017837257372732, 1.2),
  )
  // Note: 90° around X (rx=π/2) is gimbal lock for ZXY order — the euler→matrix
  // path amplifies float32 precision loss through asin near ±1. The direct
  // quaternion→matrix path (m4FromQuatTRS) doesn't have this issue. This is a
  // known mathematical limitation, not a code bug.
})
