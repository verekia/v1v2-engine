// Zero-allocation math: all functions write to `out` at `o` (offset).

// ── vec3 ──────────────────────────────────────────────────────────────

export function v3Set(out: Float32Array, o: number, x: number, y: number, z: number): void {
  out[o] = x
  out[o + 1] = y
  out[o + 2] = z
}

export function v3Copy(out: Float32Array, o: number, a: Float32Array, ao: number): void {
  out[o] = a[ao]!
  out[o + 1] = a[ao + 1]!
  out[o + 2] = a[ao + 2]!
}

export function v3Add(out: Float32Array, o: number, a: Float32Array, ao: number, b: Float32Array, bo: number): void {
  out[o] = a[ao]! + b[bo]!
  out[o + 1] = a[ao + 1]! + b[bo + 1]!
  out[o + 2] = a[ao + 2]! + b[bo + 2]!
}

export function v3Subtract(
  out: Float32Array,
  o: number,
  a: Float32Array,
  ao: number,
  b: Float32Array,
  bo: number,
): void {
  out[o] = a[ao]! - b[bo]!
  out[o + 1] = a[ao + 1]! - b[bo + 1]!
  out[o + 2] = a[ao + 2]! - b[bo + 2]!
}

export function v3Scale(out: Float32Array, o: number, a: Float32Array, ao: number, s: number): void {
  out[o] = a[ao]! * s
  out[o + 1] = a[ao + 1]! * s
  out[o + 2] = a[ao + 2]! * s
}

export function v3Length(a: Float32Array, ao: number): number {
  const x = a[ao]!
  const y = a[ao + 1]!
  const z = a[ao + 2]!
  return Math.sqrt(x * x + y * y + z * z)
}

export function v3Normalize(out: Float32Array, o: number, a: Float32Array, ao: number): void {
  const len = v3Length(a, ao) || 1
  out[o] = a[ao]! / len
  out[o + 1] = a[ao + 1]! / len
  out[o + 2] = a[ao + 2]! / len
}

export function v3Dot(a: Float32Array, ao: number, b: Float32Array, bo: number): number {
  return a[ao]! * b[bo]! + a[ao + 1]! * b[bo + 1]! + a[ao + 2]! * b[bo + 2]!
}

export function v3Cross(out: Float32Array, o: number, a: Float32Array, ao: number, b: Float32Array, bo: number): void {
  const ax = a[ao]!,
    ay = a[ao + 1]!,
    az = a[ao + 2]!
  const bx = b[bo]!,
    by = b[bo + 1]!,
    bz = b[bo + 2]!
  out[o] = ay * bz - az * by
  out[o + 1] = az * bx - ax * bz
  out[o + 2] = ax * by - ay * bx
}

// ── mat4 (column-major, WebGPU convention) ────────────────────────────

export function m4Identity(out: Float32Array, o: number): void {
  out[o] = 1
  out[o + 1] = 0
  out[o + 2] = 0
  out[o + 3] = 0
  out[o + 4] = 0
  out[o + 5] = 1
  out[o + 6] = 0
  out[o + 7] = 0
  out[o + 8] = 0
  out[o + 9] = 0
  out[o + 10] = 1
  out[o + 11] = 0
  out[o + 12] = 0
  out[o + 13] = 0
  out[o + 14] = 0
  out[o + 15] = 1
}

export function m4Multiply(
  out: Float32Array,
  o: number,
  a: Float32Array,
  ao: number,
  b: Float32Array,
  bo: number,
): void {
  const a00 = a[ao]!,
    a01 = a[ao + 1]!,
    a02 = a[ao + 2]!,
    a03 = a[ao + 3]!
  const a10 = a[ao + 4]!,
    a11 = a[ao + 5]!,
    a12 = a[ao + 6]!,
    a13 = a[ao + 7]!
  const a20 = a[ao + 8]!,
    a21 = a[ao + 9]!,
    a22 = a[ao + 10]!,
    a23 = a[ao + 11]!
  const a30 = a[ao + 12]!,
    a31 = a[ao + 13]!,
    a32 = a[ao + 14]!,
    a33 = a[ao + 15]!

  for (let i = 0; i < 4; i++) {
    const bi = bo + i * 4
    const b0 = b[bi]!,
      b1 = b[bi + 1]!,
      b2 = b[bi + 2]!,
      b3 = b[bi + 3]!
    out[o + i * 4] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    out[o + i * 4 + 1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    out[o + i * 4 + 2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    out[o + i * 4 + 3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3
  }
}

export function m4Perspective(
  out: Float32Array,
  o: number,
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): void {
  const f = 1.0 / Math.tan(fovY / 2)
  const rangeInv = 1.0 / (near - far)

  out[o] = f / aspect
  out[o + 1] = 0
  out[o + 2] = 0
  out[o + 3] = 0
  out[o + 4] = 0
  out[o + 5] = f
  out[o + 6] = 0
  out[o + 7] = 0
  out[o + 8] = 0
  out[o + 9] = 0
  out[o + 10] = far * rangeInv
  out[o + 11] = -1
  out[o + 12] = 0
  out[o + 13] = 0
  out[o + 14] = near * far * rangeInv
  out[o + 15] = 0
}

export function m4LookAt(
  out: Float32Array,
  o: number,
  eye: Float32Array,
  eo: number,
  target: Float32Array,
  to: number,
  up: Float32Array,
  uo: number,
): void {
  // scratch space: use local vars (no allocation)
  let fx = target[to]! - eye[eo]!
  let fy = target[to + 1]! - eye[eo + 1]!
  let fz = target[to + 2]! - eye[eo + 2]!
  let len = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1
  fx /= len
  fy /= len
  fz /= len

  // side = normalize(cross(forward, up))
  let sx = fy * up[uo + 2]! - fz * up[uo + 1]!
  let sy = fz * up[uo]! - fx * up[uo + 2]!
  let sz = fx * up[uo + 1]! - fy * up[uo]!
  len = Math.sqrt(sx * sx + sy * sy + sz * sz) || 1
  sx /= len
  sy /= len
  sz /= len

  // u = cross(side, forward)
  const ux = sy * fz - sz * fy
  const uy = sz * fx - sx * fz
  const uz = sx * fy - sy * fx

  out[o] = sx
  out[o + 1] = ux
  out[o + 2] = -fx
  out[o + 3] = 0
  out[o + 4] = sy
  out[o + 5] = uy
  out[o + 6] = -fy
  out[o + 7] = 0
  out[o + 8] = sz
  out[o + 9] = uz
  out[o + 10] = -fz
  out[o + 11] = 0
  out[o + 12] = -(sx * eye[eo]! + sy * eye[eo + 1]! + sz * eye[eo + 2]!)
  out[o + 13] = -(ux * eye[eo]! + uy * eye[eo + 1]! + uz * eye[eo + 2]!)
  out[o + 14] = -(-fx * eye[eo]! + -fy * eye[eo + 1]! + -fz * eye[eo + 2]!)
  out[o + 15] = 1
}

// Extract 6 frustum planes from a VP matrix (column-major).
// out: 24 floats [nx,ny,nz,d] × 6 planes (left,right,bottom,top,near,far).
// Planes are normalized so distance tests work directly.
export function m4ExtractFrustumPlanes(out: Float32Array, m: Float32Array, mo: number): void {
  // Column-major row accessors: row i = m[mo+i], m[mo+i+4], m[mo+i+8], m[mo+i+12]
  const m0 = m[mo]!,
    m1 = m[mo + 1]!,
    m2 = m[mo + 2]!,
    m3 = m[mo + 3]!
  const m4 = m[mo + 4]!,
    m5 = m[mo + 5]!,
    m6 = m[mo + 6]!,
    m7 = m[mo + 7]!
  const m8 = m[mo + 8]!,
    m9 = m[mo + 9]!,
    m10 = m[mo + 10]!,
    m11 = m[mo + 11]!
  const m12 = m[mo + 12]!,
    m13 = m[mo + 13]!,
    m14 = m[mo + 14]!,
    m15 = m[mo + 15]!

  // Gribb-Hartmann: plane = row3 ± rowN
  // Left: row3 + row0
  setPlane(out, 0, m3 + m0, m7 + m4, m11 + m8, m15 + m12)
  // Right: row3 - row0
  setPlane(out, 4, m3 - m0, m7 - m4, m11 - m8, m15 - m12)
  // Bottom: row3 + row1
  setPlane(out, 8, m3 + m1, m7 + m5, m11 + m9, m15 + m13)
  // Top: row3 - row1
  setPlane(out, 12, m3 - m1, m7 - m5, m11 - m9, m15 - m13)
  // Near: row3 + row2
  setPlane(out, 16, m3 + m2, m7 + m6, m11 + m10, m15 + m14)
  // Far: row3 - row2
  setPlane(out, 20, m3 - m2, m7 - m6, m11 - m10, m15 - m14)
}

function setPlane(out: Float32Array, o: number, a: number, b: number, c: number, d: number): void {
  const len = Math.sqrt(a * a + b * b + c * c) || 1
  out[o] = a / len
  out[o + 1] = b / len
  out[o + 2] = c / len
  out[o + 3] = d / len
}

// Test sphere (cx,cy,cz,r) against 6 frustum planes.
// Returns true if the sphere is at least partially inside.
export function frustumContainsSphere(planes: Float32Array, cx: number, cy: number, cz: number, r: number): boolean {
  for (let i = 0; i < 6; i++) {
    const o = i * 4
    const dist = planes[o]! * cx + planes[o + 1]! * cy + planes[o + 2]! * cz + planes[o + 3]!
    if (dist < -r) return false
  }
  return true
}

// Convert glTF quaternion [x,y,z,w] to Euler angles matching m4FromTRS rotation order (Z * X * Y)
export function quatToEulerZXY(out: Float32Array, o: number, q: Float32Array, qo: number): void {
  const qx = q[qo]!,
    qy = q[qo + 1]!,
    qz = q[qo + 2]!,
    qw = q[qo + 3]!

  // R(2,1) = 2(qy*qz + qw*qx)
  const r21 = 2 * (qy * qz + qw * qx)
  // rx = asin(R(2,1)), clamped for numerical safety
  out[o] = Math.asin(Math.max(-1, Math.min(1, r21)))
  // ry = atan2(-R(2,0), R(2,2))
  out[o + 1] = Math.atan2(2 * (qw * qy - qx * qz), 1 - 2 * (qx * qx + qy * qy))
  // rz = atan2(-R(0,1), R(1,1))
  out[o + 2] = Math.atan2(2 * (qw * qz - qx * qy), 1 - 2 * (qx * qx + qz * qz))
}

// ── vec3 lerp ───────────────────────────────────────────────────────

export function v3Lerp(
  out: Float32Array,
  o: number,
  a: Float32Array,
  ao: number,
  b: Float32Array,
  bo: number,
  t: number,
): void {
  out[o] = a[ao]! + (b[bo]! - a[ao]!) * t
  out[o + 1] = a[ao + 1]! + (b[bo + 1]! - a[ao + 1]!) * t
  out[o + 2] = a[ao + 2]! + (b[bo + 2]! - a[ao + 2]!) * t
}

// ── quaternion slerp ────────────────────────────────────────────────

export function quatSlerp(
  out: Float32Array,
  o: number,
  a: Float32Array,
  ao: number,
  b: Float32Array,
  bo: number,
  t: number,
): void {
  let ax = a[ao]!,
    ay = a[ao + 1]!,
    az = a[ao + 2]!,
    aw = a[ao + 3]!
  let bx = b[bo]!,
    by = b[bo + 1]!,
    bz = b[bo + 2]!,
    bw = b[bo + 3]!

  let dot = ax * bx + ay * by + az * bz + aw * bw
  if (dot < 0) {
    bx = -bx
    by = -by
    bz = -bz
    bw = -bw
    dot = -dot
  }

  if (dot > 0.9995) {
    // Linear interpolation for very close quaternions
    out[o] = ax + (bx - ax) * t
    out[o + 1] = ay + (by - ay) * t
    out[o + 2] = az + (bz - az) * t
    out[o + 3] = aw + (bw - aw) * t
  } else {
    const theta = Math.acos(dot)
    const sinTheta = Math.sin(theta)
    const wa = Math.sin((1 - t) * theta) / sinTheta
    const wb = Math.sin(t * theta) / sinTheta
    out[o] = ax * wa + bx * wb
    out[o + 1] = ay * wa + by * wb
    out[o + 2] = az * wa + bz * wb
    out[o + 3] = aw * wa + bw * wb
  }

  // Normalize
  const nx = out[o]!,
    ny = out[o + 1]!,
    nz = out[o + 2]!,
    nw = out[o + 3]!
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw) || 1
  out[o] = nx / len
  out[o + 1] = ny / len
  out[o + 2] = nz / len
  out[o + 3] = nw / len
}

// ── mat4 from quaternion + TRS ──────────────────────────────────────

export function m4FromQuatTRS(
  out: Float32Array,
  o: number,
  pos: Float32Array,
  po: number,
  quat: Float32Array,
  qo: number,
  scl: Float32Array,
  so: number,
): void {
  const qx = quat[qo]!,
    qy = quat[qo + 1]!,
    qz = quat[qo + 2]!,
    qw = quat[qo + 3]!
  const sx = scl[so]!,
    sy = scl[so + 1]!,
    sz = scl[so + 2]!

  const x2 = qx + qx,
    y2 = qy + qy,
    z2 = qz + qz
  const xx = qx * x2,
    xy = qx * y2,
    xz = qx * z2
  const yy = qy * y2,
    yz = qy * z2,
    zz = qz * z2
  const wx = qw * x2,
    wy = qw * y2,
    wz = qw * z2

  out[o] = (1 - (yy + zz)) * sx
  out[o + 1] = (xy + wz) * sx
  out[o + 2] = (xz - wy) * sx
  out[o + 3] = 0

  out[o + 4] = (xy - wz) * sy
  out[o + 5] = (1 - (xx + zz)) * sy
  out[o + 6] = (yz + wx) * sy
  out[o + 7] = 0

  out[o + 8] = (xz + wy) * sz
  out[o + 9] = (yz - wx) * sz
  out[o + 10] = (1 - (xx + yy)) * sz
  out[o + 11] = 0

  out[o + 12] = pos[po]!
  out[o + 13] = pos[po + 1]!
  out[o + 14] = pos[po + 2]!
  out[o + 15] = 1
}

export function m4FromTRS(
  out: Float32Array,
  o: number,
  pos: Float32Array,
  po: number,
  rot: Float32Array,
  ro: number,
  scl: Float32Array,
  so: number,
): void {
  const rx = rot[ro]!,
    ry = rot[ro + 1]!,
    rz = rot[ro + 2]!
  const sx = scl[so]!,
    sy = scl[so + 1]!,
    sz = scl[so + 2]!

  const cx = Math.cos(rx),
    sxr = Math.sin(rx)
  const cy = Math.cos(ry),
    syr = Math.sin(ry)
  const cz = Math.cos(rz),
    szr = Math.sin(rz)

  // Rotation order: Z * X * Y  (Z-up: yaw around Z, pitch around X, roll around Y)
  out[o] = (cz * cy - szr * sxr * syr) * sx
  out[o + 1] = (szr * cy + cz * sxr * syr) * sx
  out[o + 2] = -cx * syr * sx
  out[o + 3] = 0

  out[o + 4] = -szr * cx * sy
  out[o + 5] = cz * cx * sy
  out[o + 6] = sxr * sy
  out[o + 7] = 0

  out[o + 8] = (cz * syr + szr * sxr * cy) * sz
  out[o + 9] = (szr * syr - cz * sxr * cy) * sz
  out[o + 10] = cx * cy * sz
  out[o + 11] = 0

  out[o + 12] = pos[po]!
  out[o + 13] = pos[po + 1]!
  out[o + 14] = pos[po + 2]!
  out[o + 15] = 1
}
