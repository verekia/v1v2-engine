const TWO_PI = Math.PI * 2
const HALF_PI = Math.PI / 2 - 0.001 // clamp to avoid gimbal flip

export class OrbitControls {
  // Spherical coordinates
  theta = Math.PI // horizontal angle (radians)
  phi = Math.PI / 6 // vertical angle (radians, 0 = horizon, +PI/2 = above)
  radius = 10

  // Target (orbit center)
  targetX = 0
  targetY = 0
  targetZ = 0

  // Computed eye position (written each update)
  readonly eye = new Float32Array(3)
  readonly target = new Float32Array(3)

  // Sensitivity
  rotateSensitivity = 0.005
  panSensitivity = 0.008
  zoomSensitivity = 0.001
  minRadius = 1
  maxRadius = 200

  private canvas: HTMLCanvasElement
  private dragging: 'rotate' | 'pan' | null = null
  private prevX = 0
  private prevY = 0

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('pointerleave', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel, { passive: false })
    canvas.addEventListener('contextmenu', e => e.preventDefault())
    this.updateEye()
  }

  private onPointerDown = (e: PointerEvent) => {
    // Left button = rotate, right or middle = pan
    if (e.button === 0) this.dragging = 'rotate'
    else if (e.button === 1 || e.button === 2) this.dragging = 'pan'
    this.prevX = e.clientX
    this.prevY = e.clientY
    this.canvas.setPointerCapture(e.pointerId)
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.dragging) return
    const dx = e.clientX - this.prevX
    const dy = e.clientY - this.prevY
    this.prevX = e.clientX
    this.prevY = e.clientY

    if (this.dragging === 'rotate') {
      this.theta += dx * this.rotateSensitivity
      this.phi += dy * this.rotateSensitivity
      // Wrap theta
      if (this.theta > TWO_PI) this.theta -= TWO_PI
      if (this.theta < 0) this.theta += TWO_PI
      // Clamp phi to avoid flipping
      if (this.phi < -HALF_PI) this.phi = -HALF_PI
      if (this.phi > HALF_PI) this.phi = HALF_PI
    } else {
      // Pan: move target along camera-local right and up
      const sinT = Math.sin(this.theta)
      const cosT = Math.cos(this.theta)
      // Right vector (in xy plane)
      const rx = cosT
      const ry = -sinT
      // Up vector: world Z for simplicity
      const panScale = this.panSensitivity * this.radius * 0.1
      this.targetX += dx * rx * panScale
      this.targetY += dx * ry * panScale
      this.targetZ += dy * panScale
    }

    this.updateEye()
  }

  private onPointerUp = (_e: PointerEvent) => {
    this.dragging = null
  }

  private onWheel = (e: WheelEvent) => {
    e.preventDefault()
    this.radius *= 1 + e.deltaY * this.zoomSensitivity
    if (this.radius < this.minRadius) this.radius = this.minRadius
    if (this.radius > this.maxRadius) this.radius = this.maxRadius
    this.updateEye()
  }

  updateEye(): void {
    const cosPhi = Math.cos(this.phi)
    this.eye[0] = this.targetX + this.radius * cosPhi * Math.sin(this.theta)
    this.eye[1] = this.targetY + this.radius * cosPhi * Math.cos(this.theta)
    this.eye[2] = this.targetZ + this.radius * Math.sin(this.phi)
    this.target[0] = this.targetX
    this.target[1] = this.targetY
    this.target[2] = this.targetZ
  }

  destroy(): void {
    this.canvas.removeEventListener('pointerdown', this.onPointerDown)
    this.canvas.removeEventListener('pointermove', this.onPointerMove)
    this.canvas.removeEventListener('pointerup', this.onPointerUp)
    this.canvas.removeEventListener('pointerleave', this.onPointerUp)
    this.canvas.removeEventListener('wheel', this.onWheel)
  }
}
