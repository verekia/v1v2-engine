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

  // Multi-touch state
  private activePointers = new Map<number, { x: number; y: number }>()
  private prevPinchDist = 0

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    canvas.style.touchAction = 'none'
    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('pointerleave', this.onPointerUp)
    canvas.addEventListener('pointercancel', this.onPointerUp)
    canvas.addEventListener('wheel', this.onWheel, { passive: false })
    canvas.addEventListener('contextmenu', e => e.preventDefault())
    this.updateEye()
  }

  private pinchDist(): number {
    const pts = [...this.activePointers.values()]
    const dx = pts[1]!.x - pts[0]!.x
    const dy = pts[1]!.y - pts[0]!.y
    return Math.sqrt(dx * dx + dy * dy)
  }

  private pinchCenter(): { x: number; y: number } {
    const pts = [...this.activePointers.values()]
    return { x: (pts[0]!.x + pts[1]!.x) / 2, y: (pts[0]!.y + pts[1]!.y) / 2 }
  }

  private onPointerDown = (e: PointerEvent) => {
    this.activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
    this.canvas.setPointerCapture(e.pointerId)

    if (this.activePointers.size === 2) {
      // Switch to two-finger mode (pan + pinch-zoom)
      this.dragging = 'pan'
      this.prevPinchDist = this.pinchDist()
      const c = this.pinchCenter()
      this.prevX = c.x
      this.prevY = c.y
    } else if (this.activePointers.size === 1) {
      // Single pointer: left = rotate, right/middle = pan
      if (e.button === 0) this.dragging = 'rotate'
      else if (e.button === 1 || e.button === 2) this.dragging = 'pan'
      this.prevX = e.clientX
      this.prevY = e.clientY
    }
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.activePointers.has(e.pointerId)) return
    this.activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY })

    if (!this.dragging) return

    if (this.activePointers.size === 2) {
      // Pinch zoom
      const dist = this.pinchDist()
      if (this.prevPinchDist > 0) {
        const scale = this.prevPinchDist / dist
        this.radius *= scale
        if (this.radius < this.minRadius) this.radius = this.minRadius
        if (this.radius > this.maxRadius) this.radius = this.maxRadius
      }
      this.prevPinchDist = dist

      // Two-finger pan
      const c = this.pinchCenter()
      const dx = c.x - this.prevX
      const dy = c.y - this.prevY
      this.prevX = c.x
      this.prevY = c.y
      this.applyPan(dx, dy)
    } else if (this.activePointers.size === 1) {
      const dx = e.clientX - this.prevX
      const dy = e.clientY - this.prevY
      this.prevX = e.clientX
      this.prevY = e.clientY

      if (this.dragging === 'rotate') {
        this.theta += dx * this.rotateSensitivity
        this.phi += dy * this.rotateSensitivity
        if (this.theta > TWO_PI) this.theta -= TWO_PI
        if (this.theta < 0) this.theta += TWO_PI
        if (this.phi < -HALF_PI) this.phi = -HALF_PI
        if (this.phi > HALF_PI) this.phi = HALF_PI
      } else {
        this.applyPan(dx, dy)
      }
    }

    this.updateEye()
  }

  private applyPan(dx: number, dy: number): void {
    const sinT = Math.sin(this.theta)
    const cosT = Math.cos(this.theta)
    const rx = cosT
    const ry = -sinT
    const panScale = this.panSensitivity * this.radius * 0.1
    this.targetX += dx * rx * panScale
    this.targetY += dx * ry * panScale
    this.targetZ += dy * panScale
  }

  private onPointerUp = (e: PointerEvent) => {
    this.activePointers.delete(e.pointerId)
    if (this.activePointers.size === 0) {
      this.dragging = null
    } else if (this.activePointers.size === 1) {
      // Dropped back to one finger — switch to rotate, reset prev position
      this.dragging = 'rotate'
      const remaining = [...this.activePointers.values()][0]!
      this.prevX = remaining.x
      this.prevY = remaining.y
    }
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
    this.canvas.removeEventListener('pointercancel', this.onPointerUp)
    this.canvas.removeEventListener('wheel', this.onWheel)
  }
}
