import { m4Multiply } from './math.ts'

import type { Mesh } from './scene.ts'
import type { Scene } from './scene.ts'

// ── HtmlElement ──────────────────────────────────────────────────────────────

export interface HtmlElementOptions {
  position?: [number, number, number]
  offset?: [number, number, number]
  mesh?: Mesh
  distanceFactor?: number
  element?: HTMLElement
}

export class HtmlElement {
  readonly position: Float32Array
  readonly offset: Float32Array
  mesh: Mesh | null
  distanceFactor: number
  visible = true
  readonly element: HTMLElement
  /** @internal */
  readonly _wrapper: HTMLDivElement

  constructor(opts: HtmlElementOptions = {}) {
    this.position = new Float32Array(opts.position ?? [0, 0, 0])
    this.offset = new Float32Array(opts.offset ?? [0, 0, 0])
    this.mesh = opts.mesh ?? null
    this.distanceFactor = opts.distanceFactor ?? 0
    this.element = opts.element ?? document.createElement('div')

    this._wrapper = document.createElement('div')
    this._wrapper.style.cssText = 'position:absolute;left:0;top:0;will-change:transform;pointer-events:auto'
    this._wrapper.appendChild(this.element)
  }
}

// ── HtmlOverlay ──────────────────────────────────────────────────────────────

export class HtmlOverlay {
  private _container: HTMLDivElement
  private _elements: HtmlElement[] = []
  private _vpMatrix = new Float32Array(16)
  private _clipPos = new Float32Array(4)

  constructor(canvas: HTMLCanvasElement) {
    const parent = canvas.parentElement!
    if (getComputedStyle(parent).position === 'static') {
      parent.style.position = 'relative'
    }

    this._container = document.createElement('div')
    this._container.style.cssText = 'position:absolute;inset:0;overflow:hidden;pointer-events:none'
    canvas.insertAdjacentElement('afterend', this._container)
  }

  add(el: HtmlElement): HtmlElement {
    this._elements.push(el)
    this._container.appendChild(el._wrapper)
    return el
  }

  remove(el: HtmlElement): void {
    const idx = this._elements.indexOf(el)
    if (idx >= 0) {
      this._elements.splice(idx, 1)
      el._wrapper.remove()
    }
  }

  update(scene: Scene): void {
    const view = scene.viewMatrix
    const proj = scene.projMatrix
    const canvas = scene.canvas

    m4Multiply(this._vpMatrix, 0, proj, 0, view, 0)

    const halfW = canvas.clientWidth * 0.5
    const halfH = canvas.clientHeight * 0.5

    const camX = scene.camera.eye[0]!
    const camY = scene.camera.eye[1]!
    const camZ = scene.camera.eye[2]!

    const vp = this._vpMatrix
    const clip = this._clipPos

    for (let i = 0; i < this._elements.length; i++) {
      const el = this._elements[i]!
      const wrapper = el._wrapper

      if (!el.visible) {
        wrapper.style.display = 'none'
        continue
      }

      const src = el.mesh ? el.mesh.position : el.position
      const off = el.offset
      const px = src[0]! + off[0]!
      const py = src[1]! + off[1]!
      const pz = src[2]! + off[2]!

      // VP * [px, py, pz, 1]
      clip[0] = vp[0]! * px + vp[4]! * py + vp[8]! * pz + vp[12]!
      clip[1] = vp[1]! * px + vp[5]! * py + vp[9]! * pz + vp[13]!
      clip[2] = vp[2]! * px + vp[6]! * py + vp[10]! * pz + vp[14]!
      clip[3] = vp[3]! * px + vp[7]! * py + vp[11]! * pz + vp[15]!

      const w = clip[3]!
      if (w <= 0) {
        wrapper.style.display = 'none'
        continue
      }

      const ndcX = clip[0]! / w
      const ndcY = clip[1]! / w

      const screenX = (ndcX + 1) * halfW
      const screenY = (1 - ndcY) * halfH

      let scale = 1
      if (el.distanceFactor > 0) {
        const dx = px - camX
        const dy = py - camY
        const dz = pz - camZ
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1
        scale = el.distanceFactor / dist
      }

      wrapper.style.display = ''
      wrapper.style.transform = `translate3d(${screenX}px,${screenY}px,0) translate(-50%,-50%) scale(${scale})`
    }
  }

  destroy(): void {
    this._container.remove()
    this._elements.length = 0
  }
}
