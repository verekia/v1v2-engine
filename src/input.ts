const TRACKED_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD'])

export class InputManager {
  keys = new Set<string>()

  private onDown = (e: KeyboardEvent) => {
    if (TRACKED_KEYS.has(e.code)) {
      e.preventDefault()
      this.keys.add(e.code)
    }
  }

  private onUp = (e: KeyboardEvent) => {
    this.keys.delete(e.code)
  }

  constructor() {
    window.addEventListener('keydown', this.onDown)
    window.addEventListener('keyup', this.onUp)
  }

  isDown(code: string): boolean {
    return this.keys.has(code)
  }

  destroy(): void {
    window.removeEventListener('keydown', this.onDown)
    window.removeEventListener('keyup', this.onUp)
  }
}
