import type { Scene } from './scene.ts'

// ── Types ────────────────────────────────────────────────────────────────────

export interface SchedulerState {
  /** The scene instance */
  scene: Scene
  /** Delta time in seconds since last frame (capped at 0.1s) */
  dt: number
  /** Total elapsed time in seconds since scheduler start */
  elapsed: number
  /** Frame number (increments every rAF tick) */
  frame: number
}

export interface SchedulerCallbackOptions {
  /** Execution order among callbacks. Lower values run first. Default: 0. Can be negative. */
  priority?: number
  /** Throttle this callback to at most N executions per second. 0 or omitted = every frame. */
  fps?: number
}

export type SchedulerCallback = (state: SchedulerState) => void

// ── Internal entry ───────────────────────────────────────────────────────────

interface SchedulerEntry {
  callback: SchedulerCallback | null // null = removed, compacted on next sort pass
  priority: number
  fpsInterval: number // 0 = no throttle, else milliseconds between calls
  lastRunTime: number
}

// ── Scheduler ────────────────────────────────────────────────────────────────

export class Scheduler {
  private _scene: Scene
  private _entries: SchedulerEntry[] = []
  private _sorted = true
  private _rafId = 0
  private _lastTime = 0
  private _elapsed = 0
  private _frame = 0
  private _running = false

  // Reusable state object — avoids per-frame allocation
  private _state: SchedulerState

  constructor(scene: Scene) {
    this._scene = scene
    this._state = { scene, dt: 0, elapsed: 0, frame: 0 }
  }

  /**
   * Register a callback to be called each frame (or throttled via `fps`).
   * Returns an unsubscribe function that removes the callback.
   */
  register(callback: SchedulerCallback, options?: SchedulerCallbackOptions): () => void {
    const entry: SchedulerEntry = {
      callback,
      priority: options?.priority ?? 0,
      fpsInterval: options?.fps ? 1000 / options.fps : 0,
      lastRunTime: 0,
    }
    this._entries.push(entry)
    this._sorted = false

    return () => {
      if (!entry.callback) return // already removed
      entry.callback = null
      this._sorted = false // trigger compaction on next frame
    }
  }

  /** Start the rAF loop. No-op if already running. */
  start(): void {
    if (this._running) return
    this._running = true
    this._lastTime = performance.now()
    this._rafId = requestAnimationFrame(this._loop)
  }

  /** Stop the rAF loop. Can be resumed with start(). */
  stop(): void {
    this._running = false
    if (this._rafId) {
      cancelAnimationFrame(this._rafId)
      this._rafId = 0
    }
  }

  /** Stop the loop and remove all callbacks. */
  destroy(): void {
    this.stop()
    this._entries.length = 0
  }

  private _loop = (now: number): void => {
    if (!this._running) return

    const dt = Math.min((now - this._lastTime) / 1000, 0.1)
    this._lastTime = now
    this._elapsed += dt
    this._frame++

    // Compact dead entries + sort by priority when entries have changed
    if (!this._sorted) {
      let write = 0
      for (let read = 0; read < this._entries.length; read++) {
        if (this._entries[read]!.callback) this._entries[write++] = this._entries[read]!
      }
      this._entries.length = write
      this._entries.sort((a, b) => a.priority - b.priority)
      this._sorted = true
    }

    // Update shared state object
    const state = this._state
    state.dt = dt
    state.elapsed = this._elapsed
    state.frame = this._frame

    // Execute callbacks in priority order.
    // Cache length so mid-frame registrations don't execute this frame.
    const entries = this._entries
    const len = entries.length
    for (let i = 0; i < len; i++) {
      const entry = entries[i]!
      if (!entry.callback) continue // removed mid-frame

      // Per-callback FPS throttle with corrected dt
      if (entry.fpsInterval > 0) {
        if (entry.lastRunTime > 0 && now - entry.lastRunTime < entry.fpsInterval) continue
        if (entry.lastRunTime > 0) {
          state.dt = Math.min((now - entry.lastRunTime) / 1000, 0.1)
        }
        entry.lastRunTime = now
        entry.callback(state)
        state.dt = dt
      } else {
        entry.callback(state)
      }
    }

    this._rafId = requestAnimationFrame(this._loop)
  }
}
