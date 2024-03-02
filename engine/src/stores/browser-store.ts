import { create } from 'zustand'

type BrowserStore = {
  isPointerLocked: boolean
  isFullscreen: boolean
  isPageVisible: boolean
  canHover?: boolean
  width?: number
  height?: number
  mouseX?: number
  mouseY?: number
  mouseMovementX?: number
  mouseMovementY?: number
  setPointerLocked: (isPointerLocked: boolean) => void
  setFullscreen: (isFullscreen: boolean) => void
  setPageVisible: (isPageVisible: boolean) => void
  setSize: (width: number, height: number) => void
  setCanHover: (canHover: boolean) => void
  setMousePosition: (x: number, y: number) => void
  setMouseMovement: (x: number, y: number) => void
}

export const liveBrowserState = {
  width: undefined as number | undefined,
  height: undefined as number | undefined,
  mouseX: undefined as number | undefined,
  mouseY: undefined as number | undefined,
  mouseMovementX: undefined as number | undefined,
  mouseMovementY: undefined as number | undefined,
}

export const useBrowserStore = create<BrowserStore>(set => ({
  isPointerLocked: false,
  isFullscreen: false,
  isPageVisible: true,
  width: undefined,
  height: undefined,
  canHover: undefined,
  mouseX: undefined,
  mouseY: undefined,
  mouseMovementX: undefined,
  mouseMovementY: undefined,
  setPointerLocked: isPointerLocked => set(() => ({ isPointerLocked })),
  setFullscreen: isFullscreen => set(() => ({ isFullscreen })),
  setPageVisible: isPageVisible => set(() => ({ isPageVisible })),
  setSize: (width, height) => set(() => ({ width, height })),
  setCanHover: canHover => set(() => ({ canHover })),
  setMousePosition: (mouseX, mouseY) => set(() => ({ mouseX, mouseY })),
  setMouseMovement: (mouseMovementX, mouseMovementY) =>
    set(() => ({ mouseMovementX, mouseMovementY })),
}))

export const getBrowserState = () => useBrowserStore.getState()
