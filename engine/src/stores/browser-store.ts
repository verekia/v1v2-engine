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
  isLeftMouseDown?: boolean
  isMiddleMouseDown?: boolean
  isRightMouseDown?: boolean
  mobileJoystick1?: any
  mobileJoystick2?: any
  setPointerLocked: (isPointerLocked: boolean) => void
  setFullscreen: (isFullscreen: boolean) => void
  setPageVisible: (isPageVisible: boolean) => void
  setSize: (width: number, height: number) => void
  setCanHover: (canHover: boolean) => void
  setMousePosition: (x: number, y: number) => void
  setMouseMovement: (x: number, y: number) => void
  setLeftMouseDown: (isLeftMouseDown: boolean) => void
  setMiddleMouseDown: (isMiddleMouseDown: boolean) => void
  setRightMouseDown: (isRightMouseDown: boolean) => void
  setMobileJoystick1: (joystickData: any) => void
  setMobileJoystick2: (joystickData: any) => void
}

export const liveBrowserState = {
  width: undefined as number | undefined,
  height: undefined as number | undefined,
  mouseX: undefined as number | undefined,
  mouseY: undefined as number | undefined,
  mouseMovementX: undefined as number | undefined,
  mouseMovementY: undefined as number | undefined,
  mobileJoystick1: undefined as any,
  mobileJoystick2: undefined as any,
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
  isLeftMouseDown: false,
  isMiddleMouseDown: false,
  isRightMouseDown: false,
  mobileJoystick1: undefined,
  mobileJoystick2: undefined,
  setPointerLocked: isPointerLocked => set(() => ({ isPointerLocked })),
  setFullscreen: isFullscreen => set(() => ({ isFullscreen })),
  setPageVisible: isPageVisible => set(() => ({ isPageVisible })),
  setSize: (width, height) => set(() => ({ width, height })),
  setCanHover: canHover => set(() => ({ canHover })),
  setMousePosition: (mouseX, mouseY) => set(() => ({ mouseX, mouseY })),
  setMouseMovement: (mouseMovementX, mouseMovementY) =>
    set(() => ({ mouseMovementX, mouseMovementY })),
  setLeftMouseDown: isLeftMouseDown => set(() => ({ isLeftMouseDown })),
  setMiddleMouseDown: isMiddleMouseDown => set(() => ({ isMiddleMouseDown })),
  setRightMouseDown: isRightMouseDown => set(() => ({ isRightMouseDown })),
  setMobileJoystick1: joystickData => set(() => ({ mobileJoystick1: joystickData })),
  setMobileJoystick2: joystickData => set(() => ({ mobileJoystick2: joystickData })),
}))

export const getBrowserState = () => useBrowserStore.getState()