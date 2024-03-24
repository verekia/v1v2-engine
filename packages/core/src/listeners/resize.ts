import { mp } from '../store'

export type ResizeListenerProps = {
  onResize?: ({
    width,
    height,
    isLandscape,
    isPortrait,
  }: {
    width: number
    height: number
    isLandscape: boolean
    isPortrait: boolean
  }) => void
}

export const mountResizeListener = ({ onResize }: ResizeListenerProps) => {
  const handler = () => {
    const width = window.innerWidth
    const height = window.innerHeight

    mp().windowWidth = width
    mp().windowHeight = height
    const isPortrait = height >= width
    const isLandscape = width > height
    mp().isPortrait = isPortrait
    mp().isLandscape = isLandscape
    onResize?.({ width, height, isPortrait, isLandscape })
  }

  handler()

  window.addEventListener('resize', handler)

  return () => window.removeEventListener('resize', handler)
}
