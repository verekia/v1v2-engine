import { Renderer } from './renderer.ts'
import { WebGLRenderer } from './webgl-renderer.ts'

import type { BackendType, IRenderer } from './renderer.ts'

export async function createRenderer(
  canvas: HTMLCanvasElement,
  maxEntities = 1000,
  forceBackend?: BackendType,
  maxSkinnedEntities?: number,
): Promise<IRenderer> {
  // Try WebGPU first (unless forced to WebGL)
  if (forceBackend !== 'webgl' && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter()
      if (adapter) {
        const device = await adapter.requestDevice()
        const context = canvas.getContext('webgpu')
        if (context) {
          const format = navigator.gpu.getPreferredCanvasFormat()
          context.configure({ device, format, alphaMode: 'premultiplied' })
          return new Renderer(device, context, format, canvas, maxEntities, maxSkinnedEntities)
        }
      }
    } catch {
      // Fall through to WebGL
    }
  }

  // Fallback to WebGL
  const gl = canvas.getContext('webgl2', { depth: true })
  if (!gl) throw new Error('Neither WebGPU nor WebGL is supported')
  return new WebGLRenderer(gl, canvas, maxEntities, maxSkinnedEntities)
}
