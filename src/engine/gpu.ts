export async function initGPU(canvas: HTMLCanvasElement) {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')
  const device = await adapter.requestDevice()
  const context = canvas.getContext('webgpu')!
  const format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'premultiplied' })
  return { device, context, format }
}
