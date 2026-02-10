import { TRANSFORM, MESH_INSTANCE, type World } from './ecs.ts'
import { m4Multiply, m4ExtractFrustumPlanes, frustumContainsSphere } from './math.ts'
import { lambertShader } from './shaders.ts'

const MODEL_SLOT_SIZE = 256 // minUniformBufferOffsetAlignment

interface GeometryGPU {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
  boundingRadius: number
}

export async function initGPU(canvas: HTMLCanvasElement) {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')
  const device = await adapter.requestDevice()
  const context = canvas.getContext('webgpu')!
  const format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'premultiplied' })
  return { device, context, format }
}

export class Renderer {
  private device: GPUDevice
  private context: GPUCanvasContext
  private pipeline: GPURenderPipeline
  private depthTexture: GPUTexture
  private depthView: GPUTextureView

  // Buffers
  private cameraBuffer: GPUBuffer
  private modelBuffer: GPUBuffer
  private lightingBuffer: GPUBuffer

  // Bind groups
  private cameraBG: GPUBindGroup
  private lightingBG: GPUBindGroup
  private modelBGL: GPUBindGroupLayout
  private modelBG: GPUBindGroup

  private geometries = new Map<number, GeometryGPU>()
  private maxEntities: number

  drawCalls = 0

  // Scratch buffers (no per-frame allocation)
  private vpMat = new Float32Array(16)
  private frustumPlanes = new Float32Array(24) // 6 planes × 4 floats

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    maxEntities = 1000,
  ) {
    this.device = device
    this.context = context
    this.maxEntities = maxEntities

    // ── Shader module ─────────────────────────────────────────────────
    const shaderModule = device.createShaderModule({ code: lambertShader })

    // ── Bind group layouts ────────────────────────────────────────────
    const cameraBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    })
    this.modelBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true },
        },
      ],
    })
    const lightingBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL, lightingBGL],
    })

    // ── Pipeline ──────────────────────────────────────────────────────
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 24, // 6 floats * 4 bytes
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs',
        targets: [{ format }],
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    })

    // ── Depth texture ─────────────────────────────────────────────────
    this.depthTexture = this.createDepthTexture(canvas.width, canvas.height)
    this.depthView = this.depthTexture.createView()

    // ── Uniform buffers ───────────────────────────────────────────────
    // Camera: 2 mat4 = 128 bytes
    this.cameraBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Model: MODEL_SLOT_SIZE per entity
    this.modelBuffer = device.createBuffer({
      size: MODEL_SLOT_SIZE * maxEntities,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Lighting: 3 vec4 = 48 bytes
    this.lightingBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // ── Bind groups ───────────────────────────────────────────────────
    this.cameraBG = device.createBindGroup({
      layout: cameraBGL,
      entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }],
    })
    this.modelBG = device.createBindGroup({
      layout: this.modelBGL,
      entries: [{ binding: 0, resource: { buffer: this.modelBuffer, size: MODEL_SLOT_SIZE } }],
    })
    this.lightingBG = device.createBindGroup({
      layout: lightingBGL,
      entries: [{ binding: 0, resource: { buffer: this.lightingBuffer } }],
    })
  }

  private createDepthTexture(w: number, h: number): GPUTexture {
    return this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
  }

  resize(w: number, h: number): void {
    this.depthTexture.destroy()
    this.depthTexture = this.createDepthTexture(w, h)
    this.depthView = this.depthTexture.createView()
  }

  registerGeometry(id: number, vertices: Float32Array, indices: Uint16Array): void {
    const vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(vertexBuffer, 0, vertices.buffer as ArrayBuffer, vertices.byteOffset, vertices.byteLength)

    const indexBuffer = this.device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(indexBuffer, 0, indices.buffer as ArrayBuffer, indices.byteOffset, indices.byteLength)

    // Compute bounding sphere radius from vertex positions (stride = 6 floats)
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 6) {
      const x = vertices[i]!, y = vertices[i + 1]!, z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    this.geometries.set(id, { vertexBuffer, indexBuffer, indexCount: indices.length, boundingRadius: Math.sqrt(maxR2) })
  }

  render(world: World): void {
    const device = this.device
    const cam = world.activeCamera
    if (cam < 0) return

    // Upload camera (view + proj = 128 bytes)
    device.queue.writeBuffer(this.cameraBuffer, 0, world.viewMatrices, cam * 16, 16)
    device.queue.writeBuffer(this.cameraBuffer, 64, world.projMatrices, cam * 16, 16)

    // Upload lighting (direction: vec4, dirColor: vec4, ambient: vec4 = 48 bytes)
    const lightData = new Float32Array(12)
    lightData.set(world.directionalLightDir, 0)   // vec4 slot 0 (w=0)
    lightData.set(world.directionalLightColor, 4)  // vec4 slot 1 (w=0)
    lightData.set(world.ambientLightColor, 8)      // vec4 slot 2 (w=0)
    device.queue.writeBuffer(this.lightingBuffer, 0, lightData)

    // ── Frustum culling setup ───────────────────────────────────────────
    // VP = projection * view
    m4Multiply(this.vpMat, 0, world.projMatrices, cam * 16, world.viewMatrices, cam * 16)
    m4ExtractFrustumPlanes(this.frustumPlanes, this.vpMat, 0)
    const planes = this.frustumPlanes

    // Upload per-entity model data (only visible entities)
    const meshMask = TRANSFORM | MESH_INSTANCE
    const modelSlot = new Float32Array(MODEL_SLOT_SIZE / 4) // 64 floats
    for (let i = 0; i < world.entityCount; i++) {
      if ((world.componentMask[i]! & meshMask) !== meshMask) continue

      // Frustum cull: bounding sphere in world space
      const geo = this.geometries.get(world.geometryIds[i]!)
      if (!geo) continue
      const si = i * 3
      const sx = Math.abs(world.scales[si]!)
      const sy = Math.abs(world.scales[si + 1]!)
      const sz = Math.abs(world.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : (sy > sz ? sy : sz)
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, world.positions[si]!, world.positions[si + 1]!, world.positions[si + 2]!, r)) continue

      // worldMatrix (16 floats = 64 bytes)
      for (let j = 0; j < 16; j++) modelSlot[j] = world.worldMatrices[i * 16 + j]!
      // color (3 floats → vec4f at offset 16)
      modelSlot[16] = world.colors[i * 3]!
      modelSlot[17] = world.colors[i * 3 + 1]!
      modelSlot[18] = world.colors[i * 3 + 2]!
      modelSlot[19] = 1.0
      device.queue.writeBuffer(this.modelBuffer, i * MODEL_SLOT_SIZE, modelSlot, 0, 20)
    }

    // ── Render pass ───────────────────────────────────────────────────
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.15, g: 0.15, b: 0.2, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.cameraBG)
    pass.setBindGroup(2, this.lightingBG)

    let draws = 0
    for (let i = 0; i < world.entityCount; i++) {
      if ((world.componentMask[i]! & meshMask) !== meshMask) continue
      const geo = this.geometries.get(world.geometryIds[i]!)
      if (!geo) continue

      // Frustum cull (same test as above)
      const si = i * 3
      const sx = Math.abs(world.scales[si]!)
      const sy = Math.abs(world.scales[si + 1]!)
      const sz = Math.abs(world.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : (sy > sz ? sy : sz)
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, world.positions[si]!, world.positions[si + 1]!, world.positions[si + 2]!, r)) continue

      pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
      pass.setVertexBuffer(0, geo.vertexBuffer)
      pass.setIndexBuffer(geo.indexBuffer, 'uint16')
      pass.drawIndexed(geo.indexCount)
      draws++
    }
    this.drawCalls = draws

    pass.end()
    device.queue.submit([encoder.finish()])
  }
}
