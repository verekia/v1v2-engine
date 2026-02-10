import { m4Multiply, m4ExtractFrustumPlanes, frustumContainsSphere } from './math.ts'
import { lambertShader, skinnedLambertShader } from './shaders.ts'
import type { SkinInstance } from './skin.ts'

const MODEL_SLOT_SIZE = 256 // minUniformBufferOffsetAlignment
const MAX_JOINTS = 128
const JOINT_SLOT_SIZE = MAX_JOINTS * 64 // 128 mat4 * 64 bytes = 8192 (already 256-aligned)
const MAX_SKINNED_ENTITIES = 64

export interface RenderScene {
  cameraView: Float32Array
  cameraViewOffset: number
  cameraProj: Float32Array
  cameraProjOffset: number
  lightDirection: Float32Array
  lightDirColor: Float32Array
  lightAmbientColor: Float32Array
  entityCount: number
  renderMask: Uint8Array
  skinnedMask: Uint8Array
  positions: Float32Array
  scales: Float32Array
  worldMatrices: Float32Array
  colors: Float32Array
  geometryIds: Uint8Array
  skinInstanceIds: Int16Array
  skinInstances: SkinInstance[]
}

interface GeometryGPU {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
  indexFormat: GPUIndexFormat
  boundingRadius: number
}

interface SkinnedGeometryGPU extends GeometryGPU {
  skinBuffer: GPUBuffer
}

export class Renderer {
  private device: GPUDevice
  private context: GPUCanvasContext
  private pipeline: GPURenderPipeline
  private skinnedPipeline: GPURenderPipeline
  private depthTexture: GPUTexture
  private depthView: GPUTextureView

  // Buffers
  private cameraBuffer: GPUBuffer
  private modelBuffer: GPUBuffer
  private lightingBuffer: GPUBuffer
  private jointBuffer: GPUBuffer

  // Bind groups
  private cameraBG: GPUBindGroup
  private lightingBG: GPUBindGroup
  private modelBGL: GPUBindGroupLayout
  private modelBG: GPUBindGroup
  private jointBG: GPUBindGroup

  private geometries = new Map<number, GeometryGPU>()
  private skinnedGeometries = new Map<number, SkinnedGeometryGPU>()
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

    // ── Shader modules ─────────────────────────────────────────────
    const shaderModule = device.createShaderModule({ code: lambertShader })
    const skinnedShaderModule = device.createShaderModule({ code: skinnedLambertShader })

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
    const jointBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage', hasDynamicOffset: true },
        },
      ],
    })

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL, lightingBGL],
    })
    const skinnedPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL, lightingBGL, jointBGL],
    })

    // ── Static pipeline ────────────────────────────────────────────
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 36, // 9 floats * 4 bytes
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
              { shaderLocation: 2, offset: 24, format: 'float32x3' }, // color
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

    // ── Skinned pipeline ───────────────────────────────────────────
    this.skinnedPipeline = device.createRenderPipeline({
      layout: skinnedPipelineLayout,
      vertex: {
        module: skinnedShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 36,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
              { shaderLocation: 2, offset: 24, format: 'float32x3' }, // color
            ],
          },
          {
            arrayStride: 20, // 4 bytes joints + 16 bytes weights
            attributes: [
              { shaderLocation: 3, offset: 0, format: 'uint8x4' },    // joints
              { shaderLocation: 4, offset: 4, format: 'float32x4' },  // weights
            ],
          },
        ],
      },
      fragment: {
        module: skinnedShaderModule,
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
    // Joint matrices storage buffer
    this.jointBuffer = device.createBuffer({
      size: JOINT_SLOT_SIZE * MAX_SKINNED_ENTITIES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
    this.jointBG = device.createBindGroup({
      layout: jointBGL,
      entries: [{ binding: 0, resource: { buffer: this.jointBuffer, size: JOINT_SLOT_SIZE } }],
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

  registerGeometry(id: number, vertices: Float32Array, indices: Uint16Array | Uint32Array): void {
    const vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(vertexBuffer, 0, vertices.buffer as ArrayBuffer, vertices.byteOffset, vertices.byteLength)

    const indexByteSize = (indices.byteLength + 3) & ~3 // align to 4 bytes
    const indexBuffer = this.device.createBuffer({
      size: indexByteSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    const indexCopy = new Uint8Array(indexByteSize)
    indexCopy.set(new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength))
    this.device.queue.writeBuffer(indexBuffer, 0, indexCopy.buffer as ArrayBuffer, 0, indexByteSize)

    // Compute bounding sphere radius from vertex positions (stride = 9 floats)
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 9) {
      const x = vertices[i]!, y = vertices[i + 1]!, z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    const indexFormat: GPUIndexFormat = indices instanceof Uint32Array ? 'uint32' : 'uint16'
    this.geometries.set(id, { vertexBuffer, indexBuffer, indexCount: indices.length, indexFormat, boundingRadius: Math.sqrt(maxR2) })
  }

  registerSkinnedGeometry(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    joints: Uint8Array,
    weights: Float32Array,
  ): void {
    const device = this.device

    // Vertex buffer 0 (position, normal, color)
    const vertexBuffer = device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(vertexBuffer, 0, vertices.buffer as ArrayBuffer, vertices.byteOffset, vertices.byteLength)

    // Index buffer
    const indexByteSize = (indices.byteLength + 3) & ~3
    const indexBuffer = device.createBuffer({
      size: indexByteSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    const indexCopy = new Uint8Array(indexByteSize)
    indexCopy.set(new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength))
    device.queue.writeBuffer(indexBuffer, 0, indexCopy.buffer as ArrayBuffer, 0, indexByteSize)

    // Vertex buffer 1: interleaved [joints uint8x4, weights float32x4] = 20 bytes per vertex
    const numVertices = joints.length / 4
    const skinData = new ArrayBuffer(numVertices * 20)
    const skinView = new DataView(skinData)
    for (let i = 0; i < numVertices; i++) {
      const off = i * 20
      skinView.setUint8(off, joints[i * 4]!)
      skinView.setUint8(off + 1, joints[i * 4 + 1]!)
      skinView.setUint8(off + 2, joints[i * 4 + 2]!)
      skinView.setUint8(off + 3, joints[i * 4 + 3]!)
      skinView.setFloat32(off + 4, weights[i * 4]!, true)
      skinView.setFloat32(off + 8, weights[i * 4 + 1]!, true)
      skinView.setFloat32(off + 12, weights[i * 4 + 2]!, true)
      skinView.setFloat32(off + 16, weights[i * 4 + 3]!, true)
    }
    const skinBuffer = device.createBuffer({
      size: skinData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(skinBuffer, 0, skinData)

    // Bounding sphere
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 9) {
      const x = vertices[i]!, y = vertices[i + 1]!, z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    const indexFormat: GPUIndexFormat = indices instanceof Uint32Array ? 'uint32' : 'uint16'
    this.skinnedGeometries.set(id, {
      vertexBuffer,
      indexBuffer,
      indexCount: indices.length,
      indexFormat,
      boundingRadius: Math.sqrt(maxR2),
      skinBuffer,
    })
  }

  render(scene: RenderScene): void {
    const device = this.device

    // Upload camera (view + proj = 128 bytes)
    device.queue.writeBuffer(this.cameraBuffer, 0, scene.cameraView.buffer as ArrayBuffer, scene.cameraView.byteOffset + scene.cameraViewOffset * 4, 64)
    device.queue.writeBuffer(this.cameraBuffer, 64, scene.cameraProj.buffer as ArrayBuffer, scene.cameraProj.byteOffset + scene.cameraProjOffset * 4, 64)

    // Upload lighting (direction: vec4, dirColor: vec4, ambient: vec4 = 48 bytes)
    const lightData = new Float32Array(12)
    lightData.set(scene.lightDirection, 0)   // vec4 slot 0 (w=0)
    lightData.set(scene.lightDirColor, 4)    // vec4 slot 1 (w=0)
    lightData.set(scene.lightAmbientColor, 8) // vec4 slot 2 (w=0)
    device.queue.writeBuffer(this.lightingBuffer, 0, lightData)

    // ── Frustum culling setup ───────────────────────────────────────────
    // VP = projection * view
    m4Multiply(this.vpMat, 0, scene.cameraProj, scene.cameraProjOffset, scene.cameraView, scene.cameraViewOffset)
    m4ExtractFrustumPlanes(this.frustumPlanes, this.vpMat, 0)
    const planes = this.frustumPlanes

    // Upload per-entity model data (only visible entities)
    const modelSlot = new Float32Array(MODEL_SLOT_SIZE / 4) // 64 floats
    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue

      // Frustum cull: bounding sphere in world space
      const isSkinned = !!scene.skinnedMask[i]
      const geo = isSkinned
        ? this.skinnedGeometries.get(scene.geometryIds[i]!)
        : this.geometries.get(scene.geometryIds[i]!)
      if (!geo) continue
      const si = i * 3
      const sx = Math.abs(scene.scales[si]!)
      const sy = Math.abs(scene.scales[si + 1]!)
      const sz = Math.abs(scene.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : (sy > sz ? sy : sz)
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r)) continue

      // worldMatrix (16 floats = 64 bytes)
      for (let j = 0; j < 16; j++) modelSlot[j] = scene.worldMatrices[i * 16 + j]!
      // color (3 floats → vec4f at offset 16)
      modelSlot[16] = scene.colors[i * 3]!
      modelSlot[17] = scene.colors[i * 3 + 1]!
      modelSlot[18] = scene.colors[i * 3 + 2]!
      modelSlot[19] = 1.0
      device.queue.writeBuffer(this.modelBuffer, i * MODEL_SLOT_SIZE, modelSlot, 0, 20)
    }

    // Upload joint matrices for skinned entities
    let skinnedSlot = 0
    const skinnedSlotMap = new Map<number, number>() // entity → slot
    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.skinnedMask[i]) continue
      const instId = scene.skinInstanceIds[i]!
      if (instId < 0) continue
      const inst = scene.skinInstances[instId]
      if (!inst) continue

      skinnedSlotMap.set(i, skinnedSlot)
      device.queue.writeBuffer(
        this.jointBuffer,
        skinnedSlot * JOINT_SLOT_SIZE,
        inst.jointMatrices.buffer as ArrayBuffer,
        inst.jointMatrices.byteOffset,
        inst.jointMatrices.byteLength,
      )
      skinnedSlot++
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

    let draws = 0

    // ── Static draw pass ──────────────────────────────────────────────
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.cameraBG)
    pass.setBindGroup(2, this.lightingBG)

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (scene.skinnedMask[i]) continue // skip skinned in static pass
      const geo = this.geometries.get(scene.geometryIds[i]!)
      if (!geo) continue

      // Frustum cull (same test as above)
      const si = i * 3
      const sx = Math.abs(scene.scales[si]!)
      const sy = Math.abs(scene.scales[si + 1]!)
      const sz = Math.abs(scene.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : (sy > sz ? sy : sz)
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r)) continue

      pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
      pass.setVertexBuffer(0, geo.vertexBuffer)
      pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
      pass.drawIndexed(geo.indexCount)
      draws++
    }

    // ── Skinned draw pass ─────────────────────────────────────────────
    if (skinnedSlotMap.size > 0) {
      pass.setPipeline(this.skinnedPipeline)
      pass.setBindGroup(0, this.cameraBG)
      pass.setBindGroup(2, this.lightingBG)

      for (let i = 0; i < scene.entityCount; i++) {
        if (!scene.renderMask[i] || !scene.skinnedMask[i]) continue
        const geo = this.skinnedGeometries.get(scene.geometryIds[i]!)
        if (!geo) continue
        const slot = skinnedSlotMap.get(i)
        if (slot === undefined) continue

        // Frustum cull
        const si = i * 3
        const sx = Math.abs(scene.scales[si]!)
        const sy = Math.abs(scene.scales[si + 1]!)
        const sz = Math.abs(scene.scales[si + 2]!)
        const maxScale = sx > sy ? (sx > sz ? sx : sz) : (sy > sz ? sy : sz)
        const r = geo.boundingRadius * maxScale
        if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r)) continue

        pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
        pass.setBindGroup(3, this.jointBG, [slot * JOINT_SLOT_SIZE])
        pass.setVertexBuffer(0, geo.vertexBuffer)
        pass.setVertexBuffer(1, geo.skinBuffer)
        pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        pass.drawIndexed(geo.indexCount)
        draws++
      }
    }

    this.drawCalls = draws

    pass.end()
    device.queue.submit([encoder.finish()])
  }
}
