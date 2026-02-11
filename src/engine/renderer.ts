import { m4Multiply, m4Perspective, m4Ortho, m4ExtractFrustumPlanes, frustumContainsSphere } from './math.ts'
import {
  lambertShader,
  skinnedLambertShader,
  unlitShader,
  shadowDepthShader,
  skinnedShadowDepthShader,
} from './shaders.ts'
import { MODEL_SLOT_SIZE, JOINT_SLOT_SIZE, MAX_SKINNED_ENTITIES, SHADOW_MAP_SIZE, MSAA_SAMPLES } from './constants.ts'

import type { SkinInstance } from './skin.ts'

export type BackendType = 'webgpu' | 'webgl'

export interface IRenderer {
  readonly backendType: BackendType
  drawCalls: number
  perspective(out: Float32Array, o: number, fovY: number, aspect: number, near: number, far: number): void
  ortho(
    out: Float32Array,
    o: number,
    left: number,
    right: number,
    bottom: number,
    top: number,
    near: number,
    far: number,
  ): void
  registerGeometry(id: number, vertices: Float32Array, indices: Uint16Array | Uint32Array): void
  registerSkinnedGeometry(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    joints: Uint8Array,
    weights: Float32Array,
  ): void
  render(scene: RenderScene): void
  resize(w: number, h: number): void
  destroy(): void
}

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
  unlitMask: Uint8Array
  positions: Float32Array
  scales: Float32Array
  worldMatrices: Float32Array
  colors: Float32Array
  alphas: Float32Array
  geometryIds: Uint8Array
  skinInstanceIds: Int16Array
  skinInstances: SkinInstance[]
  shadowLightViewProj?: Float32Array
  shadowMapSize?: number
  shadowBias?: number
  shadowNormalBias?: number
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

export class Renderer implements IRenderer {
  readonly backendType = 'webgpu' as const

  private device: GPUDevice
  private context: GPUCanvasContext
  private pipeline: GPURenderPipeline
  private skinnedPipeline: GPURenderPipeline
  private unlitPipeline: GPURenderPipeline
  private transparentPipeline: GPURenderPipeline
  private transparentSkinnedPipeline: GPURenderPipeline
  private depthTexture: GPUTexture
  private depthView: GPUTextureView
  private msaaTexture: GPUTexture
  private msaaView: GPUTextureView
  private canvasFormat: GPUTextureFormat

  // Shadow
  private shadowPipeline: GPURenderPipeline
  private shadowSkinnedPipeline: GPURenderPipeline
  private shadowMapTexture: GPUTexture
  private shadowMapView: GPUTextureView
  private shadowCameraBuffer: GPUBuffer
  private shadowCameraBG: GPUBindGroup

  // Buffers
  private cameraBuffer: GPUBuffer
  private modelBuffer: GPUBuffer
  private lightingBuffer: GPUBuffer
  private jointBuffer: GPUBuffer

  // Bind groups
  private cameraBG: GPUBindGroup
  private lightingBG!: GPUBindGroup
  private lightingBGL: GPUBindGroupLayout
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
  private lightData = new Float32Array(32) // 128 bytes for lighting UBO
  private modelSlot = new Float32Array(MODEL_SLOT_SIZE / 4) // 64 floats
  private shadowCamData = new Float32Array(32)
  private skinnedSlotMap = new Map<number, number>() // entity → slot
  private _tpOrder: number[] = []
  private _tpDist: Float32Array

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    maxEntities = 1000,
  ) {
    this.device = device
    this.context = context
    this.canvasFormat = format
    this.maxEntities = maxEntities
    this._tpDist = new Float32Array(maxEntities)

    // ── Shader modules ─────────────────────────────────────────────
    const shaderModule = device.createShaderModule({ code: lambertShader })
    const skinnedShaderModule = device.createShaderModule({ code: skinnedLambertShader })
    const unlitShaderModule = device.createShaderModule({ code: unlitShader })
    const shadowShaderModule = device.createShaderModule({ code: shadowDepthShader })
    const skinnedShadowShaderModule = device.createShaderModule({ code: skinnedShadowDepthShader })

    // ── Bind group layouts ────────────────────────────────────────────
    const cameraBGL = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
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
    this.lightingBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
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
      bindGroupLayouts: [cameraBGL, this.modelBGL, this.lightingBGL],
    })
    const skinnedPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL, this.lightingBGL, jointBGL],
    })

    // Shadow pipeline layouts (minimal — no lighting group)
    const shadowPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL],
    })
    const shadowSkinnedPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL, jointBGL],
    })

    // Unlit pipeline layout (camera + model only, no lighting)
    const unlitPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL],
    })

    // ── Unlit pipeline ──────────────────────────────────────────────
    this.unlitPipeline = device.createRenderPipeline({
      layout: unlitPipelineLayout,
      vertex: {
        module: unlitShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 36,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
            ],
          },
        ],
      },
      fragment: {
        module: unlitShaderModule,
        entryPoint: 'fs',
        targets: [{ format }],
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      multisample: { count: MSAA_SAMPLES },
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
              { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
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
      multisample: { count: MSAA_SAMPLES },
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
              { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
              { shaderLocation: 2, offset: 24, format: 'float32x3' }, // color
            ],
          },
          {
            arrayStride: 20, // 4 bytes joints + 16 bytes weights
            attributes: [
              { shaderLocation: 3, offset: 0, format: 'uint8x4' }, // joints
              { shaderLocation: 4, offset: 4, format: 'float32x4' }, // weights
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
      multisample: { count: MSAA_SAMPLES },
    })

    // ── Transparent pipelines ─────────────────────────────────────────
    const alphaBlend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    }

    this.transparentPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 36,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs',
        targets: [{ format, blend: alphaBlend }],
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      multisample: { count: MSAA_SAMPLES },
    })

    this.transparentSkinnedPipeline = device.createRenderPipeline({
      layout: skinnedPipelineLayout,
      vertex: {
        module: skinnedShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 36,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
            ],
          },
          {
            arrayStride: 20,
            attributes: [
              { shaderLocation: 3, offset: 0, format: 'uint8x4' },
              { shaderLocation: 4, offset: 4, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: {
        module: skinnedShaderModule,
        entryPoint: 'fs',
        targets: [{ format, blend: alphaBlend }],
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      multisample: { count: MSAA_SAMPLES },
    })

    // ── Shadow pipelines ──────────────────────────────────────────────
    const shadowDepthStencil: GPUDepthStencilState = {
      format: 'depth32float',
      depthWriteEnabled: true,
      depthCompare: 'less',
      depthBias: 1,
      depthBiasSlopeScale: 1,
    }

    this.shadowPipeline = device.createRenderPipeline({
      layout: shadowPipelineLayout,
      vertex: {
        module: shadowShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 36,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position only
            ],
          },
        ],
      },
      depthStencil: shadowDepthStencil,
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    })

    this.shadowSkinnedPipeline = device.createRenderPipeline({
      layout: shadowSkinnedPipelineLayout,
      vertex: {
        module: skinnedShadowShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 36,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          {
            arrayStride: 20,
            attributes: [
              { shaderLocation: 3, offset: 0, format: 'uint8x4' },
              { shaderLocation: 4, offset: 4, format: 'float32x4' },
            ],
          },
        ],
      },
      depthStencil: shadowDepthStencil,
      primitive: { topology: 'triangle-list', cullMode: 'back' },
    })

    // ── MSAA + Depth textures ──────────────────────────────────────────
    this.msaaTexture = this.createMsaaTexture(canvas.width, canvas.height)
    this.msaaView = this.msaaTexture.createView()
    this.depthTexture = this.createDepthTexture(canvas.width, canvas.height)
    this.depthView = this.depthTexture.createView()

    // ── Shadow map texture ────────────────────────────────────────────
    this.shadowMapTexture = device.createTexture({
      size: [SHADOW_MAP_SIZE, SHADOW_MAP_SIZE],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.shadowMapView = this.shadowMapTexture.createView()

    const shadowSampler = device.createSampler({
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear',
    })

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
    // Lighting: 128 bytes (direction + dirColor + ambient + lightVP + shadowParams)
    this.lightingBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Joint matrices storage buffer
    this.jointBuffer = device.createBuffer({
      size: JOINT_SLOT_SIZE * MAX_SKINNED_ENTITIES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    // Shadow camera: 1 mat4 VP = 128 bytes (same layout as camera: view + proj, but we pack VP into view slot)
    this.shadowCameraBuffer = device.createBuffer({
      size: 128,
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
      layout: this.lightingBGL,
      entries: [
        { binding: 0, resource: { buffer: this.lightingBuffer } },
        { binding: 1, resource: this.shadowMapView },
        { binding: 2, resource: shadowSampler },
      ],
    })
    this.jointBG = device.createBindGroup({
      layout: jointBGL,
      entries: [{ binding: 0, resource: { buffer: this.jointBuffer, size: JOINT_SLOT_SIZE } }],
    })
    this.shadowCameraBG = device.createBindGroup({
      layout: cameraBGL,
      entries: [{ binding: 0, resource: { buffer: this.shadowCameraBuffer } }],
    })
  }

  private createMsaaTexture(w: number, h: number): GPUTexture {
    return this.device.createTexture({
      size: [w, h],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: MSAA_SAMPLES,
    })
  }

  private createDepthTexture(w: number, h: number): GPUTexture {
    return this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: MSAA_SAMPLES,
    })
  }

  perspective(out: Float32Array, o: number, fovY: number, aspect: number, near: number, far: number): void {
    m4Perspective(out, o, fovY, aspect, near, far)
  }

  ortho(
    out: Float32Array,
    o: number,
    left: number,
    right: number,
    bottom: number,
    top: number,
    near: number,
    far: number,
  ): void {
    m4Ortho(out, o, left, right, bottom, top, near, far)
  }

  resize(w: number, h: number): void {
    this.msaaTexture.destroy()
    this.msaaTexture = this.createMsaaTexture(w, h)
    this.msaaView = this.msaaTexture.createView()
    this.depthTexture.destroy()
    this.depthTexture = this.createDepthTexture(w, h)
    this.depthView = this.depthTexture.createView()
  }

  registerGeometry(id: number, vertices: Float32Array, indices: Uint16Array | Uint32Array): void {
    const vertexBuffer = this.device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(
      vertexBuffer,
      0,
      vertices.buffer as ArrayBuffer,
      vertices.byteOffset,
      vertices.byteLength,
    )

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
      const x = vertices[i]!,
        y = vertices[i + 1]!,
        z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    const indexFormat: GPUIndexFormat = indices instanceof Uint32Array ? 'uint32' : 'uint16'
    this.geometries.set(id, {
      vertexBuffer,
      indexBuffer,
      indexCount: indices.length,
      indexFormat,
      boundingRadius: Math.sqrt(maxR2),
    })
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
      const x = vertices[i]!,
        y = vertices[i + 1]!,
        z = vertices[i + 2]!
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
    device.queue.writeBuffer(
      this.cameraBuffer,
      0,
      scene.cameraView.buffer as ArrayBuffer,
      scene.cameraView.byteOffset + scene.cameraViewOffset * 4,
      64,
    )
    device.queue.writeBuffer(
      this.cameraBuffer,
      64,
      scene.cameraProj.buffer as ArrayBuffer,
      scene.cameraProj.byteOffset + scene.cameraProjOffset * 4,
      64,
    )

    // Upload lighting (128 bytes)
    const lightData = this.lightData
    lightData[0] = scene.lightDirection[0]!
    lightData[1] = scene.lightDirection[1]!
    lightData[2] = scene.lightDirection[2]!
    lightData[3] = 0
    lightData[4] = scene.lightDirColor[0]!
    lightData[5] = scene.lightDirColor[1]!
    lightData[6] = scene.lightDirColor[2]!
    lightData[7] = 0
    lightData[8] = scene.lightAmbientColor[0]!
    lightData[9] = scene.lightAmbientColor[1]!
    lightData[10] = scene.lightAmbientColor[2]!
    lightData[11] = 0
    // lightVP (mat4x4f at offset 12 floats = 48 bytes)
    const hasShadow = !!scene.shadowLightViewProj
    if (hasShadow) {
      for (let i = 0; i < 16; i++) lightData[12 + i] = scene.shadowLightViewProj![i]!
    } else {
      for (let i = 0; i < 16; i++) lightData[12 + i] = 0
    }
    // shadowParams (vec4f at offset 28 floats = 112 bytes)
    const mapSize = scene.shadowMapSize ?? SHADOW_MAP_SIZE
    lightData[28] = scene.shadowBias ?? 0.001 // bias
    lightData[29] = scene.shadowNormalBias ?? 0.05 // normal offset bias
    lightData[30] = 1.0 / mapSize // texelSize
    lightData[31] = hasShadow ? 1.0 : 0.0 // enabled
    device.queue.writeBuffer(this.lightingBuffer, 0, lightData.buffer as ArrayBuffer, lightData.byteOffset, 128)

    // ── Frustum culling setup ───────────────────────────────────────────
    // VP = projection * view
    m4Multiply(this.vpMat, 0, scene.cameraProj, scene.cameraProjOffset, scene.cameraView, scene.cameraViewOffset)
    m4ExtractFrustumPlanes(this.frustumPlanes, this.vpMat, 0)
    const planes = this.frustumPlanes

    // Upload per-entity model data (all renderable — shadow pass needs entities outside camera frustum)
    const modelSlot = this.modelSlot
    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue

      // worldMatrix (16 floats = 64 bytes)
      for (let j = 0; j < 16; j++) modelSlot[j] = scene.worldMatrices[i * 16 + j]!
      // color (3 floats → vec4f at offset 16)
      modelSlot[16] = scene.colors[i * 3]!
      modelSlot[17] = scene.colors[i * 3 + 1]!
      modelSlot[18] = scene.colors[i * 3 + 2]!
      modelSlot[19] = scene.alphas[i]!
      device.queue.writeBuffer(this.modelBuffer, i * MODEL_SLOT_SIZE, modelSlot, 0, 20)
    }

    // Upload joint matrices for skinned entities
    let skinnedSlot = 0
    const skinnedSlotMap = this.skinnedSlotMap
    skinnedSlotMap.clear()
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

    const encoder = device.createCommandEncoder()

    // ── Shadow depth pass ──────────────────────────────────────────────
    if (hasShadow) {
      // Upload shadow camera (light VP into view slot, identity into proj slot)
      // Shadow depth shaders use camera.projection * camera.view * worldPos
      // We store identity in projection and lightVP in view, so result = I * lightVP * worldPos = lightVP * worldPos
      const shadowCamData = this.shadowCamData
      for (let i = 0; i < 16; i++) shadowCamData[i] = scene.shadowLightViewProj![i]!
      // Identity matrix for projection (zero the rest, set diagonal)
      for (let i = 16; i < 32; i++) shadowCamData[i] = 0
      shadowCamData[16] = 1
      shadowCamData[21] = 1
      shadowCamData[26] = 1
      shadowCamData[31] = 1
      device.queue.writeBuffer(this.shadowCameraBuffer, 0, shadowCamData.buffer as ArrayBuffer, 0, 128)

      const shadowPass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowMapView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })

      // Static shadow casters
      shadowPass.setPipeline(this.shadowPipeline)
      shadowPass.setBindGroup(0, this.shadowCameraBG)

      for (let i = 0; i < scene.entityCount; i++) {
        if (!scene.renderMask[i]) continue
        if (scene.skinnedMask[i]) continue
        if (scene.unlitMask[i]) continue
        if (scene.alphas[i]! < 1.0) continue
        const geo = this.geometries.get(scene.geometryIds[i]!)
        if (!geo) continue

        shadowPass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
        shadowPass.setVertexBuffer(0, geo.vertexBuffer)
        shadowPass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        shadowPass.drawIndexed(geo.indexCount)
      }

      // Skinned shadow casters
      if (skinnedSlotMap.size > 0) {
        shadowPass.setPipeline(this.shadowSkinnedPipeline)
        shadowPass.setBindGroup(0, this.shadowCameraBG)

        for (let i = 0; i < scene.entityCount; i++) {
          if (!scene.renderMask[i] || !scene.skinnedMask[i]) continue
          if (scene.alphas[i]! < 1.0) continue
          const geo = this.skinnedGeometries.get(scene.geometryIds[i]!)
          if (!geo) continue
          const slot = skinnedSlotMap.get(i)
          if (slot === undefined) continue

          shadowPass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
          shadowPass.setBindGroup(2, this.jointBG, [slot * JOINT_SLOT_SIZE])
          shadowPass.setVertexBuffer(0, geo.vertexBuffer)
          shadowPass.setVertexBuffer(1, geo.skinBuffer)
          shadowPass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
          shadowPass.drawIndexed(geo.indexCount)
        }
      }

      shadowPass.end()
    }

    // ── Main render pass ───────────────────────────────────────────────
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.msaaView,
          resolveTarget: this.context.getCurrentTexture().createView(),
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

    // ── Unlit draw pass ───────────────────────────────────────────────
    pass.setPipeline(this.unlitPipeline)
    pass.setBindGroup(0, this.cameraBG)

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (!scene.unlitMask[i]) continue
      const geo = this.geometries.get(scene.geometryIds[i]!)
      if (!geo) continue

      const si = i * 3
      const sx = Math.abs(scene.scales[si]!)
      const sy = Math.abs(scene.scales[si + 1]!)
      const sz = Math.abs(scene.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : sy > sz ? sy : sz
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r))
        continue

      pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
      pass.setVertexBuffer(0, geo.vertexBuffer)
      pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
      pass.drawIndexed(geo.indexCount)
      draws++
    }

    // ── Opaque static draw pass ───────────────────────────────────────
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.cameraBG)
    pass.setBindGroup(2, this.lightingBG)

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (scene.skinnedMask[i]) continue
      if (scene.unlitMask[i]) continue
      if (scene.alphas[i]! < 1.0) continue // defer transparent
      const geo = this.geometries.get(scene.geometryIds[i]!)
      if (!geo) continue

      const si = i * 3
      const sx = Math.abs(scene.scales[si]!)
      const sy = Math.abs(scene.scales[si + 1]!)
      const sz = Math.abs(scene.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : sy > sz ? sy : sz
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r))
        continue

      pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
      pass.setVertexBuffer(0, geo.vertexBuffer)
      pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
      pass.drawIndexed(geo.indexCount)
      draws++
    }

    // ── Opaque skinned draw pass ──────────────────────────────────────
    if (skinnedSlotMap.size > 0) {
      pass.setPipeline(this.skinnedPipeline)
      pass.setBindGroup(0, this.cameraBG)
      pass.setBindGroup(2, this.lightingBG)

      for (let i = 0; i < scene.entityCount; i++) {
        if (!scene.renderMask[i] || !scene.skinnedMask[i]) continue
        if (scene.alphas[i]! < 1.0) continue
        const geo = this.skinnedGeometries.get(scene.geometryIds[i]!)
        if (!geo) continue
        const slot = skinnedSlotMap.get(i)
        if (slot === undefined) continue

        const si = i * 3
        const sx = Math.abs(scene.scales[si]!)
        const sy = Math.abs(scene.scales[si + 1]!)
        const sz = Math.abs(scene.scales[si + 2]!)
        const maxScale = sx > sy ? (sx > sz ? sx : sz) : sy > sz ? sy : sz
        const r = geo.boundingRadius * maxScale
        if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r))
          continue

        pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
        pass.setBindGroup(3, this.jointBG, [slot * JOINT_SLOT_SIZE])
        pass.setVertexBuffer(0, geo.vertexBuffer)
        pass.setVertexBuffer(1, geo.skinBuffer)
        pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        pass.drawIndexed(geo.indexCount)
        draws++
      }
    }

    // ── Transparent pass (sorted back-to-front) ────────────────────────
    // Extract camera eye from view matrix (column-major)
    const vo = scene.cameraViewOffset
    const vm = scene.cameraView
    const tx = vm[vo + 12]!,
      ty = vm[vo + 13]!,
      tz = vm[vo + 14]!
    const camX = -(vm[vo]! * tx + vm[vo + 1]! * ty + vm[vo + 2]! * tz)
    const camY = -(vm[vo + 4]! * tx + vm[vo + 5]! * ty + vm[vo + 6]! * tz)
    const camZ = -(vm[vo + 8]! * tx + vm[vo + 9]! * ty + vm[vo + 10]! * tz)

    const tpOrder = this._tpOrder
    const tpDist = this._tpDist
    tpOrder.length = 0

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (scene.alphas[i]! >= 1.0) continue
      const isSkinned = !!scene.skinnedMask[i]
      const geo = isSkinned
        ? this.skinnedGeometries.get(scene.geometryIds[i]!)
        : this.geometries.get(scene.geometryIds[i]!)
      if (!geo) continue

      const si = i * 3
      const sx = Math.abs(scene.scales[si]!)
      const sy = Math.abs(scene.scales[si + 1]!)
      const sz = Math.abs(scene.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : sy > sz ? sy : sz
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r))
        continue

      const dx = scene.positions[si]! - camX
      const dy = scene.positions[si + 1]! - camY
      const dz = scene.positions[si + 2]! - camZ
      tpDist[i] = dx * dx + dy * dy + dz * dz
      tpOrder.push(i)
    }

    tpOrder.sort((a, b) => tpDist[b]! - tpDist[a]!)

    let curSkinned = -1 // -1=unset, 0=static, 1=skinned
    for (const i of tpOrder) {
      const isSkinned = !!scene.skinnedMask[i]
      if (isSkinned) {
        if (curSkinned !== 1) {
          pass.setPipeline(this.transparentSkinnedPipeline)
          pass.setBindGroup(0, this.cameraBG)
          pass.setBindGroup(2, this.lightingBG)
          curSkinned = 1
        }
        const geo = this.skinnedGeometries.get(scene.geometryIds[i]!)!
        const slot = skinnedSlotMap.get(i)
        if (slot === undefined) continue
        pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
        pass.setBindGroup(3, this.jointBG, [slot * JOINT_SLOT_SIZE])
        pass.setVertexBuffer(0, geo.vertexBuffer)
        pass.setVertexBuffer(1, geo.skinBuffer)
        pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        pass.drawIndexed(geo.indexCount)
      } else {
        if (curSkinned !== 0) {
          pass.setPipeline(this.transparentPipeline)
          pass.setBindGroup(0, this.cameraBG)
          pass.setBindGroup(2, this.lightingBG)
          curSkinned = 0
        }
        const geo = this.geometries.get(scene.geometryIds[i]!)!
        pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
        pass.setVertexBuffer(0, geo.vertexBuffer)
        pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        pass.drawIndexed(geo.indexCount)
      }
      draws++
    }

    this.drawCalls = draws

    pass.end()
    device.queue.submit([encoder.finish()])
  }

  destroy(): void {
    for (const geo of this.geometries.values()) {
      geo.vertexBuffer.destroy()
      geo.indexBuffer.destroy()
    }
    for (const geo of this.skinnedGeometries.values()) {
      geo.vertexBuffer.destroy()
      geo.indexBuffer.destroy()
      geo.skinBuffer.destroy()
    }
    this.geometries.clear()
    this.skinnedGeometries.clear()
    this.cameraBuffer.destroy()
    this.modelBuffer.destroy()
    this.lightingBuffer.destroy()
    this.jointBuffer.destroy()
    this.shadowCameraBuffer.destroy()
    this.shadowMapTexture.destroy()
    this.msaaTexture.destroy()
    this.depthTexture.destroy()
    this.device.destroy()
  }
}
