import { m4Multiply, m4Perspective, m4Ortho, m4ExtractFrustumPlanes, frustumContainsSphere } from './math.ts'
import {
  lambertShader,
  skinnedLambertShader,
  texturedLambertShader,
  unlitShader,
  shadowDepthShader,
  skinnedShadowDepthShader,
  bloomDownsampleShader,
  bloomUpsampleShader,
  bloomCompositeShader,
} from './shaders.ts'

import type { SkinInstance } from './skin.ts'

const MODEL_SLOT_SIZE = 256 // minUniformBufferOffsetAlignment
const MAX_JOINTS = 128
const JOINT_SLOT_SIZE = MAX_JOINTS * 64 // 128 mat4 * 64 bytes = 8192 (already 256-aligned)
const DEFAULT_MAX_SKINNED_ENTITIES = 1024
const SHADOW_CASCADE_SIZE = 1024
const SHADOW_ATLAS_SIZE = SHADOW_CASCADE_SIZE * 2 // 2×2 grid of cascades
const MSAA_SAMPLES = 4

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
  registerTexturedGeometry(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    uvs: Float32Array,
  ): void
  registerTexture(id: number, data: Uint8Array, width: number, height: number): void
  render(scene: RenderScene): void
  resize(w: number, h: number): void
  destroy(): void
}

export interface RenderScene {
  cameraView: Float32Array
  cameraProj: Float32Array
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
  texturedMask: Uint8Array
  aoMapIds: Int16Array
  shadowCascadeVPs?: Float32Array
  shadowCascadeSplits?: Float32Array
  shadowCascadeCount?: number
  shadowBias?: number
  shadowNormalBias?: number
  bloomEnabled?: boolean
  bloomIntensity?: number
  bloomThreshold?: number
  bloomRadius?: number
  bloomWhiten?: number
  bloomValues?: Float32Array
  outlineEnabled?: boolean
  outlineThickness?: number
  outlineColor?: [number, number, number]
  outlineDistanceFactor?: number
  outlineMask?: Uint8Array
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

interface TexturedGeometryGPU extends GeometryGPU {
  uvBuffer: GPUBuffer
}

interface TextureGPU {
  texture: GPUTexture
  bindGroup: GPUBindGroup
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

  // MRT pipelines (bloom — 2 color targets, fsMRT entry point)
  private mrtPipeline: GPURenderPipeline
  private mrtSkinnedPipeline: GPURenderPipeline
  private mrtUnlitPipeline: GPURenderPipeline
  private mrtTexturedPipeline: GPURenderPipeline
  private mrtTransparentPipeline: GPURenderPipeline
  private mrtTransparentSkinnedPipeline: GPURenderPipeline
  private mrtTransparentTexturedPipeline: GPURenderPipeline

  // Bloom post-processing (downsample-upsample mip chain)
  private static readonly BLOOM_MIPS = 5
  private bloomDownsamplePipeline: GPURenderPipeline
  private bloomUpsamplePipeline: GPURenderPipeline
  private bloomCompositePipeline: GPURenderPipeline
  private bloomSampleBGL: GPUBindGroupLayout // shared by downsample + upsample
  private bloomCompositeBGL: GPUBindGroupLayout
  private bloomDownsampleParams: GPUBuffer[] = [] // one per downsample pass
  private bloomUpsampleParams: GPUBuffer[] = [] // one per upsample pass
  private bloomCompositeParamsBuffer: GPUBuffer
  private bloomLinearSampler: GPUSampler
  private bloomSceneTexture: GPUTexture | null = null
  private bloomSceneView: GPUTextureView | null = null
  private bloomMsaaTexture: GPUTexture | null = null
  private bloomMsaaView: GPUTextureView | null = null
  private bloomTexture: GPUTexture | null = null
  private bloomTextureView: GPUTextureView | null = null
  private outlineMsaaTexture: GPUTexture | null = null
  private outlineMsaaView: GPUTextureView | null = null
  private outlineTexture: GPUTexture | null = null
  private outlineTextureView: GPUTextureView | null = null
  private bloomMips: GPUTexture[] = []
  private bloomMipViews: GPUTextureView[] = []
  private bloomDownsampleBGs: GPUBindGroup[] = []
  private bloomUpsampleBGs: GPUBindGroup[] = []
  private bloomCompositeBGInst: GPUBindGroup | null = null
  private bloomTexW = 0
  private bloomTexH = 0

  private geometries = new Map<number, GeometryGPU>()
  private skinnedGeometries = new Map<number, SkinnedGeometryGPU>()
  private texturedGeometries = new Map<number, TexturedGeometryGPU>()
  private textures = new Map<number, TextureGPU>()
  private texturedPipeline!: GPURenderPipeline
  private transparentTexturedPipeline!: GPURenderPipeline
  private textureBGL!: GPUBindGroupLayout
  private maxEntities: number
  private maxSkinnedEntities: number

  drawCalls = 0

  // Scratch buffers (no per-frame allocation)
  private vpMat = new Float32Array(16)
  private frustumPlanes = new Float32Array(24) // 6 planes × 4 floats
  private lightData = new Float32Array(84) // 336 bytes for lighting UBO (with 4 cascade VPs)
  private modelSlot = new Float32Array(MODEL_SLOT_SIZE / 4) // 64 floats
  private shadowCamData = new Float32Array(32)
  private compositeParams = new Float32Array(12) // 48 bytes
  private upsampleData = new Float32Array(4)
  private _mipW = new Float64Array(Renderer.BLOOM_MIPS)
  private _mipH = new Float64Array(Renderer.BLOOM_MIPS)
  private _skinnedSlotMap = new Map<number, number>()
  private _tpOrder: number[] = []
  private _tpDist: Float32Array

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    canvas: HTMLCanvasElement,
    maxEntities = 1000,
    maxSkinnedEntities = DEFAULT_MAX_SKINNED_ENTITIES,
  ) {
    this.device = device
    this.context = context
    this.canvasFormat = format
    this.maxEntities = maxEntities
    this.maxSkinnedEntities = maxSkinnedEntities
    this._tpDist = new Float32Array(maxEntities)

    // ── Shader modules ─────────────────────────────────────────────
    const shaderModule = device.createShaderModule({ code: lambertShader })
    const skinnedShaderModule = device.createShaderModule({ code: skinnedLambertShader })
    const unlitShaderModule = device.createShaderModule({ code: unlitShader })
    const shadowShaderModule = device.createShaderModule({ code: shadowDepthShader })
    const skinnedShadowShaderModule = device.createShaderModule({ code: skinnedShadowDepthShader })

    // ── Bind group layouts ────────────────────────────────────────────
    const cameraBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
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
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
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
            arrayStride: 40, // 10 floats * 4 bytes
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
              { shaderLocation: 2, offset: 24, format: 'float32x3' }, // color
              { shaderLocation: 3, offset: 36, format: 'float32' }, // bloom
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
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
              { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
              { shaderLocation: 2, offset: 24, format: 'float32x3' }, // color
              { shaderLocation: 3, offset: 36, format: 'float32' }, // bloom
            ],
          },
          {
            arrayStride: 20, // 4 bytes joints + 16 bytes weights
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'uint8x4' }, // joints
              { shaderLocation: 5, offset: 4, format: 'float32x4' }, // weights
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

    // ── Textured pipeline ────────────────────────────────────────────
    const texturedShaderModule = device.createShaderModule({ code: texturedLambertShader })
    this.textureBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    })
    const texturedPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [cameraBGL, this.modelBGL, this.lightingBGL, this.textureBGL],
    })

    const texturedVertexBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 40,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x3' },
          { shaderLocation: 3, offset: 36, format: 'float32' },
        ],
      },
      {
        arrayStride: 8,
        attributes: [{ shaderLocation: 4, offset: 0, format: 'float32x2' }],
      },
    ]

    this.texturedPipeline = device.createRenderPipeline({
      layout: texturedPipelineLayout,
      vertex: {
        module: texturedShaderModule,
        entryPoint: 'vs',
        buffers: texturedVertexBuffers,
      },
      fragment: {
        module: texturedShaderModule,
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
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
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
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
            ],
          },
          {
            arrayStride: 20,
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'uint8x4' },
              { shaderLocation: 5, offset: 4, format: 'float32x4' },
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

    this.transparentTexturedPipeline = device.createRenderPipeline({
      layout: texturedPipelineLayout,
      vertex: {
        module: texturedShaderModule,
        entryPoint: 'vs',
        buffers: texturedVertexBuffers,
      },
      fragment: {
        module: texturedShaderModule,
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

    // ── MRT pipelines (bloom — 2 color targets, fsMRT entry point) ────
    const mrtTargets: GPUColorTargetState[] = [{ format }, { format }, { format }]
    const mrtAlphaTargets: GPUColorTargetState[] = [
      { format, blend: alphaBlend },
      { format, blend: alphaBlend },
      { format, blend: alphaBlend },
    ]

    this.mrtUnlitPipeline = device.createRenderPipeline({
      layout: unlitPipelineLayout,
      vertex: {
        module: unlitShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
            ],
          },
        ],
      },
      fragment: { module: unlitShaderModule, entryPoint: 'fsMRT', targets: mrtTargets },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      multisample: { count: MSAA_SAMPLES },
    })

    this.mrtPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
            ],
          },
        ],
      },
      fragment: { module: shaderModule, entryPoint: 'fsMRT', targets: mrtTargets },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      multisample: { count: MSAA_SAMPLES },
    })

    this.mrtSkinnedPipeline = device.createRenderPipeline({
      layout: skinnedPipelineLayout,
      vertex: {
        module: skinnedShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
            ],
          },
          {
            arrayStride: 20,
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'uint8x4' },
              { shaderLocation: 5, offset: 4, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: { module: skinnedShaderModule, entryPoint: 'fsMRT', targets: mrtTargets },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      multisample: { count: MSAA_SAMPLES },
    })

    this.mrtTexturedPipeline = device.createRenderPipeline({
      layout: texturedPipelineLayout,
      vertex: { module: texturedShaderModule, entryPoint: 'vs', buffers: texturedVertexBuffers },
      fragment: { module: texturedShaderModule, entryPoint: 'fsMRT', targets: mrtTargets },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      multisample: { count: MSAA_SAMPLES },
    })

    this.mrtTransparentPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
            ],
          },
        ],
      },
      fragment: { module: shaderModule, entryPoint: 'fsMRT', targets: mrtAlphaTargets },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      multisample: { count: MSAA_SAMPLES },
    })

    this.mrtTransparentSkinnedPipeline = device.createRenderPipeline({
      layout: skinnedPipelineLayout,
      vertex: {
        module: skinnedShaderModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 40,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
              { shaderLocation: 3, offset: 36, format: 'float32' },
            ],
          },
          {
            arrayStride: 20,
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'uint8x4' },
              { shaderLocation: 5, offset: 4, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: { module: skinnedShaderModule, entryPoint: 'fsMRT', targets: mrtAlphaTargets },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      multisample: { count: MSAA_SAMPLES },
    })

    this.mrtTransparentTexturedPipeline = device.createRenderPipeline({
      layout: texturedPipelineLayout,
      vertex: { module: texturedShaderModule, entryPoint: 'vs', buffers: texturedVertexBuffers },
      fragment: { module: texturedShaderModule, entryPoint: 'fsMRT', targets: mrtAlphaTargets },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
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
            arrayStride: 40,
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
            arrayStride: 40,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          {
            arrayStride: 20,
            attributes: [
              { shaderLocation: 4, offset: 0, format: 'uint8x4' },
              { shaderLocation: 5, offset: 4, format: 'float32x4' },
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
      size: [SHADOW_ATLAS_SIZE, SHADOW_ATLAS_SIZE],
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
    // Lighting: 336 bytes (direction + dirColor + ambient + 4×lightVP + shadowParams + cascadeSplits)
    this.lightingBuffer = device.createBuffer({
      size: 336,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Joint matrices storage buffer
    this.jointBuffer = device.createBuffer({
      size: JOINT_SLOT_SIZE * this.maxSkinnedEntities,
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

    // ── Bloom post-processing pipelines (downsample-upsample mip chain) ──
    const bloomDownsampleModule = device.createShaderModule({ code: bloomDownsampleShader })
    const bloomUpsampleModule = device.createShaderModule({ code: bloomUpsampleShader })
    const bloomCompositeModule = device.createShaderModule({ code: bloomCompositeShader })

    this.bloomSampleBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })
    this.bloomCompositeBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    })

    const bloomSampleLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bloomSampleBGL] })
    this.bloomDownsamplePipeline = device.createRenderPipeline({
      layout: bloomSampleLayout,
      vertex: { module: bloomDownsampleModule, entryPoint: 'vs' },
      fragment: { module: bloomDownsampleModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })
    this.bloomUpsamplePipeline = device.createRenderPipeline({
      layout: bloomSampleLayout,
      vertex: { module: bloomUpsampleModule, entryPoint: 'vs' },
      fragment: {
        module: bloomUpsampleModule,
        entryPoint: 'fs',
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one' },
              alpha: { srcFactor: 'one', dstFactor: 'one' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    })
    this.bloomCompositePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bloomCompositeBGL] }),
      vertex: { module: bloomCompositeModule, entryPoint: 'vs' },
      fragment: { module: bloomCompositeModule, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })

    const MIPS = Renderer.BLOOM_MIPS
    for (let i = 0; i < MIPS; i++) {
      this.bloomDownsampleParams.push(
        device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      )
    }
    for (let i = 0; i < MIPS - 1; i++) {
      this.bloomUpsampleParams.push(
        device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      )
    }
    this.bloomCompositeParamsBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.bloomLinearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
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

  private ensureBloomTextures(w: number, h: number): void {
    if (this.bloomTexW === w && this.bloomTexH === h && this.bloomSceneTexture) return
    this.destroyBloomTextures()

    const device = this.device
    const MIPS = Renderer.BLOOM_MIPS

    // Scene resolve target (full res)
    this.bloomSceneTexture = device.createTexture({
      size: [w, h],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.bloomSceneView = this.bloomSceneTexture.createView()

    // Bloom MRT MSAA + resolve (full res)
    this.bloomMsaaTexture = device.createTexture({
      size: [w, h],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: MSAA_SAMPLES,
    })
    this.bloomMsaaView = this.bloomMsaaTexture.createView()

    this.bloomTexture = device.createTexture({
      size: [w, h],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.bloomTextureView = this.bloomTexture.createView()

    // Outline MRT MSAA + resolve (full res)
    this.outlineMsaaTexture = device.createTexture({
      size: [w, h],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: MSAA_SAMPLES,
    })
    this.outlineMsaaView = this.outlineMsaaTexture.createView()

    this.outlineTexture = device.createTexture({
      size: [w, h],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.outlineTextureView = this.outlineTexture.createView()

    // Mip chain: 5 levels at 1/2, 1/4, 1/8, 1/16, 1/32
    let mw = w,
      mh = h
    for (let i = 0; i < MIPS; i++) {
      mw = Math.max(1, (mw / 2) | 0)
      mh = Math.max(1, (mh / 2) | 0)
      const tex = device.createTexture({
        size: [mw, mh],
        format: this.canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.bloomMips.push(tex)
      this.bloomMipViews.push(tex.createView())
    }

    // Downsample bind groups: pass i reads from source (bloomTexture or mip[i-1])
    // and writes downsample params with source texel size
    let sw = w,
      sh = h
    for (let i = 0; i < MIPS; i++) {
      const srcView = i === 0 ? this.bloomTextureView : this.bloomMipViews[i - 1]!
      const srcTexelData = new Float32Array([1 / sw, 1 / sh, 0, 0])
      device.queue.writeBuffer(this.bloomDownsampleParams[i]!, 0, srcTexelData.buffer as ArrayBuffer, 0, 16)
      this.bloomDownsampleBGs.push(
        device.createBindGroup({
          layout: this.bloomSampleBGL,
          entries: [
            { binding: 0, resource: srcView },
            { binding: 1, resource: this.bloomLinearSampler },
            { binding: 2, resource: { buffer: this.bloomDownsampleParams[i]! } },
          ],
        }),
      )
      sw = Math.max(1, (sw / 2) | 0)
      sh = Math.max(1, (sh / 2) | 0)
    }

    // Upsample bind groups: pass i reads from mip[MIPS-1-i] (lower mip)
    // and renders additively into mip[MIPS-2-i] (higher mip)
    // Upsample params (destTexelSize + offset) written per-frame in render() for dynamic radius
    for (let i = 0; i < MIPS - 1; i++) {
      const srcIdx = MIPS - 1 - i // read from lowest first
      this.bloomUpsampleBGs.push(
        device.createBindGroup({
          layout: this.bloomSampleBGL,
          entries: [
            { binding: 0, resource: this.bloomMipViews[srcIdx]! },
            { binding: 1, resource: this.bloomLinearSampler },
            { binding: 2, resource: { buffer: this.bloomUpsampleParams[i]! } },
          ],
        }),
      )
    }

    // Composite: reads scene + mip[0] (accumulated bloom) + outline
    this.bloomCompositeBGInst = device.createBindGroup({
      layout: this.bloomCompositeBGL,
      entries: [
        { binding: 0, resource: this.bloomSceneView },
        { binding: 1, resource: this.bloomMipViews[0]! },
        { binding: 2, resource: this.outlineTextureView! },
        { binding: 3, resource: this.bloomLinearSampler },
        { binding: 4, resource: { buffer: this.bloomCompositeParamsBuffer } },
      ],
    })

    this.bloomTexW = w
    this.bloomTexH = h
  }

  private destroyBloomTextures(): void {
    this.bloomSceneTexture?.destroy()
    this.bloomMsaaTexture?.destroy()
    this.bloomTexture?.destroy()
    this.outlineMsaaTexture?.destroy()
    this.outlineTexture?.destroy()
    for (const tex of this.bloomMips) tex.destroy()
    this.bloomSceneTexture = null
    this.bloomSceneView = null
    this.bloomMsaaTexture = null
    this.bloomMsaaView = null
    this.bloomTexture = null
    this.bloomTextureView = null
    this.outlineMsaaTexture = null
    this.outlineMsaaView = null
    this.outlineTexture = null
    this.outlineTextureView = null
    this.bloomMips = []
    this.bloomMipViews = []
    this.bloomDownsampleBGs = []
    this.bloomUpsampleBGs = []
    this.bloomCompositeBGInst = null
    this.bloomTexW = 0
    this.bloomTexH = 0
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
    this.destroyBloomTextures()
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

    // Compute bounding sphere radius from vertex positions (stride = 10 floats)
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 10) {
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
    for (let i = 0; i < vertices.length; i += 10) {
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

  registerTexturedGeometry(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    uvs: Float32Array,
  ): void {
    const device = this.device

    // Vertex buffer 0 (position, normal, color — same as standard)
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

    // Vertex buffer 1: UVs (float32x2 = 8 bytes per vertex)
    const uvBuffer = device.createBuffer({
      size: uvs.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(uvBuffer, 0, uvs.buffer as ArrayBuffer, uvs.byteOffset, uvs.byteLength)

    // Bounding sphere
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 10) {
      const x = vertices[i]!,
        y = vertices[i + 1]!,
        z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    const indexFormat: GPUIndexFormat = indices instanceof Uint32Array ? 'uint32' : 'uint16'
    const geoData: TexturedGeometryGPU = {
      vertexBuffer,
      indexBuffer,
      indexCount: indices.length,
      indexFormat,
      boundingRadius: Math.sqrt(maxR2),
      uvBuffer,
    }
    this.texturedGeometries.set(id, geoData)
    // Also store in geometries map (without UV) so shadow pass picks it up
    this.geometries.set(id, {
      vertexBuffer,
      indexBuffer,
      indexCount: indices.length,
      indexFormat,
      boundingRadius: Math.sqrt(maxR2),
    })
  }

  registerTexture(id: number, data: Uint8Array, width: number, height: number): void {
    const device = this.device
    const texture = device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    device.queue.writeTexture(
      { texture },
      data.buffer as ArrayBuffer,
      { bytesPerRow: width * 4, rowsPerImage: height },
      [width, height],
    )
    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    })
    const bindGroup = device.createBindGroup({
      layout: this.textureBGL,
      entries: [
        { binding: 0, resource: texture.createView() },
        { binding: 1, resource: sampler },
      ],
    })
    this.textures.set(id, { texture, bindGroup })
  }

  render(scene: RenderScene): void {
    const device = this.device

    // Upload camera (view + proj = 128 bytes)
    device.queue.writeBuffer(
      this.cameraBuffer,
      0,
      scene.cameraView.buffer as ArrayBuffer,
      scene.cameraView.byteOffset,
      64,
    )
    device.queue.writeBuffer(
      this.cameraBuffer,
      64,
      scene.cameraProj.buffer as ArrayBuffer,
      scene.cameraProj.byteOffset,
      64,
    )

    // Upload lighting (336 bytes)
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
    // lightVP[4] (4 × mat4x4f at offset 12 floats = 48 bytes, 64 floats total)
    const cascadeCount = scene.shadowCascadeCount ?? 0
    const hasShadow = cascadeCount > 0
    if (hasShadow) {
      for (let i = 0; i < cascadeCount * 16; i++) lightData[12 + i] = scene.shadowCascadeVPs![i]!
      // Zero remaining cascade slots
      for (let i = cascadeCount * 16; i < 64; i++) lightData[12 + i] = 0
    } else {
      for (let i = 0; i < 64; i++) lightData[12 + i] = 0
    }
    // shadowParams (vec4f at offset 76 floats = 304 bytes)
    lightData[76] = scene.shadowBias ?? 0.001 // bias
    lightData[77] = scene.shadowNormalBias ?? 0.05 // normal offset bias
    lightData[78] = 1.0 / SHADOW_CASCADE_SIZE // texelSize (per cascade tile)
    lightData[79] = hasShadow ? 1.0 : 0.0 // enabled
    // cascadeSplits (vec4f at offset 80 floats = 320 bytes)
    if (hasShadow) {
      for (let i = 0; i < 4; i++) lightData[80 + i] = scene.shadowCascadeSplits?.[i] ?? 0
    } else {
      lightData[80] = 0
      lightData[81] = 0
      lightData[82] = 0
      lightData[83] = 0
    }
    device.queue.writeBuffer(this.lightingBuffer, 0, lightData.buffer as ArrayBuffer, lightData.byteOffset, 336)

    // ── Frustum culling setup ───────────────────────────────────────────
    // VP = projection * view
    m4Multiply(this.vpMat, 0, scene.cameraProj, 0, scene.cameraView, 0)
    m4ExtractFrustumPlanes(this.frustumPlanes, this.vpMat, 0)
    const planes = this.frustumPlanes

    const hasPostprocessing = !!scene.bloomEnabled || !!scene.outlineEnabled
    // Extract camera eye + outline distance factor only when post-processing is active
    let eyeX = 0,
      eyeY = 0,
      eyeZ = 0,
      outlineDF = 0
    if (hasPostprocessing) {
      const vm = scene.cameraView
      const vtx = vm[12]!,
        vty = vm[13]!,
        vtz = vm[14]!
      eyeX = -(vm[0]! * vtx + vm[1]! * vty + vm[2]! * vtz)
      eyeY = -(vm[4]! * vtx + vm[5]! * vty + vm[6]! * vtz)
      eyeZ = -(vm[8]! * vtx + vm[9]! * vty + vm[10]! * vtz)
      outlineDF = scene.outlineDistanceFactor ?? 0
    }

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
      // bloom + outline (slots 20-23) — skip computation when both features disabled
      if (hasPostprocessing) {
        const bloomVal = scene.bloomValues?.[i] ?? 0
        modelSlot[20] = bloomVal
        modelSlot[21] = scene.bloomWhiten ?? 0
        const outlineGroup = scene.outlineMask?.[i] ?? 0
        const isOutlined = outlineGroup > 0
        modelSlot[22] = isOutlined ? (((outlineGroup * 37 + 1) % 255) + 1) / 255.0 : 0.0
        if (isOutlined && outlineDF > 0) {
          const dx = scene.positions[i * 3]! - eyeX
          const dy = scene.positions[i * 3 + 1]! - eyeY
          const dz = scene.positions[i * 3 + 2]! - eyeZ
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
          modelSlot[23] = Math.min(outlineDF / Math.max(dist, 0.01), 1.0)
        } else {
          modelSlot[23] = isOutlined ? 1.0 : 0.0
        }
      } else {
        modelSlot[20] = 0
        modelSlot[21] = 0
        modelSlot[22] = 0
        modelSlot[23] = 0
      }
      device.queue.writeBuffer(this.modelBuffer, i * MODEL_SLOT_SIZE, modelSlot, 0, 24)
    }

    // Upload joint matrices for skinned entities
    let skinnedSlot = 0
    const skinnedSlotMap = this._skinnedSlotMap
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

    // ── Shadow depth pass (cascaded) ──────────────────────────────────
    if (hasShadow) {
      const shadowPass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowMapView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      })

      for (let c = 0; c < cascadeCount; c++) {
        // Set viewport to the cascade's quadrant in the 2×2 atlas
        const col = c % 2
        const row = (c / 2) | 0
        shadowPass.setViewport(
          col * SHADOW_CASCADE_SIZE,
          row * SHADOW_CASCADE_SIZE,
          SHADOW_CASCADE_SIZE,
          SHADOW_CASCADE_SIZE,
          0,
          1,
        )
        shadowPass.setScissorRect(
          col * SHADOW_CASCADE_SIZE,
          row * SHADOW_CASCADE_SIZE,
          SHADOW_CASCADE_SIZE,
          SHADOW_CASCADE_SIZE,
        )

        // Upload shadow camera: cascade VP in view slot, identity in proj slot
        const shadowCamData = this.shadowCamData
        for (let i = 0; i < 16; i++) shadowCamData[i] = scene.shadowCascadeVPs![c * 16 + i]!
        for (let i = 16; i < 32; i++) shadowCamData[i] = 0
        shadowCamData[16] = 1
        shadowCamData[21] = 1
        shadowCamData[26] = 1
        shadowCamData[31] = 1
        device.queue.writeBuffer(this.shadowCameraBuffer, 0, shadowCamData.buffer as ArrayBuffer, 0, 128)

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
      }

      shadowPass.end()
    }

    // ── Main render pass ───────────────────────────────────────────────
    const usePostprocessing = !!scene.bloomEnabled || !!scene.outlineEnabled
    const canvasTex = this.context.getCurrentTexture()

    if (usePostprocessing) {
      this.ensureBloomTextures(canvasTex.width, canvasTex.height)

      // MRT pass: 3 color attachments → sceneTexture + bloomTexture + outlineTexture
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.msaaView,
            resolveTarget: this.bloomSceneView!,
            clearValue: { r: 0.15, g: 0.15, b: 0.2, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
          {
            view: this.bloomMsaaView!,
            resolveTarget: this.bloomTextureView!,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
          {
            view: this.outlineMsaaView!,
            resolveTarget: this.outlineTextureView!,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
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
      this.drawCalls = this.drawScene(pass, scene, planes, skinnedSlotMap, true)
      pass.end()

      // Downsample chain: bloomTexture → mip[0] → mip[1] → ... → mip[N-1]
      const MIPS = Renderer.BLOOM_MIPS
      for (let i = 0; i < MIPS; i++) {
        const p = encoder.beginRenderPass({
          colorAttachments: [
            { view: this.bloomMipViews[i]!, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' },
          ],
        })
        p.setPipeline(this.bloomDownsamplePipeline)
        p.setBindGroup(0, this.bloomDownsampleBGs[i]!)
        p.draw(3)
        p.end()
      }

      // Write upsample params per-frame (destTexelSize + offset based on dynamic radius)
      const radius = scene.bloomRadius ?? 1
      let uw = canvasTex.width,
        uh = canvasTex.height
      // Compute mip sizes (reuse scratch arrays)
      const mipW = this._mipW
      const mipH = this._mipH
      for (let i = 0; i < MIPS; i++) {
        uw = Math.max(1, (uw / 2) | 0)
        uh = Math.max(1, (uh / 2) | 0)
        mipW[i] = uw
        mipH[i] = uh
      }
      for (let i = 0; i < MIPS - 1; i++) {
        const srcIdx = MIPS - 1 - i // source mip (lower res)
        const dstIdx = MIPS - 2 - i // dest mip (higher res)
        const destW = mipW[dstIdx]!,
          destH = mipH[dstIdx]!
        const srcW = mipW[srcIdx]!,
          srcH = mipH[srcIdx]!
        const data = this.upsampleData
        data[0] = 1 / destW
        data[1] = 1 / destH
        data[2] = radius / srcW
        data[3] = radius / srcH
        device.queue.writeBuffer(this.bloomUpsampleParams[i]!, 0, data.buffer as ArrayBuffer, data.byteOffset, 16)
      }

      // Upsample chain: mip[N-1] → mip[N-2] → ... → mip[0] (additive blend)
      for (let i = 0; i < MIPS - 1; i++) {
        const dstIdx = MIPS - 2 - i
        const p = encoder.beginRenderPass({
          colorAttachments: [{ view: this.bloomMipViews[dstIdx]!, loadOp: 'load', storeOp: 'store' }],
        })
        p.setPipeline(this.bloomUpsamplePipeline)
        p.setBindGroup(0, this.bloomUpsampleBGs[i]!)
        p.draw(3)
        p.end()
      }

      // Composite: sceneTexture + mip[0] (accumulated bloom) + outline → canvas
      const compositeParams = this.compositeParams
      compositeParams[0] = scene.bloomIntensity ?? 1
      compositeParams[1] = scene.bloomThreshold ?? 0
      compositeParams[2] = scene.outlineEnabled ? (scene.outlineThickness ?? 3) : 0
      compositeParams[3] = 0 // pad
      compositeParams[4] = scene.outlineColor?.[0] ?? 0
      compositeParams[5] = scene.outlineColor?.[1] ?? 0
      compositeParams[6] = scene.outlineColor?.[2] ?? 0
      compositeParams[7] = 1 // alpha pad
      compositeParams[8] = 1 / canvasTex.width
      compositeParams[9] = 1 / canvasTex.height
      compositeParams[10] = 0 // pad
      compositeParams[11] = 0 // pad
      device.queue.writeBuffer(
        this.bloomCompositeParamsBuffer,
        0,
        compositeParams.buffer as ArrayBuffer,
        compositeParams.byteOffset,
        48,
      )
      const compositePass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: canvasTex.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      })
      compositePass.setPipeline(this.bloomCompositePipeline)
      compositePass.setBindGroup(0, this.bloomCompositeBGInst!)
      compositePass.draw(3)
      compositePass.end()
    } else {
      // Original single-target path
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.msaaView,
            resolveTarget: canvasTex.createView(),
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
      this.drawCalls = this.drawScene(pass, scene, planes, skinnedSlotMap, false)
      pass.end()
    }

    device.queue.submit([encoder.finish()])
  }

  private drawScene(
    pass: GPURenderPassEncoder,
    scene: RenderScene,
    planes: Float32Array,
    skinnedSlotMap: Map<number, number>,
    useMRT: boolean,
  ): number {
    let draws = 0

    // ── Unlit draw pass ───────────────────────────────────────────────
    pass.setPipeline(useMRT ? this.mrtUnlitPipeline : this.unlitPipeline)
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
    pass.setPipeline(useMRT ? this.mrtPipeline : this.pipeline)
    pass.setBindGroup(0, this.cameraBG)
    pass.setBindGroup(2, this.lightingBG)

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (scene.skinnedMask[i]) continue
      if (scene.unlitMask[i]) continue
      if (scene.texturedMask[i]) continue
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

    // ── Opaque textured draw pass ─────────────────────────────────────
    pass.setPipeline(useMRT ? this.mrtTexturedPipeline : this.texturedPipeline)
    pass.setBindGroup(0, this.cameraBG)
    pass.setBindGroup(2, this.lightingBG)

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (!scene.texturedMask[i]) continue
      if (scene.alphas[i]! < 1.0) continue
      const geo = this.texturedGeometries.get(scene.geometryIds[i]!)
      if (!geo) continue
      const texId = scene.aoMapIds[i]!
      const tex = this.textures.get(texId)
      if (!tex) continue

      const si = i * 3
      const sx = Math.abs(scene.scales[si]!)
      const sy = Math.abs(scene.scales[si + 1]!)
      const sz = Math.abs(scene.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : sy > sz ? sy : sz
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r))
        continue

      pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
      pass.setBindGroup(3, tex.bindGroup)
      pass.setVertexBuffer(0, geo.vertexBuffer)
      pass.setVertexBuffer(1, geo.uvBuffer)
      pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
      pass.drawIndexed(geo.indexCount)
      draws++
    }

    // ── Opaque skinned draw pass ──────────────────────────────────────
    if (skinnedSlotMap.size > 0) {
      pass.setPipeline(useMRT ? this.mrtSkinnedPipeline : this.skinnedPipeline)
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
    const vm = scene.cameraView
    const tx = vm[12]!,
      ty = vm[13]!,
      tz = vm[14]!
    const camX = -(vm[0]! * tx + vm[1]! * ty + vm[2]! * tz)
    const camY = -(vm[4]! * tx + vm[5]! * ty + vm[6]! * tz)
    const camZ = -(vm[8]! * tx + vm[9]! * ty + vm[10]! * tz)

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

    let curPipeType = -1 // -1=unset, 0=static, 1=skinned, 2=textured
    for (const i of tpOrder) {
      const isSkinned = !!scene.skinnedMask[i]
      const isTextured = !!scene.texturedMask[i]
      if (isSkinned) {
        if (curPipeType !== 1) {
          pass.setPipeline(useMRT ? this.mrtTransparentSkinnedPipeline : this.transparentSkinnedPipeline)
          pass.setBindGroup(0, this.cameraBG)
          pass.setBindGroup(2, this.lightingBG)
          curPipeType = 1
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
      } else if (isTextured) {
        if (curPipeType !== 2) {
          pass.setPipeline(useMRT ? this.mrtTransparentTexturedPipeline : this.transparentTexturedPipeline)
          pass.setBindGroup(0, this.cameraBG)
          pass.setBindGroup(2, this.lightingBG)
          curPipeType = 2
        }
        const geo = this.texturedGeometries.get(scene.geometryIds[i]!)
        if (!geo) continue
        const texId = scene.aoMapIds[i]!
        const tex = this.textures.get(texId)
        if (!tex) continue
        pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
        pass.setBindGroup(3, tex.bindGroup)
        pass.setVertexBuffer(0, geo.vertexBuffer)
        pass.setVertexBuffer(1, geo.uvBuffer)
        pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        pass.drawIndexed(geo.indexCount)
      } else {
        if (curPipeType !== 0) {
          pass.setPipeline(useMRT ? this.mrtTransparentPipeline : this.transparentPipeline)
          pass.setBindGroup(0, this.cameraBG)
          pass.setBindGroup(2, this.lightingBG)
          curPipeType = 0
        }
        const geo = this.geometries.get(scene.geometryIds[i]!)!
        pass.setBindGroup(1, this.modelBG, [i * MODEL_SLOT_SIZE])
        pass.setVertexBuffer(0, geo.vertexBuffer)
        pass.setIndexBuffer(geo.indexBuffer, geo.indexFormat)
        pass.drawIndexed(geo.indexCount)
      }
      draws++
    }

    return draws
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
    for (const geo of this.texturedGeometries.values()) {
      geo.uvBuffer.destroy()
      // vertexBuffer + indexBuffer already destroyed via geometries map
    }
    for (const tex of this.textures.values()) {
      tex.texture.destroy()
    }
    this.geometries.clear()
    this.skinnedGeometries.clear()
    this.texturedGeometries.clear()
    this.textures.clear()
    this.cameraBuffer.destroy()
    this.modelBuffer.destroy()
    this.lightingBuffer.destroy()
    this.jointBuffer.destroy()
    this.shadowCameraBuffer.destroy()
    this.shadowMapTexture.destroy()
    this.msaaTexture.destroy()
    this.depthTexture.destroy()
    this.destroyBloomTextures()
    for (const buf of this.bloomDownsampleParams) buf.destroy()
    for (const buf of this.bloomUpsampleParams) buf.destroy()
    this.bloomCompositeParamsBuffer.destroy()
    this.device.destroy()
  }
}
