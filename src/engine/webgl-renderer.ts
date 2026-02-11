import { m4Multiply, m4PerspectiveGL, m4OrthoGL, m4ExtractFrustumPlanes, frustumContainsSphere } from './math.ts'
import {
  glLambertVS,
  glLambertFS,
  glUnlitVS,
  glUnlitFS,
  glSkinnedLambertVS,
  glTexturedLambertVS,
  glTexturedLambertFS,
  glShadowDepthVS,
  glSkinnedShadowDepthVS,
  glShadowDepthFS,
} from './webgl-shaders.ts'

import type { IRenderer, RenderScene } from './renderer.ts'

const MODEL_SLOT_SIZE = 256 // match WebGPU alignment
const MAX_JOINTS = 128
const JOINT_SLOT_SIZE = MAX_JOINTS * 64 // 8192 bytes per slot
const MAX_SKINNED_ENTITIES = 64
const SHADOW_MAP_SIZE = 2048

// UBO binding points
const UBO_CAMERA = 0
const UBO_MODEL = 1
const UBO_LIGHTING = 2
const UBO_JOINTS = 3
const UBO_SHADOW_CAMERA = 4

interface GeometryGL {
  vao: WebGLVertexArrayObject
  vbo: WebGLBuffer
  ibo: WebGLBuffer
  indexCount: number
  indexType: number // gl.UNSIGNED_SHORT or gl.UNSIGNED_INT
  boundingRadius: number
}

interface SkinnedGeometryGL extends GeometryGL {
  skinVbo: WebGLBuffer
}

interface TexturedGeometryGL extends GeometryGL {
  uvVbo: WebGLBuffer
}

interface TextureGL {
  texture: WebGLTexture
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    throw new Error(`Shader compile error: ${info}`)
  }
  return shader
}

function linkProgram(gl: WebGL2RenderingContext, vs: WebGLShader, fs: WebGLShader): WebGLProgram {
  const program = gl.createProgram()!
  gl.attachShader(program, vs)
  gl.attachShader(program, fs)
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program)
    gl.deleteProgram(program)
    throw new Error(`Program link error: ${info}`)
  }
  return program
}

function createProgram(gl: WebGL2RenderingContext, vsSrc: string, fsSrc: string): WebGLProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc)
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc)
  return linkProgram(gl, vs, fs)
}

export class WebGLRenderer implements IRenderer {
  readonly backendType = 'webgl' as const

  private gl: WebGL2RenderingContext
  private staticProgram: WebGLProgram
  private skinnedProgram: WebGLProgram
  private unlitProgram: WebGLProgram
  private texturedProgram: WebGLProgram
  private shadowDepthProgram: WebGLProgram
  private shadowSkinnedDepthProgram: WebGLProgram

  // UBOs
  private cameraUBO: WebGLBuffer
  private modelUBO: WebGLBuffer
  private lightingUBO: WebGLBuffer
  private jointUBO: WebGLBuffer
  private shadowCameraUBO: WebGLBuffer

  // Shadow
  private shadowFBO: WebGLFramebuffer
  private shadowTexture: WebGLTexture
  private shadowMapLoc: WebGLUniformLocation | null
  private shadowMapSkinnedLoc: WebGLUniformLocation | null

  // Textured
  private texturedShadowMapLoc: WebGLUniformLocation | null = null
  private texturedAoMapLoc: WebGLUniformLocation | null = null

  private geometries = new Map<number, GeometryGL>()
  private skinnedGeometries = new Map<number, SkinnedGeometryGL>()
  private texturedGeometries = new Map<number, TexturedGeometryGL>()
  private glTextures = new Map<number, TextureGL>()
  private maxEntities: number

  drawCalls = 0

  // Scratch buffers
  private vpMat = new Float32Array(16)
  private frustumPlanes = new Float32Array(24)
  private modelSlot = new Float32Array(MODEL_SLOT_SIZE / 4)
  private lightData = new Float32Array(32) // 128 bytes
  private _tpOrder: number[] = []
  private _tpDist: Float32Array

  constructor(gl: WebGL2RenderingContext, _canvas: HTMLCanvasElement, maxEntities = 1000) {
    this.gl = gl
    this.maxEntities = maxEntities
    this._tpDist = new Float32Array(maxEntities)

    // ── Compile programs ─────────────────────────────────────────────
    this.staticProgram = createProgram(gl, glLambertVS, glLambertFS)
    this.skinnedProgram = createProgram(gl, glSkinnedLambertVS, glLambertFS)
    this.unlitProgram = createProgram(gl, glUnlitVS, glUnlitFS)
    this.texturedProgram = createProgram(gl, glTexturedLambertVS, glTexturedLambertFS)
    this.shadowDepthProgram = createProgram(gl, glShadowDepthVS, glShadowDepthFS)
    this.shadowSkinnedDepthProgram = createProgram(gl, glSkinnedShadowDepthVS, glShadowDepthFS)

    // ── Bind UBO block indices ──────────────────────────────────────
    for (const prog of [this.staticProgram, this.skinnedProgram, this.texturedProgram]) {
      const cameraIdx = gl.getUniformBlockIndex(prog, 'Camera')
      if (cameraIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, cameraIdx, UBO_CAMERA)

      const modelIdx = gl.getUniformBlockIndex(prog, 'Model')
      if (modelIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, modelIdx, UBO_MODEL)

      const lightIdx = gl.getUniformBlockIndex(prog, 'Lighting')
      if (lightIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, lightIdx, UBO_LIGHTING)
    }

    // Unlit program uses Camera + Model only (no Lighting)
    {
      const cameraIdx = gl.getUniformBlockIndex(this.unlitProgram, 'Camera')
      if (cameraIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(this.unlitProgram, cameraIdx, UBO_CAMERA)
      const modelIdx = gl.getUniformBlockIndex(this.unlitProgram, 'Model')
      if (modelIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(this.unlitProgram, modelIdx, UBO_MODEL)
    }

    // Joint UBO only in skinned program
    const jointIdx = gl.getUniformBlockIndex(this.skinnedProgram, 'JointMatrices')
    if (jointIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(this.skinnedProgram, jointIdx, UBO_JOINTS)

    // Shadow depth programs use Camera and Model UBOs
    for (const prog of [this.shadowDepthProgram, this.shadowSkinnedDepthProgram]) {
      const cameraIdx = gl.getUniformBlockIndex(prog, 'Camera')
      if (cameraIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, cameraIdx, UBO_SHADOW_CAMERA)

      const modelIdx = gl.getUniformBlockIndex(prog, 'Model')
      if (modelIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, modelIdx, UBO_MODEL)
    }

    // Joint UBO for skinned shadow depth
    const shadowJointIdx = gl.getUniformBlockIndex(this.shadowSkinnedDepthProgram, 'JointMatrices')
    if (shadowJointIdx !== gl.INVALID_INDEX)
      gl.uniformBlockBinding(this.shadowSkinnedDepthProgram, shadowJointIdx, UBO_JOINTS)

    // Shadow map sampler uniform locations
    this.shadowMapLoc = gl.getUniformLocation(this.staticProgram, 'uShadowMap')
    this.shadowMapSkinnedLoc = gl.getUniformLocation(this.skinnedProgram, 'uShadowMap')
    this.texturedShadowMapLoc = gl.getUniformLocation(this.texturedProgram, 'uShadowMap')
    this.texturedAoMapLoc = gl.getUniformLocation(this.texturedProgram, 'uAoMap')

    // ── Create UBOs ─────────────────────────────────────────────────
    this.cameraUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.cameraUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 128, gl.DYNAMIC_DRAW)

    this.modelUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.modelUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, MODEL_SLOT_SIZE * maxEntities, gl.DYNAMIC_DRAW)

    this.lightingUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.lightingUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 128, gl.DYNAMIC_DRAW)

    this.jointUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.jointUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, JOINT_SLOT_SIZE * MAX_SKINNED_ENTITIES, gl.DYNAMIC_DRAW)

    this.shadowCameraUBO = gl.createBuffer()!
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.shadowCameraUBO)
    gl.bufferData(gl.UNIFORM_BUFFER, 128, gl.DYNAMIC_DRAW)

    gl.bindBuffer(gl.UNIFORM_BUFFER, null)

    // ── Shadow map texture + FBO ─────────────────────────────────────
    this.shadowTexture = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTexture)
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT32F, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LESS)
    gl.bindTexture(gl.TEXTURE_2D, null)

    this.shadowFBO = gl.createFramebuffer()!
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.shadowTexture, 0)
    gl.drawBuffers([gl.NONE])
    gl.readBuffer(gl.NONE)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)

    // ── GL state ────────────────────────────────────────────────────
    gl.enable(gl.DEPTH_TEST)
    gl.depthFunc(gl.LESS)
    gl.enable(gl.CULL_FACE)
    gl.cullFace(gl.BACK)
    gl.clearColor(0.15, 0.15, 0.2, 1)
  }

  perspective(out: Float32Array, o: number, fovY: number, aspect: number, near: number, far: number): void {
    m4PerspectiveGL(out, o, fovY, aspect, near, far)
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
    m4OrthoGL(out, o, left, right, bottom, top, near, far)
  }

  resize(w: number, h: number): void {
    this.gl.viewport(0, 0, w, h)
  }

  registerGeometry(id: number, vertices: Float32Array, indices: Uint16Array | Uint32Array): void {
    const gl = this.gl

    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)

    // VBO
    const vbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    // position (location 0): 3 floats at offset 0, stride 36
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0)
    // normal (location 1): 3 floats at offset 12
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12)
    // color (location 2): 3 floats at offset 24
    gl.enableVertexAttribArray(2)
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24)

    // IBO
    const ibo = gl.createBuffer()!
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

    gl.bindVertexArray(null)

    // Bounding sphere
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 9) {
      const x = vertices[i]!,
        y = vertices[i + 1]!,
        z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    const indexType = indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT
    this.geometries.set(id, {
      vao,
      vbo,
      ibo,
      indexCount: indices.length,
      indexType,
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
    const gl = this.gl

    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)

    // VBO 0: position/normal/color (stride 36)
    const vbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0)
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12)
    gl.enableVertexAttribArray(2)
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24)

    // VBO 1: interleaved joints (uint8x4) + weights (float32x4) = 20 bytes per vertex
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

    const skinVbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, skinVbo)
    gl.bufferData(gl.ARRAY_BUFFER, skinData, gl.STATIC_DRAW)

    // joints (location 3): uint8x4, use vertexAttribIPointer for integer attribs
    gl.enableVertexAttribArray(3)
    gl.vertexAttribIPointer(3, 4, gl.UNSIGNED_BYTE, 20, 0)
    // weights (location 4): float32x4
    gl.enableVertexAttribArray(4)
    gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 20, 4)

    // IBO
    const ibo = gl.createBuffer()!
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

    gl.bindVertexArray(null)

    // Bounding sphere
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 9) {
      const x = vertices[i]!,
        y = vertices[i + 1]!,
        z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    const indexType = indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT
    this.skinnedGeometries.set(id, {
      vao,
      vbo,
      ibo,
      skinVbo,
      indexCount: indices.length,
      indexType,
      boundingRadius: Math.sqrt(maxR2),
    })
  }

  registerTexturedGeometry(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    uvs: Float32Array,
  ): void {
    const gl = this.gl

    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)

    // VBO 0: position/normal/color (stride 36)
    const vbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0)
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12)
    gl.enableVertexAttribArray(2)
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24)

    // VBO 1: UVs (float32x2 = 8 bytes per vertex)
    const uvVbo = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, uvVbo)
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW)

    gl.enableVertexAttribArray(3)
    gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 8, 0)

    // IBO
    const ibo = gl.createBuffer()!
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo)
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW)

    gl.bindVertexArray(null)

    // Bounding sphere
    let maxR2 = 0
    for (let i = 0; i < vertices.length; i += 9) {
      const x = vertices[i]!,
        y = vertices[i + 1]!,
        z = vertices[i + 2]!
      const r2 = x * x + y * y + z * z
      if (r2 > maxR2) maxR2 = r2
    }

    const indexType = indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT
    const geoData: TexturedGeometryGL = {
      vao,
      vbo,
      ibo,
      uvVbo,
      indexCount: indices.length,
      indexType,
      boundingRadius: Math.sqrt(maxR2),
    }
    this.texturedGeometries.set(id, geoData)

    // Also create a shadow-only VAO in the geometries map (position-only, no UVs needed)
    const shadowVao = gl.createVertexArray()!
    gl.bindVertexArray(shadowVao)
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo) // reuse same VBO
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 36, 0)
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 36, 12)
    gl.enableVertexAttribArray(2)
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 36, 24)
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo) // reuse same IBO
    gl.bindVertexArray(null)

    this.geometries.set(id, {
      vao: shadowVao,
      vbo,
      ibo,
      indexCount: indices.length,
      indexType,
      boundingRadius: Math.sqrt(maxR2),
    })
  }

  registerTexture(id: number, data: Uint8Array, width: number, height: number): void {
    const gl = this.gl
    const texture = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.bindTexture(gl.TEXTURE_2D, null)
    this.glTextures.set(id, { texture })
  }

  render(scene: RenderScene): void {
    const gl = this.gl

    // ── Upload camera UBO (view + proj = 128 bytes) ──────────────────
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.cameraUBO)
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, scene.cameraView, 0, 16)
    gl.bufferSubData(gl.UNIFORM_BUFFER, 64, scene.cameraProj, 0, 16)

    // ── Upload lighting UBO (128 bytes) ──────────────────────────────
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
    const hasShadow = !!scene.shadowLightViewProj
    if (hasShadow) {
      for (let i = 0; i < 16; i++) lightData[12 + i] = scene.shadowLightViewProj![i]!
    } else {
      for (let i = 0; i < 16; i++) lightData[12 + i] = 0
    }
    const mapSize = scene.shadowMapSize ?? SHADOW_MAP_SIZE
    lightData[28] = scene.shadowBias ?? 0.001
    lightData[29] = scene.shadowNormalBias ?? 0.05 // normal offset bias
    lightData[30] = 1.0 / mapSize
    lightData[31] = hasShadow ? 1.0 : 0.0
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.lightingUBO)
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, lightData)

    // ── Frustum culling setup ────────────────────────────────────────
    m4Multiply(this.vpMat, 0, scene.cameraProj, 0, scene.cameraView, 0)
    m4ExtractFrustumPlanes(this.frustumPlanes, this.vpMat, 0)
    const planes = this.frustumPlanes

    // ── Upload per-entity model data (all renderable — shadow pass needs entities outside camera frustum)
    const modelSlot = this.modelSlot
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.modelUBO)
    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue

      // worldMatrix (16 floats)
      for (let j = 0; j < 16; j++) modelSlot[j] = scene.worldMatrices[i * 16 + j]!
      // color (3 floats → vec4 at offset 16)
      modelSlot[16] = scene.colors[i * 3]!
      modelSlot[17] = scene.colors[i * 3 + 1]!
      modelSlot[18] = scene.colors[i * 3 + 2]!
      modelSlot[19] = scene.alphas[i]!

      gl.bufferSubData(gl.UNIFORM_BUFFER, i * MODEL_SLOT_SIZE, modelSlot, 0, 20)
    }

    // ── Upload joint matrices ────────────────────────────────────────
    let skinnedSlot = 0
    const skinnedSlotMap = new Map<number, number>()
    gl.bindBuffer(gl.UNIFORM_BUFFER, this.jointUBO)
    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.skinnedMask[i]) continue
      const instId = scene.skinInstanceIds[i]!
      if (instId < 0) continue
      const inst = scene.skinInstances[instId]
      if (!inst) continue

      skinnedSlotMap.set(i, skinnedSlot)
      gl.bufferSubData(gl.UNIFORM_BUFFER, skinnedSlot * JOINT_SLOT_SIZE, inst.jointMatrices)
      skinnedSlot++
    }

    gl.bindBuffer(gl.UNIFORM_BUFFER, null)

    // ── Shadow depth pass ─────────────────────────────────────────────
    if (hasShadow) {
      // Upload shadow camera UBO (light VP as view, identity as projection)
      const shadowCamData = new Float32Array(32)
      for (let i = 0; i < 16; i++) shadowCamData[i] = scene.shadowLightViewProj![i]!
      shadowCamData[16] = 1
      shadowCamData[21] = 1
      shadowCamData[26] = 1
      shadowCamData[31] = 1
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.shadowCameraUBO)
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, shadowCamData)
      gl.bindBuffer(gl.UNIFORM_BUFFER, null)

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowFBO)
      gl.viewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
      gl.clear(gl.DEPTH_BUFFER_BIT)
      gl.enable(gl.POLYGON_OFFSET_FILL)
      gl.polygonOffset(1.0, 1.0)

      gl.bindBufferBase(gl.UNIFORM_BUFFER, UBO_SHADOW_CAMERA, this.shadowCameraUBO)

      // Static shadow casters
      gl.useProgram(this.shadowDepthProgram)
      for (let i = 0; i < scene.entityCount; i++) {
        if (!scene.renderMask[i]) continue
        if (scene.skinnedMask[i]) continue
        if (scene.unlitMask[i]) continue
        if (scene.alphas[i]! < 1.0) continue
        const geo = this.geometries.get(scene.geometryIds[i]!)
        if (!geo) continue

        gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
        gl.bindVertexArray(geo.vao)
        gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      }

      // Skinned shadow casters
      if (skinnedSlotMap.size > 0) {
        gl.useProgram(this.shadowSkinnedDepthProgram)
        for (let i = 0; i < scene.entityCount; i++) {
          if (!scene.renderMask[i] || !scene.skinnedMask[i]) continue
          if (scene.alphas[i]! < 1.0) continue
          const geo = this.skinnedGeometries.get(scene.geometryIds[i]!)
          if (!geo) continue
          const slot = skinnedSlotMap.get(i)
          if (slot === undefined) continue

          gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
          gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_JOINTS, this.jointUBO, slot * JOINT_SLOT_SIZE, JOINT_SLOT_SIZE)
          gl.bindVertexArray(geo.vao)
          gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
        }
      }

      gl.disable(gl.POLYGON_OFFSET_FILL)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight)
    }

    // ── Clear ────────────────────────────────────────────────────────
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    let draws = 0

    // ── Bind shared UBOs ─────────────────────────────────────────────
    gl.bindBufferBase(gl.UNIFORM_BUFFER, UBO_CAMERA, this.cameraUBO)
    gl.bindBufferBase(gl.UNIFORM_BUFFER, UBO_LIGHTING, this.lightingUBO)

    // ── Bind shadow map texture ──────────────────────────────────────
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTexture)

    // ── Unlit draw pass ──────────────────────────────────────────────
    gl.useProgram(this.unlitProgram)
    gl.disable(gl.CULL_FACE)

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

      gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
      gl.bindVertexArray(geo.vao)
      gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      draws++
    }

    gl.enable(gl.CULL_FACE)

    // ── Opaque static draw pass ─────────────────────────────────────
    gl.useProgram(this.staticProgram)
    if (this.shadowMapLoc !== null) gl.uniform1i(this.shadowMapLoc, 0)

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (scene.skinnedMask[i]) continue
      if (scene.unlitMask[i]) continue
      if (scene.texturedMask[i]) continue
      if (scene.alphas[i]! < 1.0) continue
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

      gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
      gl.bindVertexArray(geo.vao)
      gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      draws++
    }

    // ── Opaque textured draw pass ─────────────────────────────────────
    gl.useProgram(this.texturedProgram)
    if (this.texturedShadowMapLoc !== null) gl.uniform1i(this.texturedShadowMapLoc, 0)
    if (this.texturedAoMapLoc !== null) gl.uniform1i(this.texturedAoMapLoc, 1)

    for (let i = 0; i < scene.entityCount; i++) {
      if (!scene.renderMask[i]) continue
      if (!scene.texturedMask[i]) continue
      if (scene.alphas[i]! < 1.0) continue
      const geo = this.texturedGeometries.get(scene.geometryIds[i]!)
      if (!geo) continue
      const texId = scene.aoMapIds[i]!
      const tex = this.glTextures.get(texId)
      if (!tex) continue

      const si = i * 3
      const sx = Math.abs(scene.scales[si]!)
      const sy = Math.abs(scene.scales[si + 1]!)
      const sz = Math.abs(scene.scales[si + 2]!)
      const maxScale = sx > sy ? (sx > sz ? sx : sz) : sy > sz ? sy : sz
      const r = geo.boundingRadius * maxScale
      if (!frustumContainsSphere(planes, scene.positions[si]!, scene.positions[si + 1]!, scene.positions[si + 2]!, r))
        continue

      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, tex.texture)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, this.shadowTexture)

      gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
      gl.bindVertexArray(geo.vao)
      gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      draws++
    }

    // ── Opaque skinned draw pass ──────────────────────────────────────
    if (skinnedSlotMap.size > 0) {
      gl.useProgram(this.skinnedProgram)
      if (this.shadowMapSkinnedLoc !== null) gl.uniform1i(this.shadowMapSkinnedLoc, 0)

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

        gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
        gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_JOINTS, this.jointUBO, slot * JOINT_SLOT_SIZE, JOINT_SLOT_SIZE)
        gl.bindVertexArray(geo.vao)
        gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
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

    gl.enable(gl.BLEND)
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
    gl.depthMask(false)
    gl.disable(gl.CULL_FACE)

    let curPipeType = -1 // -1=unset, 0=static, 1=skinned, 2=textured
    for (const i of tpOrder) {
      const isSkinned = !!scene.skinnedMask[i]
      const isTextured = !!scene.texturedMask[i]
      if (isSkinned) {
        if (curPipeType !== 1) {
          gl.useProgram(this.skinnedProgram)
          if (this.shadowMapSkinnedLoc !== null) gl.uniform1i(this.shadowMapSkinnedLoc, 0)
          curPipeType = 1
        }
        const geo = this.skinnedGeometries.get(scene.geometryIds[i]!)!
        const slot = skinnedSlotMap.get(i)
        if (slot === undefined) continue
        gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
        gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_JOINTS, this.jointUBO, slot * JOINT_SLOT_SIZE, JOINT_SLOT_SIZE)
        gl.bindVertexArray(geo.vao)
        gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      } else if (isTextured) {
        if (curPipeType !== 2) {
          gl.useProgram(this.texturedProgram)
          if (this.texturedShadowMapLoc !== null) gl.uniform1i(this.texturedShadowMapLoc, 0)
          if (this.texturedAoMapLoc !== null) gl.uniform1i(this.texturedAoMapLoc, 1)
          curPipeType = 2
        }
        const geo = this.texturedGeometries.get(scene.geometryIds[i]!)
        if (!geo) continue
        const texId = scene.aoMapIds[i]!
        const tex = this.glTextures.get(texId)
        if (!tex) continue
        gl.activeTexture(gl.TEXTURE1)
        gl.bindTexture(gl.TEXTURE_2D, tex.texture)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, this.shadowTexture)
        gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
        gl.bindVertexArray(geo.vao)
        gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      } else {
        if (curPipeType !== 0) {
          gl.useProgram(this.staticProgram)
          if (this.shadowMapLoc !== null) gl.uniform1i(this.shadowMapLoc, 0)
          curPipeType = 0
        }
        const geo = this.geometries.get(scene.geometryIds[i]!)!
        gl.bindBufferRange(gl.UNIFORM_BUFFER, UBO_MODEL, this.modelUBO, i * MODEL_SLOT_SIZE, MODEL_SLOT_SIZE)
        gl.bindVertexArray(geo.vao)
        gl.drawElements(gl.TRIANGLES, geo.indexCount, geo.indexType, 0)
      }
      draws++
    }

    gl.disable(gl.BLEND)
    gl.depthMask(true)
    gl.enable(gl.CULL_FACE)

    this.drawCalls = draws
    gl.bindVertexArray(null)
  }

  destroy(): void {
    const gl = this.gl
    for (const geo of this.geometries.values()) {
      gl.deleteVertexArray(geo.vao)
      gl.deleteBuffer(geo.vbo)
      gl.deleteBuffer(geo.ibo)
    }
    for (const geo of this.skinnedGeometries.values()) {
      gl.deleteVertexArray(geo.vao)
      gl.deleteBuffer(geo.vbo)
      gl.deleteBuffer(geo.ibo)
      gl.deleteBuffer(geo.skinVbo)
    }
    for (const geo of this.texturedGeometries.values()) {
      gl.deleteVertexArray(geo.vao)
      gl.deleteBuffer(geo.uvVbo)
      // vbo + ibo already destroyed via geometries map
    }
    for (const tex of this.glTextures.values()) {
      gl.deleteTexture(tex.texture)
    }
    this.geometries.clear()
    this.skinnedGeometries.clear()
    this.texturedGeometries.clear()
    this.glTextures.clear()
    gl.deleteBuffer(this.cameraUBO)
    gl.deleteBuffer(this.modelUBO)
    gl.deleteBuffer(this.lightingUBO)
    gl.deleteBuffer(this.jointUBO)
    gl.deleteBuffer(this.shadowCameraUBO)
    gl.deleteFramebuffer(this.shadowFBO)
    gl.deleteTexture(this.shadowTexture)
    gl.deleteProgram(this.staticProgram)
    gl.deleteProgram(this.skinnedProgram)
    gl.deleteProgram(this.texturedProgram)
    gl.deleteProgram(this.unlitProgram)
    gl.deleteProgram(this.shadowDepthProgram)
    gl.deleteProgram(this.shadowSkinnedDepthProgram)
  }
}
