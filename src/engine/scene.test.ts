import { describe, test, expect } from 'bun:test'

import { m4Perspective, m4LookAt } from './math.ts'
import { Camera, Mesh, Scene } from './scene.ts'

import type { IRenderer, RenderScene } from './renderer.ts'

const EPSILON = 1e-5

// ── Mock renderer ───────────────────────────────────────────────────────────

function createMockRenderer(): IRenderer & {
  registered: Map<number, { vertices: Float32Array; indices: Uint16Array | Uint32Array }>
  skinnedRegistered: Map<number, { vertices: Float32Array; indices: Uint16Array | Uint32Array }>
  texturedRegistered: Map<number, { vertices: Float32Array; indices: Uint16Array | Uint32Array; uvs: Float32Array }>
  texturesRegistered: Map<number, { data: Uint8Array; width: number; height: number }>
  lastRenderScene: RenderScene | null
  renderCount: number
  resizeCalls: [number, number][]
} {
  const registered = new Map<number, { vertices: Float32Array; indices: Uint16Array | Uint32Array }>()
  const skinnedRegistered = new Map<number, { vertices: Float32Array; indices: Uint16Array | Uint32Array }>()
  const texturedRegistered = new Map<
    number,
    { vertices: Float32Array; indices: Uint16Array | Uint32Array; uvs: Float32Array }
  >()
  const texturesRegistered = new Map<number, { data: Uint8Array; width: number; height: number }>()
  let lastRenderScene: RenderScene | null = null
  let renderCount = 0
  const resizeCalls: [number, number][] = []

  return {
    backendType: 'webgpu' as const,
    drawCalls: 0,
    registered,
    skinnedRegistered,
    texturedRegistered,
    texturesRegistered,
    lastRenderScene,
    renderCount,
    resizeCalls,

    perspective(out, o, fovY, aspect, near, far) {
      m4Perspective(out, o, fovY, aspect, near, far)
    },
    ortho(out, o, left, right, bottom, top, near, far) {
      // identity-ish stub: just zero out
      for (let i = 0; i < 16; i++) out[o + i] = 0
      out[o] = 2 / (right - left)
      out[o + 5] = 2 / (top - bottom)
      out[o + 10] = -2 / (far - near)
      out[o + 15] = 1
    },
    registerGeometry(id, vertices, indices) {
      registered.set(id, { vertices, indices })
    },
    registerSkinnedGeometry(id, vertices, indices) {
      skinnedRegistered.set(id, { vertices, indices })
    },
    registerTexturedGeometry(id, vertices, indices, uvs) {
      texturedRegistered.set(id, { vertices, indices, uvs })
    },
    registerTexture(id, data, width, height) {
      texturesRegistered.set(id, { data, width, height })
    },
    render(scene) {
      lastRenderScene = scene
      renderCount++
      // Update the outer object references
      ;(this as any).lastRenderScene = scene
      ;(this as any).renderCount = renderCount
    },
    resize(w, h) {
      resizeCalls.push([w, h])
      ;(this as any).resizeCalls = resizeCalls
    },
    destroy() {},
  }
}

function createMockCanvas(w = 800, h = 600): HTMLCanvasElement {
  return { width: w, height: h } as unknown as HTMLCanvasElement
}

// ── Camera ──────────────────────────────────────────────────────────────────

describe('Camera', () => {
  test('has sensible defaults', () => {
    const cam = new Camera()
    expect(cam.fov).toBeCloseTo(Math.PI / 3, 5)
    expect(cam.near).toBe(0.1)
    expect(cam.far).toBe(1000)
    expect(cam.eye[0]).toBe(0)
    expect(cam.eye[1]).toBe(-10)
    expect(cam.eye[2]).toBe(5)
    expect(cam.target[0]).toBe(0)
    expect(cam.target[1]).toBe(0)
    expect(cam.target[2]).toBe(0)
    expect(cam.up[2]).toBe(1)
  })

  test('eye/target/up are writable Float32Arrays', () => {
    const cam = new Camera()
    cam.eye.set([1, 2, 3])
    cam.target.set([4, 5, 6])
    cam.up.set([0, 1, 0])
    expect(cam.eye[0]).toBe(1)
    expect(cam.target[1]).toBe(5)
    expect(cam.up[1]).toBe(1)
  })
})

// ── Mesh ────────────────────────────────────────────────────────────────────

describe('Mesh', () => {
  test('uses option values', () => {
    const m = new Mesh({
      geometry: 5,
      position: [1, 2, 3],
      rotation: [0.1, 0.2, 0.3],
      scale: [2, 2, 2],
      color: [1, 0, 0],
      alpha: 0.5,
      unlit: true,
      skinned: true,
      skinInstanceId: 7,
    })
    expect(m.geometry).toBe(5)
    expect(m.position[0]).toBe(1)
    expect(m.position[1]).toBe(2)
    expect(m.position[2]).toBe(3)
    expect(m.rotation[0]).toBeCloseTo(0.1, 5)
    expect(m.scale[0]).toBe(2)
    expect(m.color[0]).toBe(1)
    expect(m.color[1]).toBe(0)
    expect(m.alpha).toBe(0.5)
    expect(m.unlit).toBe(true)
    expect(m.skinned).toBe(true)
    expect(m.skinInstanceId).toBe(7)
    expect(m.visible).toBe(true)
  })

  test('defaults are sensible', () => {
    const m = new Mesh({ geometry: 0 })
    expect(m.position[0]).toBe(0)
    expect(m.scale[0]).toBe(1)
    expect(m.scale[1]).toBe(1)
    expect(m.scale[2]).toBe(1)
    expect(m.color[0]).toBe(1)
    expect(m.color[1]).toBe(1)
    expect(m.color[2]).toBe(1)
    expect(m.alpha).toBe(1)
    expect(m.unlit).toBe(false)
    expect(m.skinned).toBe(false)
    expect(m.skinInstanceId).toBe(-1)
    expect(m.aoMap).toBe(-1)
    expect(m.bloom).toBe(0)
  })

  test('aoMap can be set via options', () => {
    const m = new Mesh({ geometry: 0, aoMap: 3 })
    expect(m.aoMap).toBe(3)
  })

  test('bloom defaults to 0 and can be set via options', () => {
    const m0 = new Mesh({ geometry: 0 })
    expect(m0.bloom).toBe(0)
    const m1 = new Mesh({ geometry: 0, bloom: 1.5 })
    expect(m1.bloom).toBe(1.5)
  })
})

// ── Scene: geometry management ──────────────────────────────────────────────

describe('Scene geometry management', () => {
  test('registerGeometry returns sequential IDs and delegates to renderer', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    const v = new Float32Array(9)
    const i16 = new Uint16Array(3)

    const id0 = scene.registerGeometry(v, i16)
    const id1 = scene.registerGeometry(v, i16)

    expect(id0).toBe(0)
    expect(id1).toBe(1)
    expect(renderer.registered.size).toBe(2)
    expect(renderer.registered.has(0)).toBe(true)
    expect(renderer.registered.has(1)).toBe(true)
  })

  test('registerSkinnedGeometry returns sequential IDs (shared counter)', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    const v = new Float32Array(9)
    const idx = new Uint16Array(3)
    const j = new Uint8Array(4)
    const w = new Float32Array(4)

    const id0 = scene.registerGeometry(v, idx)
    const id1 = scene.registerSkinnedGeometry(v, idx, j, w)

    expect(id0).toBe(0)
    expect(id1).toBe(1)
    expect(renderer.registered.size).toBe(1)
    expect(renderer.skinnedRegistered.size).toBe(1)
  })

  test('registerTexturedGeometry returns sequential IDs (shared counter)', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    const v = new Float32Array(9)
    const idx = new Uint16Array(3)
    const uvs = new Float32Array(6)

    const id0 = scene.registerGeometry(v, idx)
    const id1 = scene.registerTexturedGeometry(v, idx, uvs)

    expect(id0).toBe(0)
    expect(id1).toBe(1)
    expect(renderer.registered.size).toBe(1)
    expect(renderer.texturedRegistered.size).toBe(1)
    expect(renderer.texturedRegistered.has(1)).toBe(true)
  })

  test('registerTexture returns sequential IDs', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    const data = new Uint8Array(16)

    const id0 = scene.registerTexture(data, 2, 2)
    const id1 = scene.registerTexture(data, 2, 2)

    expect(id0).toBe(0)
    expect(id1).toBe(1)
    expect(renderer.texturesRegistered.size).toBe(2)
  })
})

// ── Scene: mesh management ──────────────────────────────────────────────────

describe('Scene mesh management', () => {
  test('add returns the same mesh and it appears in meshes', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    const m = new Mesh({ geometry: 0 })

    const result = scene.add(m)
    expect(result).toBe(m)
    expect(scene.meshes.length).toBe(1)
    expect(scene.meshes[0]).toBe(m)
  })

  test('remove eliminates mesh from list', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    const m1 = new Mesh({ geometry: 0 })
    const m2 = new Mesh({ geometry: 0 })
    scene.add(m1)
    scene.add(m2)

    scene.remove(m1)
    expect(scene.meshes.length).toBe(1)
    expect(scene.meshes[0]).toBe(m2)
  })

  test('remove of non-existent mesh is a no-op', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    const m = new Mesh({ geometry: 0 })
    scene.remove(m) // should not throw
    expect(scene.meshes.length).toBe(0)
  })
})

// ── Scene: lighting ─────────────────────────────────────────────────────────

describe('Scene lighting', () => {
  test('setDirectionalLight sets direction and color', () => {
    const scene = new Scene(createMockRenderer(), createMockCanvas(), 100)
    scene.setDirectionalLight([-1, -1, -2], [0.5, 0.4, 0.3])
    expect(scene.lightDirection[0]).toBe(-1)
    expect(scene.lightDirection[1]).toBe(-1)
    expect(scene.lightDirection[2]).toBe(-2)
    expect(scene.lightDirColor[0]).toBeCloseTo(0.5, 5)
    expect(scene.lightDirColor[1]).toBeCloseTo(0.4, 5)
    expect(scene.lightDirColor[2]).toBeCloseTo(0.3, 5)
  })

  test('setAmbientLight sets ambient color', () => {
    const scene = new Scene(createMockRenderer(), createMockCanvas(), 100)
    scene.setAmbientLight([0.8, 0.9, 1])
    expect(scene.lightAmbientColor[0]).toBeCloseTo(0.8, 5)
    expect(scene.lightAmbientColor[1]).toBeCloseTo(0.9, 5)
    expect(scene.lightAmbientColor[2]).toBe(1)
  })
})

// ── Scene: shadow config ────────────────────────────────────────────────────

describe('Scene shadow config', () => {
  test('shadow defaults to disabled', () => {
    const scene = new Scene(createMockRenderer(), createMockCanvas(), 100)
    expect(scene.shadow.enabled).toBe(false)
    expect(scene.shadow.distance).toBe(400)
    expect(scene.shadow.extent).toBe(150)
    expect(scene.shadow.near).toBe(1)
    expect(scene.shadow.far).toBe(800)
    expect(scene.shadow.bias).toBe(0.0001)
  })

  test('shadow config is mutable', () => {
    const scene = new Scene(createMockRenderer(), createMockCanvas(), 100)
    scene.shadow.enabled = true
    scene.shadow.distance = 200
    scene.shadow.extent = 50
    scene.shadow.target.set([10, 20, 30])
    expect(scene.shadow.enabled).toBe(true)
    expect(scene.shadow.distance).toBe(200)
    expect(scene.shadow.extent).toBe(50)
    expect(scene.shadow.target[0]).toBe(10)
  })
})

// ── Scene: bloom config ──────────────────────────────────────────────────────

describe('Scene bloom config', () => {
  test('bloom defaults to disabled with intensity=1, threshold=0, radius=1, whiten=0', () => {
    const scene = new Scene(createMockRenderer(), createMockCanvas(), 100)
    expect(scene.bloom.enabled).toBe(false)
    expect(scene.bloom.intensity).toBe(1)
    expect(scene.bloom.threshold).toBe(0)
    expect(scene.bloom.radius).toBe(1)
    expect(scene.bloom.whiten).toBe(0)
  })

  test('bloom config is mutable', () => {
    const scene = new Scene(createMockRenderer(), createMockCanvas(), 100)
    scene.bloom.enabled = true
    scene.bloom.intensity = 2.5
    scene.bloom.threshold = 0.3
    scene.bloom.radius = 5
    scene.bloom.whiten = 0.7
    expect(scene.bloom.enabled).toBe(true)
    expect(scene.bloom.intensity).toBe(2.5)
    expect(scene.bloom.threshold).toBe(0.3)
    expect(scene.bloom.radius).toBe(5)
    expect(scene.bloom.whiten).toBe(0.7)
  })
})

// ── Scene: render ───────────────────────────────────────────────────────────

describe('Scene render', () => {
  test('render calls renderer.render with correct entity count', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0, position: [1, 2, 3], color: [1, 0, 0] }))
    scene.add(new Mesh({ geometry: 0, position: [4, 5, 6], color: [0, 1, 0] }))

    scene.render()

    expect(renderer.renderCount).toBe(1)
    const rs = renderer.lastRenderScene!
    expect(rs.entityCount).toBe(2)
  })

  test('render syncs mesh positions into SoA arrays', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0, position: [10, 20, 30] }))

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.positions[0]).toBe(10)
    expect(rs.positions[1]).toBe(20)
    expect(rs.positions[2]).toBe(30)
  })

  test('render syncs colors and alpha', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0, color: [0.5, 0.6, 0.7], alpha: 0.3 }))

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.colors[0]).toBeCloseTo(0.5, 5)
    expect(rs.colors[1]).toBeCloseTo(0.6, 5)
    expect(rs.colors[2]).toBeCloseTo(0.7, 5)
    expect(rs.alphas[0]).toBeCloseTo(0.3, 5)
  })

  test('render sets renderMask=0 for invisible meshes', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    const m = scene.add(new Mesh({ geometry: 0 }))
    m.visible = false

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.renderMask[0]).toBe(0)
  })

  test('render sets unlitMask and skinnedMask correctly', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0, unlit: true }))
    scene.add(new Mesh({ geometry: 0, skinned: true }))

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.unlitMask[0]).toBe(1)
    expect(rs.unlitMask[1]).toBe(0)
    expect(rs.skinnedMask[0]).toBe(0)
    expect(rs.skinnedMask[1]).toBe(1)
  })

  test('render computes view matrix from camera', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0 }))

    scene.camera.eye.set([0, -10, 5])
    scene.camera.target.set([0, 0, 0])
    scene.camera.up.set([0, 0, 1])
    scene.render()

    // Verify against manual m4LookAt
    const expected = new Float32Array(16)
    m4LookAt(expected, 0, scene.camera.eye, 0, scene.camera.target, 0, scene.camera.up, 0)

    const rs = renderer.lastRenderScene!
    for (let i = 0; i < 16; i++) {
      expect(Math.abs(rs.cameraView[i]! - expected[i]!)).toBeLessThan(EPSILON)
    }
  })

  test('render computes projection matrix from camera config + aspect', () => {
    const renderer = createMockRenderer()
    const canvas = createMockCanvas(800, 400) // aspect = 2.0
    const scene = new Scene(renderer, canvas, 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0 }))

    scene.camera.fov = Math.PI / 4
    scene.camera.near = 1
    scene.camera.far = 500
    scene.render()

    // Verify against manual perspective (mock renderer uses m4Perspective)
    const expected = new Float32Array(16)
    m4Perspective(expected, 0, Math.PI / 4, 2.0, 1, 500)

    const rs = renderer.lastRenderScene!
    for (let i = 0; i < 16; i++) {
      expect(Math.abs(rs.cameraProj[i]! - expected[i]!)).toBeLessThan(EPSILON)
    }
  })

  test('render without shadow does not set shadowLightViewProj', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0 }))
    scene.shadow.enabled = false

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.shadowLightViewProj).toBeUndefined()
  })

  test('render syncs texturedMask and aoMapIds', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0 }))
    scene.add(new Mesh({ geometry: 0, aoMap: 2 }))

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.texturedMask[0]).toBe(0)
    expect(rs.texturedMask[1]).toBe(1)
    expect(rs.aoMapIds[0]).toBe(-1)
    expect(rs.aoMapIds[1]).toBe(2)
  })

  test('render syncs bloom values and bloom config into RenderScene', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0 }))
    scene.add(new Mesh({ geometry: 0, bloom: 1.5 }))

    scene.bloom.enabled = true
    scene.bloom.intensity = 2.0
    scene.bloom.threshold = 0.1
    scene.bloom.radius = 5
    scene.bloom.whiten = 0.6
    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.bloomEnabled).toBe(true)
    expect(rs.bloomIntensity).toBe(2.0)
    expect(rs.bloomThreshold).toBeCloseTo(0.1, 5)
    expect(rs.bloomRadius).toBe(5)
    expect(rs.bloomWhiten).toBeCloseTo(0.6, 5)
    expect(rs.bloomValues![0]).toBe(0)
    expect(rs.bloomValues![1]).toBeCloseTo(1.5, 5)
  })

  test('render without bloom sets bloomEnabled false', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0 }))

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.bloomEnabled).toBe(false)
  })

  test('render with shadow enabled sets shadowLightViewProj and bias', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.registerGeometry(new Float32Array(9), new Uint16Array(3))
    scene.add(new Mesh({ geometry: 0 }))

    scene.setDirectionalLight([-1, -1, -2], [0.5, 0.5, 0.5])
    scene.shadow.enabled = true
    scene.shadow.bias = 0.002

    scene.render()

    const rs = renderer.lastRenderScene!
    expect(rs.shadowLightViewProj).toBeDefined()
    expect(rs.shadowLightViewProj!.length).toBe(16)
    expect(rs.shadowBias).toBe(0.002)
    // The VP matrix should be non-zero (a real computed matrix)
    let nonZero = false
    for (let i = 0; i < 16; i++) {
      if (rs.shadowLightViewProj![i] !== 0) nonZero = true
    }
    expect(nonZero).toBe(true)
  })
})

// ── Scene: getters ──────────────────────────────────────────────────────────

describe('Scene getters', () => {
  test('backendType delegates to renderer', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    expect(scene.backendType).toBe('webgpu')
  })

  test('drawCalls delegates to renderer', () => {
    const renderer = createMockRenderer()
    renderer.drawCalls = 42
    const scene = new Scene(renderer, createMockCanvas(), 100)
    expect(scene.drawCalls).toBe(42)
  })
})

// ── Scene: resize ───────────────────────────────────────────────────────────

describe('Scene resize', () => {
  test('resize delegates to renderer', () => {
    const renderer = createMockRenderer()
    const scene = new Scene(renderer, createMockCanvas(), 100)
    scene.resize(1920, 1080)
    expect(renderer.resizeCalls.length).toBe(1)
    expect(renderer.resizeCalls[0]![0]).toBe(1920)
    expect(renderer.resizeCalls[0]![1]).toBe(1080)
  })
})
