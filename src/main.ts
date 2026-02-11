import { World, TRANSFORM, MESH_INSTANCE, CAMERA, INPUT_RECEIVER, SKINNED } from './ecs/index.ts'
import { InputManager } from './ecs/index.ts'
import {
  createRenderer,
  OrbitControls,
  loadGlb,
  createSkeleton,
  createSkinInstance,
  updateSkinInstance,
  createSphereGeometry,
  mergeGeometries,
  m4FromTRS,
  m4LookAt,
} from './engine/index.ts'

import type { BackendType, IRenderer, RenderScene, SkinInstance } from './engine/index.ts'

const MOVE_SPEED = 3
const SPHERE_GEO_ID = 1
const MEGAXE_GEO_ID = 2
const EDEN_GEO_START = 3
const PLAYER_BODY_GEO_ID = 4
const UP = new Float32Array([0, 0, 1])

const EDEN_COLORS: Record<string, [number, number, number]> = {
  Eden_1: [0.78, 0.44, 0.25],
  Eden_2: [0.85, 0.68, 0.3],
  Eden_3: [0.75, 0.75, 0.75],
  Eden_4: [0.78, 0.3, 0.2],
  Eden_5: [0.25, 0.55, 0.2],
  Eden_6: [0.3, 0.36, 0.3],
  Eden_7: [0.75, 0.75, 0.75],
  Eden_8: [0.15, 0.15, 0.18],
  Eden_9: [0.0, 0.75, 0.7],
  Eden_10: [0.55, 0.5, 0.42],
  Eden_11: [0.85, 0.35, 0.55],
  Eden_12: [0.95, 0.95, 0.95],
  Eden_13: [0.8, 0.65, 0.15],
  Eden_14: [0.65, 0.65, 0.62],
  Eden_15: [0.3, 0.65, 0.2],
  Eden_16: [0.5, 0.32, 0.15],
  Eden_17: [0.5, 0.25, 0.65],
  Eden_18: [0.9, 0.8, 0.2],
  Eden_19: [0.15, 0.15, 0.18],
  Eden_20: [0.75, 0.75, 0.75],
  Eden_21: [0.95, 0.95, 0.95],
  Eden_22: [0.6, 0.4, 0.22],
  Eden_23: [0.6, 0.4, 0.22],
  Eden_24: [0.0, 0.75, 0.7],
  Eden_25: [0.9, 0.8, 0.2],
  Eden_26: [0.85, 0.35, 0.55],
  Eden_27: [0.3, 0.36, 0.3],
}

// Geometry registration record for re-registration on backend switch
type GeoReg =
  | { id: number; vertices: Float32Array; indices: Uint16Array | Uint32Array; skinned: false }
  | {
      id: number
      vertices: Float32Array
      indices: Uint16Array | Uint32Array
      joints: Uint8Array
      weights: Float32Array
      skinned: true
    }

export async function startDemo(canvas: HTMLCanvasElement) {
  // ── Persistent state (survives backend switches) ───────────────────
  const world = new World()
  const input = new InputManager()
  const geoRegs: GeoReg[] = []
  const webgpuAvailable = !!navigator.gpu

  // ── Mutable renderer state ─────────────────────────────────────────
  let currentCanvas = canvas
  let renderer: IRenderer = await createRenderer(currentCanvas, 10000)
  let orbit = new OrbitControls(currentCanvas)
  let aspect = currentCanvas.width / currentCanvas.height
  let resizeObs: ResizeObserver

  function trackGeo(id: number, vertices: Float32Array, indices: Uint16Array | Uint32Array) {
    renderer.registerGeometry(id, vertices, indices)
    geoRegs.push({ id, vertices, indices, skinned: false })
  }

  function trackSkinnedGeo(
    id: number,
    vertices: Float32Array,
    indices: Uint16Array | Uint32Array,
    joints: Uint8Array,
    weights: Float32Array,
  ) {
    renderer.registerSkinnedGeometry(id, vertices, indices, joints, weights)
    geoRegs.push({ id, vertices, indices, joints, weights, skinned: true })
  }

  function observeResize() {
    resizeObs = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        const dpr = devicePixelRatio
        const w = (width * dpr) | 0
        const h = (height * dpr) | 0
        if (w === 0 || h === 0) continue
        currentCanvas.width = w
        currentCanvas.height = h
        aspect = w / h
        renderer.resize(w, h)
      }
    })
    resizeObs.observe(currentCanvas)
  }

  async function switchBackend(type: BackendType) {
    if (renderer.backendType === type) return

    // Save orbit state
    const { theta, phi, radius, targetX, targetY, targetZ } = orbit

    // Cleanup
    renderer.destroy()
    orbit.destroy()
    resizeObs.disconnect()

    // Replace canvas (a canvas can only have one context type)
    const newCanvas = document.createElement('canvas')
    newCanvas.style.cssText = currentCanvas.style.cssText
    newCanvas.width = currentCanvas.width
    newCanvas.height = currentCanvas.height
    currentCanvas.parentNode!.replaceChild(newCanvas, currentCanvas)
    currentCanvas = newCanvas

    // Create new renderer + re-register geometries
    renderer = await createRenderer(currentCanvas, 10000, type)
    for (const reg of geoRegs) {
      if (reg.skinned) {
        renderer.registerSkinnedGeometry(reg.id, reg.vertices, reg.indices, reg.joints, reg.weights)
      } else {
        renderer.registerGeometry(reg.id, reg.vertices, reg.indices)
      }
    }

    // Restore orbit controls
    orbit = new OrbitControls(currentCanvas)
    orbit.theta = theta
    orbit.phi = phi
    orbit.radius = radius
    orbit.targetX = targetX
    orbit.targetY = targetY
    orbit.targetZ = targetZ
    orbit.updateEye()

    observeResize()
  }

  // ── Register geometries ─────────────────────────────────────────────
  const sphere = createSphereGeometry(16, 24)
  trackGeo(SPHERE_GEO_ID, sphere.vertices, sphere.indices)

  // ── Camera entity ─────────────────────────────────────────────────
  const cam = world.createEntity()
  world.addTransform(cam, { position: [0, -8, 3] })
  world.addCamera(cam, { fov: (60 * Math.PI) / 180, near: 0.1, far: 500 })
  world.activeCamera = cam

  // ── Load static GLB (megaxe + Eden) ───────────────────────────────
  const megaxeColors = new Map<number, [number, number, number]>([
    [0, [1, 1, 1]], // material 0 → white
    [1, [0, 0, 0]], // material 1 → black
    [2, [0, 0.9, 0.8]], // material 2 → bright teal
  ])
  const { meshes: glbMeshes } = await loadGlb('/static-bundle.glb', '/draco-1.5.7/', megaxeColors)
  const megaxe = glbMeshes.find(m => m.name === 'megaxe')
  if (megaxe) {
    trackGeo(MEGAXE_GEO_ID, megaxe.vertices, megaxe.indices)
    const axe = world.createEntity()
    world.addTransform(axe, { position: [3, 0, 0], scale: megaxe.scale })
    world.addMeshInstance(axe, { geometryId: MEGAXE_GEO_ID, color: [1, 1, 1] })
  }

  // ── Eden entity (merged into 1 draw call) ──────────────────────
  const edenMeshes = glbMeshes.filter(m => m.name.startsWith('Eden_'))
  if (edenMeshes.length > 0) {
    const merged = mergeGeometries(
      edenMeshes.map(em => ({
        vertices: em.vertices,
        indices: em.indices,
        color: EDEN_COLORS[em.name] ?? [1, 1, 1],
      })),
    )
    trackGeo(EDEN_GEO_START, merged.vertices, merged.indices)
    const eden = world.createEntity()
    world.addTransform(eden, { position: [-30, 0, 0], scale: edenMeshes[0]!.scale })
    world.addMeshInstance(eden, { geometryId: EDEN_GEO_START, color: [1, 1, 1] })
  }

  // ── Load player GLB ───────────────────────────────────────────────
  const skinInstances: SkinInstance[] = []
  const playerResult = await loadGlb('/player-bundle.glb')
  const bodyMesh = playerResult.meshes.find(m => m.name === 'Body' && m.skinIndex !== undefined)

  if (bodyMesh && bodyMesh.skinJoints && bodyMesh.skinWeights && bodyMesh.skinIndex !== undefined) {
    trackSkinnedGeo(PLAYER_BODY_GEO_ID, bodyMesh.vertices, bodyMesh.indices, bodyMesh.skinJoints, bodyMesh.skinWeights)

    const skin = playerResult.skins[bodyMesh.skinIndex]!
    const skeleton = createSkeleton(skin, playerResult.nodeTransforms)

    // Find "Run" animation
    let runClipIdx = 0
    for (let i = 0; i < playerResult.animations.length; i++) {
      if (playerResult.animations[i]!.name.toLowerCase().includes('run')) {
        runClipIdx = i
        break
      }
    }

    const skinInst = createSkinInstance(skeleton, runClipIdx)
    skinInstances.push(skinInst)

    // Player entity (replaces the cube)
    const player = world.createEntity()
    world.addTransform(player, { position: [0, 0, 0], scale: bodyMesh.scale })
    world.addMeshInstance(player, { geometryId: PLAYER_BODY_GEO_ID, color: [1, 1, 1] })
    world.addSkinned(player, 0)
    world.addInputReceiver(player)
  }

  // ── Sphere entities (behind the player, positive Y) ──────────────
  const SPHERE_COLS = 7
  const SPHERE_ROWS = 5
  const SPACING = 2.5
  for (let row = 0; row < SPHERE_ROWS; row++) {
    for (let col = 0; col < SPHERE_COLS; col++) {
      const s = world.createEntity()
      const x = (col - (SPHERE_COLS - 1) / 2) * SPACING
      const z = (row - (SPHERE_ROWS - 1) / 2) * SPACING
      world.addTransform(s, { position: [x, 8, z], scale: [0.8, 0.8, 0.8] })
      // Vary color by position
      const r = 0.3 + (col / (SPHERE_COLS - 1)) * 0.5
      const g = 0.3 + (row / (SPHERE_ROWS - 1)) * 0.5
      const b = 0.6
      world.addMeshInstance(s, { geometryId: SPHERE_GEO_ID, color: [r, g, b] })
    }
  }

  // ── Lighting ──────────────────────────────────────────────────────
  world.setDirectionalLight([-1, -1, -1], [1, 1, 1])
  world.setAmbientLight([0.4, 0.4, 0.4])

  // ── Stats overlay ────────────────────────────────────────────────
  const stats = document.createElement('div')
  stats.style.cssText =
    'position:fixed;top:8px;left:8px;color:#fff;font:14px/1.4 monospace;background:rgba(0,0,0,0.5);padding:4px 8px;border-radius:4px;pointer-events:none'
  document.body.appendChild(stats)
  let frames = 0
  let fpsTime = performance.now()
  let fps = 0
  setInterval(() => {
    const now = performance.now()
    fps = Math.round((frames * 1000) / (now - fpsTime))
    frames = 0
    fpsTime = now
    stats.textContent = `FPS: ${fps}  Draw calls: ${renderer.drawCalls}  ${renderer.backendType}`
  }, 500)

  // ── Backend toggle checkbox ───────────────────────────────────────
  if (webgpuAvailable) {
    const toggle = document.createElement('div')
    toggle.style.cssText =
      'position:fixed;top:8px;right:8px;color:#fff;font:14px/1.4 monospace;background:rgba(0,0,0,0.5);padding:4px 8px;border-radius:4px;display:flex;align-items:center;gap:6px;user-select:none'
    const checkbox = document.createElement('input')
    checkbox.type = 'checkbox'
    checkbox.id = 'backend-toggle'
    checkbox.checked = renderer.backendType === 'webgpu'
    checkbox.style.cssText = 'cursor:pointer'
    const label = document.createElement('label')
    label.htmlFor = 'backend-toggle'
    label.textContent = 'WebGPU'
    label.style.cssText = 'cursor:pointer'
    toggle.appendChild(checkbox)
    toggle.appendChild(label)
    document.body.appendChild(toggle)
    checkbox.addEventListener('change', async () => {
      checkbox.disabled = true
      try {
        await switchBackend(checkbox.checked ? 'webgpu' : 'webgl')
      } catch {
        // Revert checkbox if switch failed
        checkbox.checked = renderer.backendType === 'webgpu'
      }
      checkbox.disabled = false
    })
  }

  // ── Resize handling ───────────────────────────────────────────────
  observeResize()

  // ── Pre-allocate render masks ─────────────────────────────────────
  const MAX_ENTITIES = 10_000
  const renderMask = new Uint8Array(MAX_ENTITIES)
  const skinnedMask = new Uint8Array(MAX_ENTITIES)
  const meshMask = TRANSFORM | MESH_INSTANCE

  // ── Game loop ─────────────────────────────────────────────────────
  let lastTime = performance.now()

  const inputMask = TRANSFORM | INPUT_RECEIVER

  function loop(now: number) {
    const dt = Math.min((now - lastTime) / 1000, 0.1) // cap delta
    lastTime = now

    // ── Input system ──────────────────────────────────────────────
    for (let i = 0; i < world.entityCount; i++) {
      if ((world.componentMask[i]! & inputMask) !== inputMask) continue
      const pi = i * 3
      if (input.isDown('KeyW')) world.positions[pi + 1]! += MOVE_SPEED * dt
      if (input.isDown('KeyS')) world.positions[pi + 1]! -= MOVE_SPEED * dt
      if (input.isDown('KeyA')) world.positions[pi]! -= MOVE_SPEED * dt
      if (input.isDown('KeyD')) world.positions[pi]! += MOVE_SPEED * dt
    }

    // ── Animation system ──────────────────────────────────────────
    for (const inst of skinInstances) {
      updateSkinInstance(inst, playerResult.animations, dt)
    }

    // ── Transform system ──────────────────────────────────────────
    for (let i = 0; i < world.entityCount; i++) {
      if (!(world.componentMask[i]! & TRANSFORM)) continue
      m4FromTRS(world.worldMatrices, i * 16, world.positions, i * 3, world.rotations, i * 3, world.scales, i * 3)
    }

    // ── Camera system (orbit controls) ─────────────────────────────
    for (let i = 0; i < world.entityCount; i++) {
      if (!(world.componentMask[i]! & CAMERA)) continue
      m4LookAt(world.viewMatrices, i * 16, orbit.eye, 0, orbit.target, 0, UP, 0)
      renderer.perspective(world.projMatrices, i * 16, world.fovs[i]!, aspect, world.nears[i]!, world.fars[i]!)
    }

    // ── Build render masks ──────────────────────────────────────────
    for (let i = 0; i < world.entityCount; i++) {
      const mask = world.componentMask[i]!
      renderMask[i] = (mask & meshMask) === meshMask ? 1 : 0
      skinnedMask[i] = mask & SKINNED ? 1 : 0
    }

    // ── Render ────────────────────────────────────────────────────
    const c = world.activeCamera
    const scene: RenderScene = {
      cameraView: world.viewMatrices,
      cameraViewOffset: c * 16,
      cameraProj: world.projMatrices,
      cameraProjOffset: c * 16,
      lightDirection: world.directionalLightDir,
      lightDirColor: world.directionalLightColor,
      lightAmbientColor: world.ambientLightColor,
      entityCount: world.entityCount,
      renderMask,
      skinnedMask,
      positions: world.positions,
      scales: world.scales,
      worldMatrices: world.worldMatrices,
      colors: world.colors,
      geometryIds: world.geometryIds,
      skinInstanceIds: world.skinInstanceIds,
      skinInstances,
    }
    renderer.render(scene)
    frames++

    requestAnimationFrame(loop)
  }

  requestAnimationFrame(loop)
}
