import {
  Scene,
  Mesh,
  createRenderer,
  OrbitControls,
  loadGlb,
  createSkeleton,
  createSkinInstance,
  updateSkinInstance,
  transitionTo,
  createSphereGeometry,
  mergeGeometries,
  cubeVertices,
  cubeIndices,
  m4LookAt,
  m4Multiply,
  v3Normalize,
  v3Scale,
} from './engine/index.ts'

import type { BackendType, IRenderer } from './engine/index.ts'

const MOVE_SPEED = 3
const MEGAXE_GEO_ID = 2
const EDEN_GEO_START = 3
const PLAYER_BODY_GEO_ID = 4
const CUBE_GEO_ID = 5
const SKY_GEO_ID = 6
const UP = new Float32Array([0, 0, 1])

// Shadow mapping — fixed volume covering the entire Eden map (0,0 to 200,200)
const SHADOW_EXTENT = 150
const SHADOW_NEAR = 1
const SHADOW_FAR = 800
const SHADOW_DISTANCE = 400
const lightView = new Float32Array(16)
const lightProj = new Float32Array(16)
const lightVP = new Float32Array(16)
const lightEye = new Float32Array(3)
const lightTarget = new Float32Array([100, 100, 0])
const lightDirNorm = new Float32Array(3)

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
  const scene = new Scene()
  const keys = new Set<string>()
  window.addEventListener('keydown', e => {
    if ('KeyW KeyA KeyS KeyD'.includes(e.code)) {
      e.preventDefault()
      keys.add(e.code)
    }
  })
  window.addEventListener('keyup', e => keys.delete(e.code))
  const geoRegs: GeoReg[] = []
  const webgpuAvailable = !!navigator.gpu

  // ── Mutable renderer state ─────────────────────────────────────────
  let currentCanvas = canvas
  const savedBackend = localStorage.getItem('renderer-backend') as BackendType | null
  let renderer: IRenderer = await createRenderer(currentCanvas, 10000, savedBackend ?? undefined)
  let orbit = new OrbitControls(currentCanvas)
  orbit.targetX = 50
  orbit.targetY = 80
  orbit.targetZ = 0
  orbit.theta = Math.PI
  orbit.phi = Math.atan2(5, 10)
  orbit.radius = Math.hypot(10, 5) * 1.5
  orbit.updateEye()
  let aspect = currentCanvas.width / currentCanvas.height
  let resizeObs: ResizeObserver
  let switching = false

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

    switching = true

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
    switching = false
  }

  // ── Register geometries ─────────────────────────────────────────────
  trackGeo(CUBE_GEO_ID, cubeVertices, cubeIndices)
  const skySphere = createSphereGeometry(16, 24, true)
  trackGeo(SKY_GEO_ID, skySphere.vertices, skySphere.indices)

  // ── Camera config ───────────────────────────────────────────────────
  const fov = (60 * Math.PI) / 180
  const near = 0.1
  const far = 5000

  // ── Sky sphere ──────────────────────────────────────────────────────
  scene.add(
    new Mesh({
      geometryId: SKY_GEO_ID,
      position: [50, 80, 0],
      scale: [1500, 1500, 1500],
      color: [85 / 255, 221 / 255, 1],
      unlit: true,
    }),
  )

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
    scene.add(
      new Mesh({
        geometryId: MEGAXE_GEO_ID,
        position: [44, 80, 0],
        scale: megaxe.scale,
        color: [1, 1, 1],
      }),
    )
  }

  // ── Eden (merged into 1 draw call) ──────────────────────────────
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
    scene.add(
      new Mesh({
        geometryId: EDEN_GEO_START,
        position: [0, 0, 0],
        scale: edenMeshes[0]!.scale,
        color: [1, 1, 1],
      }),
    )
  }

  // ── Load player GLB ───────────────────────────────────────────────
  const animCycleCallbacks: ((dt: number) => void)[] = []
  const playerResult = await loadGlb('/player-bundle.glb')
  const bodyMesh = playerResult.meshes.find(m => m.name === 'Body' && m.skinIndex !== undefined)

  let player: Mesh | undefined

  if (bodyMesh && bodyMesh.skinJoints && bodyMesh.skinWeights && bodyMesh.skinIndex !== undefined) {
    trackSkinnedGeo(PLAYER_BODY_GEO_ID, bodyMesh.vertices, bodyMesh.indices, bodyMesh.skinJoints, bodyMesh.skinWeights)

    const skin = playerResult.skins[bodyMesh.skinIndex]!
    const skeleton = createSkeleton(skin, playerResult.nodeTransforms)

    // Build animation name → index map
    const animNames = ['Idle', 'Jump', 'Run', 'SlashRight']
    const animIndices: number[] = []
    for (const name of animNames) {
      const idx = playerResult.animations.findIndex(a => a.name.toLowerCase().includes(name.toLowerCase()))
      animIndices.push(idx >= 0 ? idx : 0)
    }

    const skinInst = createSkinInstance(skeleton, animIndices[0]!)
    updateSkinInstance(skinInst, playerResult.animations, 0) // compute initial joint matrices
    scene.skinInstances.push(skinInst)

    // Player mesh (skinned + movable via WASD)
    player = scene.add(
      new Mesh({
        geometryId: PLAYER_BODY_GEO_ID,
        position: [42, 80, 0],
        scale: bodyMesh.scale,
        color: [1, 1, 1],
        skinned: true,
        skinInstanceId: 0,
      }),
    )

    // Cycle through animations every second with 0.2s crossfade
    let animCycleIndex = 0
    let animTimer = 0
    const ANIM_INTERVAL = 1.0
    const BLEND_DURATION = 0.2
    animCycleCallbacks.push((dt: number) => {
      animTimer += dt
      if (animTimer >= ANIM_INTERVAL) {
        animTimer -= ANIM_INTERVAL
        animCycleIndex = (animCycleIndex + 1) % animIndices.length
        transitionTo(skinInst, animIndices[animCycleIndex]!, BLEND_DURATION)
      }
    })
  }

  // ── Transparent cubes ───────────────────────────────────────────
  const CUBE_COUNT = 5
  const cubeColors: [number, number, number][] = [
    [1, 0.2, 0.2],
    [0.2, 0.8, 0.2],
    [0.3, 0.4, 1],
    [1, 0.8, 0.1],
    [0.9, 0.3, 0.9],
  ]
  const cubes: Mesh[] = []
  for (let i = 0; i < CUBE_COUNT; i++) {
    const angle = (i / CUBE_COUNT) * Math.PI * 2
    const x = 50 + Math.cos(angle) * 4
    const y = 80 + Math.sin(angle) * 4
    const cube = scene.add(
      new Mesh({ geometryId: CUBE_GEO_ID, position: [x, y, 3], color: cubeColors[i]!, alpha: 0.5 }),
    )
    cubes.push(cube)
  }

  // ── Lighting ──────────────────────────────────────────────────────
  scene.setDirectionalLight([-1, -1, -2.5], [0.3, 0.3, 0.3])
  scene.setAmbientLight([0.95, 0.95, 0.95])

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
      const target: BackendType = checkbox.checked ? 'webgpu' : 'webgl'
      try {
        await switchBackend(target)
        localStorage.setItem('renderer-backend', target)
      } catch {
        // Revert checkbox if switch failed
        checkbox.checked = renderer.backendType === 'webgpu'
      }
      checkbox.disabled = false
    })
  }

  // ── Resize handling ───────────────────────────────────────────────
  observeResize()

  // ── Game loop ─────────────────────────────────────────────────────
  let lastTime = performance.now()

  function loop(now: number) {
    if (switching) {
      requestAnimationFrame(loop)
      return
    }
    const dt = Math.min((now - lastTime) / 1000, 0.1) // cap delta
    lastTime = now

    // ── Input ───────────────────────────────────────────────────────
    if (player) {
      if (keys.has('KeyW')) player.position[1]! += MOVE_SPEED * dt
      if (keys.has('KeyS')) player.position[1]! -= MOVE_SPEED * dt
      if (keys.has('KeyA')) player.position[0]! -= MOVE_SPEED * dt
      if (keys.has('KeyD')) player.position[0]! += MOVE_SPEED * dt
    }

    // ── Rotate transparent cubes ─────────────────────────────────
    for (const cube of cubes) {
      cube.rotation[2]! += dt
    }

    // ── Animation cycling ─────────────────────────────────────────
    for (const cb of animCycleCallbacks) cb(dt)

    // ── Animation update ──────────────────────────────────────────
    for (const inst of scene.skinInstances) {
      updateSkinInstance(inst, playerResult.animations, dt)
    }

    // ── Camera (orbit controls) ──────────────────────────────────
    m4LookAt(scene.viewMatrix, 0, orbit.eye, 0, orbit.target, 0, UP, 0)
    renderer.perspective(scene.projMatrix, 0, fov, aspect, near, far)

    // ── Shadow: compute light VP matrix ──────────────────────────
    v3Normalize(lightDirNorm, 0, scene.lightDirection, 0)
    v3Scale(lightEye, 0, lightDirNorm, 0, -SHADOW_DISTANCE)
    lightEye[0]! += lightTarget[0]!
    lightEye[1]! += lightTarget[1]!
    lightEye[2]! += lightTarget[2]!
    m4LookAt(lightView, 0, lightEye, 0, lightTarget, 0, UP, 0)
    renderer.ortho(lightProj, 0, -SHADOW_EXTENT, SHADOW_EXTENT, -SHADOW_EXTENT, SHADOW_EXTENT, SHADOW_NEAR, SHADOW_FAR)
    m4Multiply(lightVP, 0, lightProj, 0, lightView, 0)

    // ── Render ──────────────────────────────────────────────────────
    renderer.render(scene.buildRenderScene({ lightViewProj: lightVP, bias: 0.0001 }))
    frames++

    requestAnimationFrame(loop)
  }

  requestAnimationFrame(loop)
}
