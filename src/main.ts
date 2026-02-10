import { World, TRANSFORM, MESH_INSTANCE, CAMERA, INPUT_RECEIVER } from './ecs.ts'
import { m4FromTRS, m4LookAt, m4Perspective } from './math.ts'
import { cubeVertices, cubeIndices, createSphereGeometry } from './geometry.ts'
import { initGPU, Renderer } from './renderer.ts'
import { InputManager } from './input.ts'
import { OrbitControls } from './orbit-controls.ts'
import { loadGlb } from './gltf.ts'

const MOVE_SPEED = 3
const CUBE_GEO_ID = 0
const SPHERE_GEO_ID = 1
const MEGAXE_GEO_ID = 2
const UP = new Float32Array([0, 1, 0])

export async function startDemo(canvas: HTMLCanvasElement) {
  // ── Init ──────────────────────────────────────────────────────────
  const { device, context, format } = await initGPU(canvas)
  const world = new World()
  const input = new InputManager()
  const renderer = new Renderer(device, context, format, canvas, 10000)
  const orbit = new OrbitControls(canvas)

  renderer.registerGeometry(CUBE_GEO_ID, cubeVertices, cubeIndices)

  const sphere = createSphereGeometry(16, 24)
  renderer.registerGeometry(SPHERE_GEO_ID, sphere.vertices, sphere.indices)

  // ── Camera entity ─────────────────────────────────────────────────
  const cam = world.createEntity()
  world.addTransform(cam, { position: [0, 3, 8] })
  world.addCamera(cam, { fov: (60 * Math.PI) / 180, near: 0.1, far: 500 })
  world.activeCamera = cam

  // ── Cube entity ───────────────────────────────────────────────────
  const cube = world.createEntity()
  world.addTransform(cube, { position: [0, 0, 0], scale: [1, 1, 1] })
  world.addMeshInstance(cube, { geometryId: CUBE_GEO_ID, color: [0.8, 0.2, 0.2] })
  world.addInputReceiver(cube)

  // ── Megaxe entity ────────────────────────────────────────────────
  const megaxeColors = new Map<number, [number, number, number]>([
    [0, [1, 1, 1]],       // material 0 → white
    [1, [0, 0, 0]],       // material 1 → black
    [2, [0, 0.9, 0.8]],   // material 2 → bright teal
  ])
  const glbMeshes = await loadGlb('/static-bundle.glb', '/draco-1.5.7/', megaxeColors)
  const megaxe = glbMeshes.find(m => m.name === 'megaxe')
  if (megaxe) {
    renderer.registerGeometry(MEGAXE_GEO_ID, megaxe.vertices, megaxe.indices)
    const axe = world.createEntity()
    world.addTransform(axe, { position: [3, 0, 0], scale: megaxe.scale })
    world.addMeshInstance(axe, { geometryId: MEGAXE_GEO_ID, color: [1, 1, 1] })
  }

  // ── Megaxe grid (behind spheres) ──────────────────────────────────
  if (megaxe) {
    const GRID_COLS = 100
    const GRID_ROWS = 50
    const GRID_SPACING = 3
    const GRID_Z = -20
    for (let row = 0; row < GRID_ROWS; row++) {
      for (let col = 0; col < GRID_COLS; col++) {
        const e = world.createEntity()
        const x = (col - (GRID_COLS - 1) / 2) * GRID_SPACING
        const y = (row - (GRID_ROWS - 1) / 2) * GRID_SPACING
        world.addTransform(e, { position: [x, y, GRID_Z], scale: megaxe.scale })
        world.addMeshInstance(e, { geometryId: MEGAXE_GEO_ID, color: [1, 1, 1] })
      }
    }
  }

  // ── Sphere entities (behind the cube, negative Z) ─────────────────
  const SPHERE_COLS = 7
  const SPHERE_ROWS = 5
  const SPACING = 2.5
  for (let row = 0; row < SPHERE_ROWS; row++) {
    for (let col = 0; col < SPHERE_COLS; col++) {
      const s = world.createEntity()
      const x = (col - (SPHERE_COLS - 1) / 2) * SPACING
      const y = (row - (SPHERE_ROWS - 1) / 2) * SPACING
      world.addTransform(s, { position: [x, y, -8], scale: [0.8, 0.8, 0.8] })
      // Vary color by position
      const r = 0.3 + (col / (SPHERE_COLS - 1)) * 0.5
      const g = 0.3 + (row / (SPHERE_ROWS - 1)) * 0.5
      const b = 0.6
      world.addMeshInstance(s, { geometryId: SPHERE_GEO_ID, color: [r, g, b] })
    }
  }

  // ── Lighting ──────────────────────────────────────────────────────
  world.setDirectionalLight([-1, -1, -1], [1, 1, 1])
  world.setAmbientLight([0.15, 0.15, 0.15])

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
    stats.textContent = `FPS: ${fps}  Draw calls: ${renderer.drawCalls}`
  }, 500)

  // ── Resize handling ───────────────────────────────────────────────
  let aspect = canvas.width / canvas.height

  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const { width, height } = entry.contentRect
      const dpr = devicePixelRatio
      const w = (width * dpr) | 0
      const h = (height * dpr) | 0
      if (w === 0 || h === 0) continue
      canvas.width = w
      canvas.height = h
      aspect = w / h
      renderer.resize(w, h)
    }
  })
  resizeObserver.observe(canvas)

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
      if (input.isDown('KeyW')) world.positions[pi + 2]! -= MOVE_SPEED * dt
      if (input.isDown('KeyS')) world.positions[pi + 2]! += MOVE_SPEED * dt
      if (input.isDown('KeyA')) world.positions[pi]! -= MOVE_SPEED * dt
      if (input.isDown('KeyD')) world.positions[pi]! += MOVE_SPEED * dt
    }

    // ── Transform system ──────────────────────────────────────────
    for (let i = 0; i < world.entityCount; i++) {
      if (!(world.componentMask[i]! & TRANSFORM)) continue
      m4FromTRS(
        world.worldMatrices, i * 16,
        world.positions, i * 3,
        world.rotations, i * 3,
        world.scales, i * 3,
      )
    }

    // ── Camera system (orbit controls) ─────────────────────────────
    for (let i = 0; i < world.entityCount; i++) {
      if (!(world.componentMask[i]! & CAMERA)) continue
      m4LookAt(
        world.viewMatrices, i * 16,
        orbit.eye, 0,
        orbit.target, 0,
        UP, 0,
      )
      m4Perspective(
        world.projMatrices, i * 16,
        world.fovs[i]!,
        aspect,
        world.nears[i]!,
        world.fars[i]!,
      )
    }

    // ── Render ────────────────────────────────────────────────────
    renderer.render(world)
    frames++

    requestAnimationFrame(loop)
  }

  requestAnimationFrame(loop)
}
