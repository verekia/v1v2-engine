import { World, TRANSFORM, MESH_INSTANCE, CAMERA, INPUT_RECEIVER } from './ecs.ts'
import { m4FromTRS, m4LookAt, m4Perspective } from './math.ts'
import { cubeVertices, cubeIndices, createSphereGeometry, mergeGeometries } from './geometry.ts'
import { initGPU, Renderer } from './renderer.ts'
import { InputManager } from './input.ts'
import { OrbitControls } from './orbit-controls.ts'
import { loadGlb } from './gltf.ts'
import { createSkeleton, createSkinInstance, updateSkinInstance, type SkinInstance } from './skin.ts'

const MOVE_SPEED = 3
const SPHERE_GEO_ID = 1
const MEGAXE_GEO_ID = 2
const EDEN_GEO_START = 3
const PLAYER_BODY_GEO_ID = 4
const UP = new Float32Array([0, 1, 0])

const EDEN_COLORS: Record<string, [number, number, number]> = {
  Eden_1: [0.78, 0.44, 0.25],
  Eden_2: [0.85, 0.68, 0.30],
  Eden_3: [0.75, 0.75, 0.75],
  Eden_4: [0.78, 0.30, 0.20],
  Eden_5: [0.25, 0.55, 0.20],
  Eden_6: [0.30, 0.36, 0.30],
  Eden_7: [0.75, 0.75, 0.75],
  Eden_8: [0.15, 0.15, 0.18],
  Eden_9: [0.00, 0.75, 0.70],
  Eden_10: [0.55, 0.50, 0.42],
  Eden_11: [0.85, 0.35, 0.55],
  Eden_12: [0.95, 0.95, 0.95],
  Eden_13: [0.80, 0.65, 0.15],
  Eden_14: [0.65, 0.65, 0.62],
  Eden_15: [0.30, 0.65, 0.20],
  Eden_16: [0.50, 0.32, 0.15],
  Eden_17: [0.50, 0.25, 0.65],
  Eden_18: [0.90, 0.80, 0.20],
  Eden_19: [0.15, 0.15, 0.18],
  Eden_20: [0.75, 0.75, 0.75],
  Eden_21: [0.95, 0.95, 0.95],
  Eden_22: [0.60, 0.40, 0.22],
  Eden_23: [0.60, 0.40, 0.22],
  Eden_24: [0.00, 0.75, 0.70],
  Eden_25: [0.90, 0.80, 0.20],
  Eden_26: [0.85, 0.35, 0.55],
  Eden_27: [0.30, 0.36, 0.30],
}

export async function startDemo(canvas: HTMLCanvasElement) {
  // ── Init ──────────────────────────────────────────────────────────
  const { device, context, format } = await initGPU(canvas)
  const world = new World()
  const input = new InputManager()
  const renderer = new Renderer(device, context, format, canvas, 10000)
  const orbit = new OrbitControls(canvas)

  const sphere = createSphereGeometry(16, 24)
  renderer.registerGeometry(SPHERE_GEO_ID, sphere.vertices, sphere.indices)

  // ── Camera entity ─────────────────────────────────────────────────
  const cam = world.createEntity()
  world.addTransform(cam, { position: [0, 3, 8] })
  world.addCamera(cam, { fov: (60 * Math.PI) / 180, near: 0.1, far: 500 })
  world.activeCamera = cam

  // ── Load static GLB (megaxe + Eden) ───────────────────────────────
  const megaxeColors = new Map<number, [number, number, number]>([
    [0, [1, 1, 1]],       // material 0 → white
    [1, [0, 0, 0]],       // material 1 → black
    [2, [0, 0.9, 0.8]],   // material 2 → bright teal
  ])
  const { meshes: glbMeshes } = await loadGlb('/static-bundle.glb', '/draco-1.5.7/', megaxeColors)
  const megaxe = glbMeshes.find(m => m.name === 'megaxe')
  if (megaxe) {
    renderer.registerGeometry(MEGAXE_GEO_ID, megaxe.vertices, megaxe.indices)
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
    renderer.registerGeometry(EDEN_GEO_START, merged.vertices, merged.indices)
    const eden = world.createEntity()
    world.addTransform(eden, { position: [-30, 0, 0], scale: edenMeshes[0]!.scale })
    world.addMeshInstance(eden, { geometryId: EDEN_GEO_START, color: [1, 1, 1] })
  }

  // ── Load player GLB ───────────────────────────────────────────────
  const skinInstances: SkinInstance[] = []
  const playerResult = await loadGlb('/player-bundle.glb')
  const bodyMesh = playerResult.meshes.find(m => m.name === 'Body' && m.skinIndex !== undefined)

  if (bodyMesh && bodyMesh.skinJoints && bodyMesh.skinWeights && bodyMesh.skinIndex !== undefined) {
    renderer.registerSkinnedGeometry(
      PLAYER_BODY_GEO_ID,
      bodyMesh.vertices,
      bodyMesh.indices,
      bodyMesh.skinJoints,
      bodyMesh.skinWeights,
    )

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

    // ── Animation system ──────────────────────────────────────────
    for (const inst of skinInstances) {
      updateSkinInstance(inst, playerResult.animations, dt)
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
    renderer.render(world, skinInstances)
    frames++

    requestAnimationFrame(loop)
  }

  requestAnimationFrame(loop)
}
