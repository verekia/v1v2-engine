import {
  createScene,
  Mesh,
  OrbitControls,
  loadGlb,
  loadKTX2,
  createSkeleton,
  createSkinInstance,
  updateSkinInstance,
  transitionTo,
  createBoxGeometry,
  createSphereGeometry,
  mergeGeometries,
  createRaycastHit,
  HtmlOverlay,
  HtmlElement,
} from './engine/index.ts'

import type { BackendType } from './engine/index.ts'

function splitByMaterial(
  vertices: Float32Array,
  indices: Uint16Array | Uint32Array,
  materialIndices: Uint8Array,
  matchFn: (matIdx: number) => boolean,
): { vertices: Float32Array; indices: Uint16Array | Uint32Array } | null {
  const vertexMap = new Map<number, number>()
  const newVerts: number[] = []
  const newIdx: number[] = []

  for (let f = 0; f < indices.length; f += 3) {
    const i0 = indices[f]!
    if (!matchFn(materialIndices[i0]!)) continue
    for (let k = 0; k < 3; k++) {
      const oldIdx = indices[f + k]!
      if (!vertexMap.has(oldIdx)) {
        const newI = vertexMap.size
        vertexMap.set(oldIdx, newI)
        for (let j = 0; j < 9; j++) newVerts.push(vertices[oldIdx * 9 + j]!)
      }
      newIdx.push(vertexMap.get(oldIdx)!)
    }
  }

  if (newIdx.length === 0) return null
  return {
    vertices: new Float32Array(newVerts),
    indices: newIdx.length <= 65535 ? new Uint16Array(newIdx) : new Uint32Array(newIdx),
  }
}

const MOVE_SPEED = 3
const UP: [number, number, number] = [0, 0, 1]

const EDEN_COLORS: Record<string, [number, number, number]> = {
  Eden_1: [0.78, 0.44, 0.25],
  Eden_2: [0.85, 0.68, 0.3],
  Eden_3: [0.75, 0.75, 0.75],
  Eden_4: [0.725, 0.38, 0.09],
  Eden_5: [0.25, 0.55, 0.2],
  Eden_6: [0.212, 0.212, 0.212],
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

export async function startDemo(canvas: HTMLCanvasElement) {
  const webgpuAvailable = !!navigator.gpu
  const savedBackend = localStorage.getItem('renderer-backend') as BackendType | null
  let currentCanvas = canvas
  const scene = await createScene(currentCanvas, { maxEntities: 1000, backend: savedBackend ?? undefined })

  // ── Input (demo-only) ────────────────────────────────────────────────
  const keys = new Set<string>()
  window.addEventListener('keydown', e => {
    if ('KeyW KeyA KeyS KeyD'.includes(e.code)) {
      e.preventDefault()
      keys.add(e.code)
    }
  })
  window.addEventListener('keyup', e => keys.delete(e.code))

  // ── Camera ────────────────────────────────────────────────────────────
  let orbit = new OrbitControls(currentCanvas)
  orbit.targetX = 50
  orbit.targetY = 80
  orbit.targetZ = 0
  orbit.theta = Math.PI
  orbit.phi = Math.atan2(5, 10)
  orbit.radius = Math.hypot(10, 5) * 1.5
  orbit.updateEye()

  scene.camera.fov = (60 * Math.PI) / 180
  scene.camera.near = 0.1
  scene.camera.far = 5000
  scene.camera.up.set(UP)

  // ── Lighting ──────────────────────────────────────────────────────────
  scene.setDirectionalLight([-1, -1, -2.5], [0.3, 0.3, 0.3])
  scene.setAmbientLight([0.95, 0.95, 0.95])

  // ── Shadows ───────────────────────────────────────────────────────────
  scene.shadow.enabled = true
  scene.shadow.target.set([100, 100, 0])
  scene.shadow.distance = 400
  scene.shadow.extent = 150
  scene.shadow.near = 1
  scene.shadow.far = 800
  scene.shadow.bias = 0.0001

  // ── Bloom ───────────────────────────────────────────────────────────
  scene.bloom.enabled = true
  scene.bloom.intensity = 0.5
  scene.bloom.threshold = 0
  scene.bloom.radius = 1
  scene.bloom.whiten = 0.5

  // ── Outline ────────────────────────────────────────────────────────
  scene.outline.enabled = true
  scene.outline.thickness = 5
  scene.outline.distanceFactor = 10

  // ── Register geometries ───────────────────────────────────────────────
  const box = createBoxGeometry(2, 2, 2)
  const cubeGeo = scene.registerGeometry(box.vertices, box.indices)
  const skySphere = createSphereGeometry(1, 24, 16, true)
  const skyGeo = scene.registerGeometry(skySphere.vertices, skySphere.indices)

  // ── Sky sphere ────────────────────────────────────────────────────────
  scene.add(
    new Mesh({
      geometry: skyGeo,
      position: [50, 80, 0],
      scale: [1500, 1500, 1500],
      color: [85 / 255, 221 / 255, 1],
      unlit: true,
    }),
  )

  // ── Load static GLB (megaxe + Eden) ───────────────────────────────────
  const megaxeColors = new Map<number, [number, number, number]>([
    [0, [1, 1, 1]],
    [1, [0, 0, 0]],
    [2, [0, 0.9, 0.8]],
  ])
  const { meshes: glbMeshes } = await loadGlb('/static-bundle.glb', '/draco-1.5.7/', megaxeColors)
  const megaxe = glbMeshes.find(m => m.name === 'megaxe')
  let axeMesh: Mesh | undefined
  let axeBloomMesh: Mesh | undefined
  let axeBaseGeo: number | undefined
  let axeGlowGeo: number | undefined
  if (megaxe) {
    if (megaxe.materialIndices) {
      // Split: non-bloom materials (0,1) and bloom material (2)
      const base = splitByMaterial(megaxe.vertices, megaxe.indices, megaxe.materialIndices, idx => idx !== 2)
      const glow = splitByMaterial(megaxe.vertices, megaxe.indices, megaxe.materialIndices, idx => idx === 2)
      if (base) {
        axeBaseGeo = scene.registerGeometry(base.vertices, base.indices)
        axeMesh = scene.add(
          new Mesh({
            geometry: axeBaseGeo,
            position: [0, 0.3, 0],
            rotation: [0, 0, Math.PI],
            scale: megaxe.scale,
            color: [1, 1, 1],
            outline: 1,
          }),
        )
      }
      if (glow) {
        axeGlowGeo = scene.registerGeometry(glow.vertices, glow.indices)
        axeBloomMesh = scene.add(
          new Mesh({
            geometry: axeGlowGeo,
            position: [0, 0.3, 0],
            rotation: [0, 0, Math.PI],
            scale: megaxe.scale,
            color: [1, 1, 1],
            bloom: 1.0,
            outline: 1,
          }),
        )
      }
    } else {
      axeBaseGeo = scene.registerGeometry(megaxe.vertices, megaxe.indices)
      axeMesh = scene.add(
        new Mesh({
          geometry: axeBaseGeo,
          position: [0, 0.3, 0],
          rotation: [0, 0, Math.PI],
          scale: megaxe.scale,
          color: [1, 1, 1],
          outline: 1,
        }),
      )
    }
  }

  // ── Load AO texture ──────────────────────────────────────────────────
  const aoTex = await loadKTX2('/city-ao.ktx2', '/basis-1.50/')
  const aoTexId = scene.registerTexture(aoTex.data, aoTex.width, aoTex.height)

  // ── Eden (merged into 1 draw call with AO map, bloom meshes kept separate) ──
  const edenBloomNames = new Set(['Eden_24'])
  const edenMeshes = glbMeshes.filter(m => m.name.startsWith('Eden_'))
  const edenNonBloom = edenMeshes.filter(m => !edenBloomNames.has(m.name))
  const edenBloom = edenMeshes.filter(m => edenBloomNames.has(m.name))
  const edenScale = edenMeshes[0]?.scale

  let edenWorldMesh: Mesh | undefined
  if (edenNonBloom.length > 0) {
    const merged = mergeGeometries(
      edenNonBloom.map(em => ({
        vertices: em.vertices,
        indices: em.indices,
        color: EDEN_COLORS[em.name] ?? [1, 1, 1],
        uvs: em.uvs,
      })),
    )
    if (merged.uvs) {
      const geo = scene.registerTexturedGeometry(merged.vertices, merged.indices, merged.uvs)
      edenWorldMesh = scene.add(
        new Mesh({
          geometry: geo,
          position: [0, 0, 0],
          scale: edenScale,
          color: [1, 1, 1],
          aoMap: aoTexId,
          outline: 2,
        }),
      )
    } else {
      const geo = scene.registerGeometry(merged.vertices, merged.indices)
      edenWorldMesh = scene.add(
        new Mesh({ geometry: geo, position: [0, 0, 0], scale: edenScale, color: [1, 1, 1], outline: 2 }),
      )
    }
    // Pre-build BVH for ground raycasting
    scene.buildBVH(edenWorldMesh.geometry)
  }

  for (const em of edenBloom) {
    const geo = scene.registerGeometry(em.vertices, em.indices)
    scene.add(
      new Mesh({
        geometry: geo,
        position: [0, 0, 0],
        scale: edenScale,
        color: EDEN_COLORS[em.name] ?? [1, 1, 1],
        bloom: 1.0,
        outline: 2,
      }),
    )
  }

  // ── Load player GLB ───────────────────────────────────────────────────
  const animCycleCallbacks: ((dt: number) => void)[] = []
  const playerResult = await loadGlb('/player-bundle.glb')
  const bodyMesh = playerResult.meshes.find(m => m.name === 'Body' && m.skinIndex !== undefined)

  let player: Mesh | undefined

  if (bodyMesh && bodyMesh.skinJoints && bodyMesh.skinWeights && bodyMesh.skinIndex !== undefined) {
    const geo = scene.registerSkinnedGeometry(
      bodyMesh.vertices,
      bodyMesh.indices,
      bodyMesh.skinJoints,
      bodyMesh.skinWeights,
    )

    const skin = playerResult.skins[bodyMesh.skinIndex]!
    const skeleton = createSkeleton(skin, playerResult.nodeTransforms)

    const animNames = ['Idle', 'Jump', 'Run', 'SlashRight']
    const animIndices: number[] = []
    for (const name of animNames) {
      const idx = playerResult.animations.findIndex(a => a.name.toLowerCase().includes(name.toLowerCase()))
      animIndices.push(idx >= 0 ? idx : 0)
    }

    const skinInst = createSkinInstance(skeleton, animIndices[0]!)
    updateSkinInstance(skinInst, playerResult.animations, 0)
    scene.skinInstances.push(skinInst)

    player = scene.add(
      new Mesh({
        geometry: geo,
        position: [42, 80, 0],
        scale: bodyMesh.scale,
        color: [1, 1, 1],
        skinned: true,
        skinInstanceId: 0,
        outline: 1,
      }),
    )

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

    // Attach megaxe to player's right hand bone
    if (axeMesh) scene.attachToBone(axeMesh, player, 'Hand.R')
    if (axeBloomMesh) scene.attachToBone(axeBloomMesh, player, 'Hand.R')

    // ── NPC grid ──────────────────────────────────────────────────────
    if (edenWorldMesh) {
      const NPC_COLS = 15
      const NPC_ROWS = 14
      const ANIM_CYCLE_LENGTH = ANIM_INTERVAL * animIndices.length // 4.0s full cycle
      const npcRayHit = createRaycastHit()
      const npcEdenFilter = [edenWorldMesh] as readonly Mesh[]
      let npcCount = 0

      for (let row = 0; row < NPC_ROWS; row++) {
        for (let col = 0; col < NPC_COLS; col++) {
          const x = (col / (NPC_COLS - 1)) * 150
          const y = (row / (NPC_ROWS - 1)) * 150

          // Check for ground at this position
          const hit = scene.raycast(x, y, 200, 0, 0, -1, npcRayHit, npcEdenFilter)
          if (!hit) continue

          const groundZ = npcRayHit.pointZ
          const skinInstId = scene.skinInstances.length

          // Stagger animation: spread each NPC evenly across the full cycle
          const timeOffset = (npcCount / (NPC_COLS * NPC_ROWS)) * ANIM_CYCLE_LENGTH
          const initialAnimIdx = Math.floor(timeOffset / ANIM_INTERVAL) % animIndices.length

          const npcSkinInst = createSkinInstance(skeleton, animIndices[initialAnimIdx]!)
          updateSkinInstance(npcSkinInst, playerResult.animations, 0)
          scene.skinInstances.push(npcSkinInst)

          // Body
          const npcBody = scene.add(
            new Mesh({
              geometry: geo,
              position: [x, y, groundZ],
              scale: bodyMesh.scale,
              color: [1, 1, 1],
              skinned: true,
              skinInstanceId: skinInstId,
            }),
          )

          // Axe (base)
          if (axeBaseGeo !== undefined) {
            const npcAxe = scene.add(
              new Mesh({
                geometry: axeBaseGeo,
                position: [0, 0.3, 0],
                rotation: [0, 0, Math.PI],
                scale: megaxe!.scale,
                color: [1, 1, 1],
              }),
            )
            scene.attachToBone(npcAxe, npcBody, 'Hand.R')
          }

          // Axe (bloom)
          if (axeGlowGeo !== undefined) {
            const npcAxeGlow = scene.add(
              new Mesh({
                geometry: axeGlowGeo,
                position: [0, 0.3, 0],
                rotation: [0, 0, Math.PI],
                scale: megaxe!.scale,
                color: [1, 1, 1],
                bloom: 1.0,
              }),
            )
            scene.attachToBone(npcAxeGlow, npcBody, 'Hand.R')
          }

          // Animation cycle callback (staggered timer)
          let npcAnimIndex = initialAnimIdx
          let npcAnimTimer = timeOffset % ANIM_INTERVAL
          animCycleCallbacks.push((dt: number) => {
            npcAnimTimer += dt
            if (npcAnimTimer >= ANIM_INTERVAL) {
              npcAnimTimer -= ANIM_INTERVAL
              npcAnimIndex = (npcAnimIndex + 1) % animIndices.length
              transitionTo(npcSkinInst, animIndices[npcAnimIndex]!, BLEND_DURATION)
            }
          })

          npcCount++
        }
      }
    }
  }

  // ── Transparent cubes ─────────────────────────────────────────────────
  const cubeColors: [number, number, number][] = [
    [1, 0.2, 0.2],
    [0.2, 0.8, 0.2],
    [0.3, 0.4, 1],
    [1, 0.8, 0.1],
    [0.9, 0.3, 0.9],
  ]
  const cubes: Mesh[] = []
  for (let i = 0; i < cubeColors.length; i++) {
    const angle = (i / cubeColors.length) * Math.PI * 2
    const cube = scene.add(
      new Mesh({
        geometry: cubeGeo,
        position: [50 + Math.cos(angle) * 4, 80 + Math.sin(angle) * 4, 3],
        color: cubeColors[i]!,
        alpha: 0.5,
        outline: 1,
      }),
    )
    cubes.push(cube)
  }

  // ── HTML overlay labels ──────────────────────────────────────────────
  const labelStyle =
    'color:#fff;font:bold 14px/1 sans-serif;background:rgba(0,0,0,0.6);padding:4px 8px;border-radius:4px;white-space:nowrap'

  let overlay = new HtmlOverlay(currentCanvas)

  const cubeLabel = document.createElement('div')
  cubeLabel.textContent = 'Cubes'
  cubeLabel.style.cssText = labelStyle
  overlay.add(new HtmlElement({ position: [50, 80, 6], element: cubeLabel, distanceFactor: 20 }))

  if (player) {
    const playerLabel = document.createElement('div')
    playerLabel.textContent = 'Player'
    playerLabel.style.cssText = labelStyle
    overlay.add(new HtmlElement({ mesh: player, offset: [0, 0, 4], element: playerLabel, distanceFactor: 20 }))
  }

  // ── Stats overlay ─────────────────────────────────────────────────────
  const stats = document.createElement('div')
  stats.style.cssText =
    'position:fixed;top:8px;left:8px;color:#fff;font:14px/1.4 monospace;background:rgba(0,0,0,0.5);padding:4px 8px;border-radius:4px;pointer-events:none'
  document.body.appendChild(stats)
  let frames = 0
  let fpsTime = performance.now()
  setInterval(() => {
    const now = performance.now()
    const fps = Math.round((frames * 1000) / (now - fpsTime))
    frames = 0
    fpsTime = now
    stats.textContent = `FPS: ${fps}  Draw calls: ${scene.drawCalls}  ${scene.backendType}`
  }, 500)

  // ── Backend toggle ────────────────────────────────────────────────────
  let switching = false
  let resizeObs: ResizeObserver

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
        scene.resize(w, h)
      }
    })
    resizeObs.observe(currentCanvas)
  }

  async function switchBackend(type: BackendType) {
    if (scene.backendType === type) return
    switching = true

    const { theta, phi, radius, targetX, targetY, targetZ } = orbit
    orbit.destroy()
    overlay.destroy()
    resizeObs.disconnect()

    const newCanvas = document.createElement('canvas')
    newCanvas.style.cssText = currentCanvas.style.cssText
    newCanvas.width = currentCanvas.width
    newCanvas.height = currentCanvas.height
    currentCanvas.parentNode!.replaceChild(newCanvas, currentCanvas)
    currentCanvas = newCanvas

    await scene.switchBackend(currentCanvas, type)

    orbit = new OrbitControls(currentCanvas)
    orbit.theta = theta
    orbit.phi = phi
    orbit.radius = radius
    orbit.targetX = targetX
    orbit.targetY = targetY
    orbit.targetZ = targetZ
    orbit.updateEye()

    overlay = new HtmlOverlay(currentCanvas)
    overlay.add(new HtmlElement({ position: [50, 80, 6], element: cubeLabel, distanceFactor: 20 }))
    if (player) {
      const playerLabel2 = document.createElement('div')
      playerLabel2.textContent = 'Player'
      playerLabel2.style.cssText = labelStyle
      overlay.add(new HtmlElement({ mesh: player, offset: [0, 0, 4], element: playerLabel2, distanceFactor: 20 }))
    }

    observeResize()
    switching = false
  }

  if (webgpuAvailable) {
    const toggle = document.createElement('div')
    toggle.style.cssText =
      'position:fixed;top:8px;right:8px;color:#fff;font:14px/1.4 monospace;background:rgba(0,0,0,0.5);padding:4px 8px;border-radius:4px;display:flex;align-items:center;gap:6px;user-select:none'
    const checkbox = document.createElement('input')
    checkbox.type = 'checkbox'
    checkbox.id = 'backend-toggle'
    checkbox.checked = scene.backendType === 'webgpu'
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
        checkbox.checked = scene.backendType === 'webgpu'
      }
      checkbox.disabled = false
    })
  }

  observeResize()

  // ── Raycast receiver (reused every frame — no allocations) ──────────
  const rayHit = createRaycastHit()
  const edenFilter = edenWorldMesh ? ([edenWorldMesh] as readonly Mesh[]) : undefined

  // ── Game loop ─────────────────────────────────────────────────────────
  let lastTime = performance.now()

  function loop(now: number) {
    if (switching) {
      requestAnimationFrame(loop)
      return
    }
    const dt = Math.min((now - lastTime) / 1000, 0.1)
    lastTime = now

    // Input
    if (player) {
      if (keys.has('KeyW')) player.position[1]! += MOVE_SPEED * dt
      if (keys.has('KeyS')) player.position[1]! -= MOVE_SPEED * dt
      if (keys.has('KeyA')) player.position[0]! -= MOVE_SPEED * dt
      if (keys.has('KeyD')) player.position[0]! += MOVE_SPEED * dt

      // Snap player to ground via downward raycast against Eden world
      if (edenFilter) {
        const hit = scene.raycast(
          player.position[0]!,
          player.position[1]!,
          player.position[2]! + 50,
          0,
          0,
          -1,
          rayHit,
          edenFilter,
        )
        if (hit) {
          player.position[2] = rayHit.pointZ
        }
      }
    }

    // Rotate cubes
    for (const cube of cubes) cube.rotation[2]! += dt

    // Animations
    for (const cb of animCycleCallbacks) cb(dt)
    for (const inst of scene.skinInstances) updateSkinInstance(inst, playerResult.animations, dt)

    // Camera — feed orbit controls into scene camera
    scene.camera.eye.set(orbit.eye)
    scene.camera.target.set(orbit.target)

    // Render
    scene.render()
    overlay.update(scene)
    frames++
    requestAnimationFrame(loop)
  }

  requestAnimationFrame(loop)
}
