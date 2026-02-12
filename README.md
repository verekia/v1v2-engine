# V1V2 Engine

Engine in development for [Mana Blade](https://manablade.com/).

A minimal, high-performance WebGPU/WebGL rendering engine with a Three.js-like API.

## Quickstart

```ts
import { createScene, Mesh, OrbitControls, cubeVertices, cubeIndices } from '@v1v2/engine'

const canvas = document.querySelector('canvas')!
const scene = await createScene(canvas)

// Register geometry
const cubeGeo = scene.registerGeometry(cubeVertices, cubeIndices)

// Add a mesh
const cube = scene.add(
  new Mesh({
    geometry: cubeGeo,
    position: [0, 0, 0],
    color: [1, 0.3, 0.3],
  }),
)

// Camera
const orbit = new OrbitControls(canvas)
scene.camera.fov = Math.PI / 3
scene.camera.near = 0.1
scene.camera.far = 1000

// Lighting
scene.setDirectionalLight([-1, -1, -2], [0.5, 0.5, 0.5])
scene.setAmbientLight([0.8, 0.8, 0.8])

// Shadows
scene.shadow.enabled = true
scene.shadow.target.set([0, 0, 0])

// Render loop
function loop() {
  cube.rotation[2]! += 0.01
  scene.camera.eye.set(orbit.eye)
  scene.camera.target.set(orbit.target)
  scene.render()
  requestAnimationFrame(loop)
}
requestAnimationFrame(loop)
```

## Features

- WebGPU with automatic WebGL2 fallback
- Structure-of-Arrays internals for cache-friendly rendering
- Frustum culling with bounding spheres
- Shadow mapping (directional light)
- Bloom post-processing (5-mip downsample/upsample chain)
- Outline rendering (MRT-based, configurable thickness/color)
- GPU skinning with animation crossfade blending
- Bone attachment system (parent meshes to skeleton bones)
- BVH-accelerated raycasting (SAH-binned, zero allocation in hot path)
- GLB/glTF loading with Draco mesh compression
- KTX2/Basis Universal texture loading
- HTML overlay (project DOM elements to 3D world positions)
- Orbit controls (pan, rotate, zoom)
- Transparent object sorting (back-to-front)
- Zero per-frame allocations in the hot path

## API

### Scene

```ts
const scene = await createScene(canvas, { maxEntities: 10_000, backend: 'webgpu' })
```

### Geometry

```ts
const geo = scene.registerGeometry(vertices, indices)
const skinnedGeo = scene.registerSkinnedGeometry(vertices, indices, joints, weights)
const texturedGeo = scene.registerTexturedGeometry(vertices, indices, uvs)
```

### Built-in Primitives

```ts
import { createBoxGeometry, createSphereGeometry, mergeGeometries } from '@v1v2/engine'

const box = createBoxGeometry(1, 1, 1)
const sphere = createSphereGeometry(0.5, 32, 16)
const merged = mergeGeometries([
  { vertices: box.vertices, indices: box.indices, color: [1, 0, 0] },
  { vertices: sphere.vertices, indices: sphere.indices, color: [0, 0, 1] },
])
```

### Meshes

```ts
const mesh = scene.add(
  new Mesh({
    geometry: geo,
    position: [0, 0, 0],
    rotation: [0, 0, 0],
    scale: [1, 1, 1],
    color: [1, 1, 1],
    alpha: 1,
    unlit: false,
  }),
)

mesh.position[0]! += 1 // direct mutation
mesh.visible = false // hide
mesh.bloom = 1.5 // emissive glow (0 = off)
mesh.outline = 1 // outline group (0 = off)
scene.remove(mesh) // remove from scene
```

### Camera

```ts
scene.camera.fov = Math.PI / 3
scene.camera.near = 0.1
scene.camera.far = 5000
scene.camera.eye.set([0, -10, 5])
scene.camera.target.set([0, 0, 0])
scene.camera.up.set([0, 0, 1])
```

### Lighting

```ts
scene.setDirectionalLight([dx, dy, dz], [r, g, b])
scene.setAmbientLight([r, g, b])
```

### Shadows

```ts
scene.shadow.enabled = true
scene.shadow.target.set([0, 0, 0])
scene.shadow.distance = 400
scene.shadow.extent = 150
scene.shadow.near = 1
scene.shadow.far = 800
scene.shadow.bias = 0.0001
```

### Bloom

```ts
scene.bloom.enabled = true
scene.bloom.intensity = 1.0
scene.bloom.threshold = 0.0
scene.bloom.radius = 1.0
scene.bloom.whiten = 0.0
```

### Outline

```ts
scene.outline.enabled = true
scene.outline.thickness = 3
scene.outline.color = [1, 0.5, 0]
scene.outline.distanceFactor = 0 // distance-based scaling
```

### Raycasting

```ts
import { createRaycastHit } from '@v1v2/engine'

const hit = createRaycastHit() // reusable receiver — no per-frame allocation
scene.buildBVH(groundMesh.geometry) // optional pre-build (lazy-built on first raycast otherwise)

if (scene.raycast(x, y, z + 50, 0, 0, -1, hit, [groundMesh])) {
  player.position[2] = hit.pointZ
  // hit.distance, hit.normalX/Y/Z, hit.faceIndex, hit.mesh
}
```

### GLB/glTF Loading

```ts
import { loadGlb } from '@v1v2/engine'

const glb = await loadGlb('/model.glb', '/draco-1.5.7/')
// glb.meshes, glb.skins, glb.animations, glb.nodeTransforms
```

### Skinning & Animation

```ts
import { createSkeleton, createSkinInstance, updateSkinInstance, transitionTo } from '@v1v2/engine'

const skeleton = createSkeleton(glb.skins[0], glb.nodeTransforms)
const skin = createSkinInstance(skeleton, 0) // start with clip 0
scene.skinInstances.push(skin)

// In render loop:
updateSkinInstance(skin, glb.animations, dt)

// Crossfade to a new clip:
transitionTo(skin, 2, 0.2) // blend to clip 2 over 0.2s
```

### Bone Attachment

```ts
scene.attachToBone(weaponMesh, characterMesh, 'Hand.R')
scene.detachFromBone(weaponMesh)
```

### Textures

```ts
import { loadKTX2 } from '@v1v2/engine'

const tex = await loadKTX2('/texture.ktx2', '/basis/')
const texId = scene.registerTexture(tex.data, tex.width, tex.height)
const geo = scene.registerTexturedGeometry(vertices, indices, uvs)
const mesh = scene.add(new Mesh({ geometry: geo, aoMap: texId }))
```

### HTML Overlay

```ts
import { HtmlOverlay, HtmlElement } from '@v1v2/engine'

const overlay = new HtmlOverlay(containerDiv)
const label = overlay.add(new HtmlElement({ position: [0, 0, 5], element: myDiv, distanceFactor: 1 }))
label.mesh = someMesh // track a mesh

// In render loop:
overlay.update(scene)
```

### Rendering

```ts
scene.render() // sync + draw in one call
scene.resize(width, height) // resize canvas/backbuffer
scene.drawCalls // frame draw call count
```

### Backend switching

```ts
await scene.switchBackend(newCanvas, 'webgl')
```

### Cleanup

```ts
scene.destroy()
```

## Coordinate system

Right-handed Z-up (Blender convention): +X right, +Y forward, +Z up.

## Development

```bash
bun install
bun run dev
```
