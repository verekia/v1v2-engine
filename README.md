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
- GPU skinning with animation blending
- GLB/glTF loading with Draco mesh compression
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

### Rendering

```ts
scene.render() // sync + draw in one call
```

### Backend switching

```ts
await scene.switchBackend(newCanvas, 'webgl')
```

## Coordinate system

Right-handed Z-up (Blender convention): +X right, +Y forward, +Z up.

## Development

```bash
bun install
bun run dev
```
