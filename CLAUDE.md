# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- `bun run dev` — Start Next.js dev server
- `bun run build` — Production build
- `bunx tsc --noEmit` — Type-check (no output files)
- `bun test` — Run tests (bun:test, not jest/vitest)
- `bun install` — Install dependencies (not npm/yarn/pnpm)

## Tooling

Default to **Bun** instead of Node.js for all commands (`bun <file>`, `bun install`, `bunx`, `bun run <script>`). Bun auto-loads `.env` — don't use dotenv.

TypeScript is strict with `noUncheckedIndexedAccess` enabled — all indexed TypedArray accesses need `!` assertions. WebGPU types come from `@webgpu/types`.

All `src/` imports use explicit `.ts` extensions (`from './scene.ts'`), enabled by `allowImportingTsExtensions` in tsconfig.

When passing TypedArrays to `device.queue.writeBuffer`, cast the buffer: `writeBuffer(buf, 0, arr.buffer as ArrayBuffer, arr.byteOffset, arr.byteLength)` — strict TS rejects `ArrayBufferLike` where `ArrayBuffer` is expected.

## Architecture

**Mana Engine** is a minimal WebGPU/WebGL rendering engine using a **right-handed Z-up coordinate system** (Blender convention): +X right, +Y forward, +Z up. glTF assets are exported from Blender with the "Y up" option unchecked to preserve Z-up coordinates — no axis conversion needed at load time.

Next.js serves a single page (`pages/index.tsx`) that mounts a fullscreen `<canvas>` and dynamically imports the engine (`src/main.ts`) to avoid SSR issues with browser APIs.

The engine is pure TypeScript with no React dependency. All engine code lives under `src/engine/`:

- **`scene.ts`** — `Scene`, `Mesh`, `Camera` classes and `createScene()` factory. The Scene manages the full lifecycle: geometry registration, mesh management, camera, lighting, shadows, bloom, outline, raycasting, and rendering.
- **`renderer.ts`** — WebGPU renderer. Accepts `RenderScene` interface (internal contract between Scene and renderer). MRT for bloom/outline.
- **`webgl-renderer.ts`** — WebGL2 fallback renderer, same `RenderScene` interface with MRT support.
- **`gpu.ts`** — `createRenderer()` factory (picks WebGPU or WebGL).
- **`math.ts`** — Zero-allocation vec3/mat4 math (all write to pre-allocated Float32Arrays).
- **`geometry.ts`** — Cube/sphere primitives, `mergeGeometries()`.
- **`gltf.ts`** — GLB/glTF loader with Draco support.
- **`skin.ts`** — GPU skinning: skeleton, skin instances, animation sampling with crossfade blending.
- **`bvh.ts`** — SAH-binned BVH acceleration structure for raycasting. Flat-array node storage, stack-based traversal, Möller–Trumbore intersection.
- **`orbit-controls.ts`** — Spherical orbit camera controls.
- **`html-overlay.ts`** — `HtmlOverlay` / `HtmlElement` for projecting DOM elements to 3D world positions with distance-based scaling.
- **`ktx2.ts`** — KTX2/Basis Universal texture loader, transcodes to RGBA8.
- **`shaders.ts`** / **`webgl-shaders.ts`** — WGSL and GLSL shader sources.
- **`index.ts`** — Barrel export.
- **`src/main.ts`** — Demo app that consumes the engine.

### Consumer API (Three.js-like):

```ts
const scene = await createScene(canvas)

const geo = scene.registerGeometry(vertices, indices)
const mesh = scene.add(new Mesh({ geometry: geo, position: [0, 0, 0], color: [1, 0, 0] }))

scene.camera.fov = Math.PI / 3
scene.camera.eye.set([0, -10, 5])
scene.camera.target.set([0, 0, 0])

scene.setDirectionalLight([-1, -1, -2], [0.5, 0.5, 0.5])
scene.setAmbientLight([0.8, 0.8, 0.8])

scene.shadow.enabled = true
scene.bloom.enabled = true
scene.outline.enabled = true

function loop() {
  mesh.rotation[2] += 0.01
  scene.render()
  requestAnimationFrame(loop)
}
requestAnimationFrame(loop)
```

### Raycasting (BVH-accelerated):

```ts
import { createRaycastHit } from './engine/index.ts'

const hit = createRaycastHit() // reusable receiver — zero allocation per frame
scene.buildBVH(groundMesh.geometry) // optional: pre-build BVH (lazy-built on first raycast otherwise)

// Cast ray downward from (x, y, z+50), filter to ground mesh only
if (scene.raycast(x, y, z + 50, 0, 0, -1, hit, [groundMesh])) {
  player.position[2] = hit.pointZ // snap to surface
  // hit.normalX/Y/Z, hit.distance, hit.faceIndex, hit.mesh also available
}
```

### Bone attachment:

```ts
scene.attachToBone(weaponMesh, characterMesh, 'Hand.R')
scene.detachFromBone(weaponMesh)
```

### Bloom & Outline:

```ts
scene.bloom.enabled = true
scene.bloom.intensity = 1.0
scene.bloom.threshold = 0.0
scene.bloom.radius = 1.0
scene.bloom.whiten = 0.0

mesh.bloom = 1.5 // emission value (0 = off)

scene.outline.enabled = true
scene.outline.thickness = 3
scene.outline.color = [1, 0.5, 0]
scene.outline.distanceFactor = 0

mesh.outline = 1 // outline group (0 = off)
```

### HTML Overlay:

```ts
const overlay = new HtmlOverlay(containerDiv)
const label = overlay.add(new HtmlElement({ position: [0, 0, 5], element: myDiv, distanceFactor: 1 }))
label.mesh = someMesh // track a mesh instead of fixed position
overlay.update(scene) // call each frame to project 3D → 2D
```

### Internal performance architecture:

Internally, `Scene` uses a **Structure-of-Arrays** layout — parallel TypedArrays for positions, scales, world matrices, colors, etc. On each `scene.render()` call, mesh object data is synced to these dense arrays, world matrices are computed via `m4FromTRS`, and the result is passed to the renderer as a zero-copy `RenderScene` interface. This gives Three.js-like ergonomics with high performance.

### Key design patterns:

- **Zero-allocation math** (`engine/math.ts`): All vec3/mat4 functions write to a pre-allocated `Float32Array` at a given offset. No temp objects, no GC pressure in the hot path. Animation math: `v3Lerp`, `quatSlerp`, `m4FromQuatTRS`. Matrix utilities: `m4Invert`, `m4TransformPoint`, `m4TransformDirection`, `m4TransformNormal`.
- **Sync-to-arrays** (`engine/scene.ts`): Mesh objects own small Float32Arrays for position/rotation/scale/color. Before each render, a tight loop copies these into SoA arrays and computes world matrices. The renderer iterates dense arrays, not scattered objects.
- **Auto geometry management** (`engine/scene.ts`): `scene.registerGeometry()` returns an auto-assigned ID, tracks registrations internally, and re-registers on backend switch — consumers never manage IDs.
- **Dynamic uniform buffer** (`engine/renderer.ts`): Per-entity model data packed into 256-byte aligned slots in a single GPU buffer, bound via dynamic offsets.
- **Frustum culling** (`engine/renderer.ts` + `engine/math.ts`): Gribb-Hartmann plane extraction from the VP matrix, bounding sphere test per entity.
- **Shadow auto-computation** (`engine/scene.ts`): When `scene.shadow.enabled = true`, the Scene computes the light VP matrix from the directional light direction + shadow config (target, distance, extent, near/far).
- **OrbitControls** (`engine/orbit-controls.ts`): Spherical coordinates around a target point with Z as vertical. Consumer feeds `orbit.eye`/`orbit.target` into `scene.camera.eye`/`scene.camera.target`.
- **GLB/glTF loader** (`engine/gltf.ts`): Loads `.glb` with Draco mesh compression. Draco WASM loaded from `public/draco-1.5.7/`. Returns `GlbResult { meshes, skins, animations, nodeTransforms }`. Supports optional material color mapping.
- **GPU Skinning** (`engine/skin.ts` + `engine/renderer.ts`): Separate skinned pipeline with joint matrices storage buffer. `updateSkinInstance()` samples keyframes and computes joint matrices. `transitionTo()` crossfade-blends between animation clips. `findBoneNodeIndex()` locates bones by name.
- **BVH Raycasting** (`engine/bvh.ts` + `engine/scene.ts`): SAH-binned BVH (12 bins, max 4 tris/leaf) with flat-array node storage (8 floats/node) and stack-based traversal. `scene.raycast()` transforms rays to local space, handles world matrices via `m4Invert`, returns closest hit into a reusable `RaycastHit` receiver.
- **Bone Attachment** (`engine/scene.ts`): `scene.attachToBone(mesh, parentMesh, boneName)` parents a mesh to a skeleton bone. World matrix is computed from the bone's global matrix each frame. Used for weapons/props on animated characters.
- **Bloom** (`engine/renderer.ts` + `engine/webgl-renderer.ts`): 5-mip downsample/upsample chain via MRT. Meshes with `bloom > 0` emit into a separate render target, which is blurred and composited.
- **Outline** (`engine/renderer.ts` + `engine/webgl-renderer.ts`): MRT-based outline rendering. Meshes with `outline > 0` write a group ID; edges between groups are detected and drawn.
- **HTML Overlay** (`engine/html-overlay.ts`): Projects DOM elements to 3D world positions using the scene's view/projection matrices. Supports mesh tracking and distance-based scaling.
- **KTX2 Textures** (`engine/ktx2.ts`): Loads KTX2/Basis Universal textures, transcodes to RGBA8 via the Basis transcoder WASM.

### Mesh properties:

- `position: Float32Array` — [x, y, z]
- `rotation: Float32Array` — Euler ZXY [rx, ry, rz]
- `scale: Float32Array` — [sx, sy, sz]
- `color: Float32Array` — [r, g, b]
- `alpha: number` — transparency (0-1), enables transparent pipeline when < 1
- `visible: boolean` — visibility toggle
- `unlit: boolean` — skip lighting, render flat color
- `skinned: boolean` / `skinInstanceId: number` — skeletal animation
- `bloom: number` — emission value (0 = off, >0 = emissive glow)
- `outline: number` — outline group (0 = off, >0 = group ID)
- `aoMap: number` — texture ID for AO map (-1 = none)

### Textured geometry:

```ts
const texId = scene.registerTexture(rgbaData, width, height)
const geo = scene.registerTexturedGeometry(vertices, indices, uvs)
const mesh = scene.add(new Mesh({ geometry: geo, aoMap: texId }))
```

### Scene lifecycle:

```ts
scene.resize(width, height) // resize canvas/backbuffer
await scene.switchBackend(canvas, 'webgl') // switch renderer at runtime
scene.destroy() // clean up GPU resources
```

### Shader bind group layout (WGSL in `engine/shaders.ts`):

- Group 0: Camera (view + projection mat4x4f)
- Group 1: Model with dynamic offset (world mat4x4f + color vec4f)
- Group 2: Lighting (direction, dirColor, ambientColor — all vec4f for 16-byte alignment)
- Group 3: Joint matrices storage buffer with dynamic offset (skinned pipeline only — `array<mat4x4f>`, 128 joints per slot)

### Geometry format:

Interleaved vertex data: `[px, py, pz, nx, ny, nz, cr, cg, cb]` per vertex (stride 36 bytes). Vertex colors are multiplied with the per-entity uniform color in the fragment shader — set vertex colors to white `[1,1,1]` for uniform-only coloring, or set uniform color to white for vertex-color-only coloring. GLB meshes loaded via `loadGlb()` support both uint16 and uint32 index buffers. `mergeGeometries()` merges multiple primitives into a single geometry, baking per-primitive colors into vertex RGB.

### Renderer limits:

| Constant             | Value | Purpose                          |
| -------------------- | ----- | -------------------------------- |
| MAX_JOINTS           | 128   | Max bones per skeleton           |
| MAX_SKINNED_ENTITIES | 256   | Max concurrent skinned meshes    |
| MODEL_SLOT_SIZE      | 256   | Uniform buffer alignment (bytes) |
| SHADOW_MAP_SIZE      | 2048  | Shadow texture resolution        |
| MSAA_SAMPLES         | 4     | Anti-aliasing samples            |
| BLOOM_MIPS           | 5     | Bloom pyramid levels             |
| SAH_BINS             | 12    | BVH SAH bin count                |
| MAX_LEAF_TRIS        | 4     | BVH max triangles per leaf       |
