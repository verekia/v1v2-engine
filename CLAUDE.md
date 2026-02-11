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

- **`scene.ts`** — `Scene`, `Mesh`, `Camera` classes and `createScene()` factory. The Scene manages the full lifecycle: geometry registration, mesh management, camera, lighting, shadows, and rendering.
- **`renderer.ts`** — WebGPU renderer. Accepts `RenderScene` interface (internal contract between Scene and renderer).
- **`webgl-renderer.ts`** — WebGL2 fallback renderer, same `RenderScene` interface.
- **`gpu.ts`** — `createRenderer()` factory (picks WebGPU or WebGL).
- **`math.ts`** — Zero-allocation vec3/mat4 math (all write to pre-allocated Float32Arrays).
- **`geometry.ts`** — Cube/sphere primitives, `mergeGeometries()`.
- **`gltf.ts`** — GLB/glTF loader with Draco support.
- **`skin.ts`** — GPU skinning: skeleton, skin instances, animation sampling.
- **`orbit-controls.ts`** — Spherical orbit camera controls.
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

function loop() {
  mesh.rotation[2] += 0.01
  scene.render()
  requestAnimationFrame(loop)
}
requestAnimationFrame(loop)
```

### Internal performance architecture:

Internally, `Scene` uses a **Structure-of-Arrays** layout — parallel TypedArrays for positions, scales, world matrices, colors, etc. On each `scene.render()` call, mesh object data is synced to these dense arrays, world matrices are computed via `m4FromTRS`, and the result is passed to the renderer as a zero-copy `RenderScene` interface. This gives Three.js-like ergonomics with high performance.

### Key design patterns:

- **Zero-allocation math** (`engine/math.ts`): All vec3/mat4 functions write to a pre-allocated `Float32Array` at a given offset. No temp objects, no GC pressure in the hot path. Animation math: `v3Lerp`, `quatSlerp`, `m4FromQuatTRS`.
- **Sync-to-arrays** (`engine/scene.ts`): Mesh objects own small Float32Arrays for position/rotation/scale/color. Before each render, a tight loop copies these into SoA arrays and computes world matrices. The renderer iterates dense arrays, not scattered objects.
- **Auto geometry management** (`engine/scene.ts`): `scene.registerGeometry()` returns an auto-assigned ID, tracks registrations internally, and re-registers on backend switch — consumers never manage IDs.
- **Dynamic uniform buffer** (`engine/renderer.ts`): Per-entity model data packed into 256-byte aligned slots in a single GPU buffer, bound via dynamic offsets.
- **Frustum culling** (`engine/renderer.ts` + `engine/math.ts`): Gribb-Hartmann plane extraction from the VP matrix, bounding sphere test per entity.
- **Shadow auto-computation** (`engine/scene.ts`): When `scene.shadow.enabled = true`, the Scene computes the light VP matrix from the directional light direction + shadow config (target, distance, extent, near/far).
- **OrbitControls** (`engine/orbit-controls.ts`): Spherical coordinates around a target point with Z as vertical. Consumer feeds `orbit.eye`/`orbit.target` into `scene.camera.eye`/`scene.camera.target`.
- **GLB/glTF loader** (`engine/gltf.ts`): Loads `.glb` with Draco mesh compression. Draco WASM loaded from `public/draco-1.5.7/`. Returns `GlbResult { meshes, skins, animations, nodeTransforms }`.
- **GPU Skinning** (`engine/skin.ts` + `engine/renderer.ts`): Separate skinned pipeline with joint matrices storage buffer. `updateSkinInstance()` samples keyframes and computes joint matrices.

### Shader bind group layout (WGSL in `engine/shaders.ts`):

- Group 0: Camera (view + projection mat4x4f)
- Group 1: Model with dynamic offset (world mat4x4f + color vec4f)
- Group 2: Lighting (direction, dirColor, ambientColor — all vec4f for 16-byte alignment)
- Group 3: Joint matrices storage buffer with dynamic offset (skinned pipeline only — `array<mat4x4f>`, 128 joints per slot)

### Geometry format:

Interleaved vertex data: `[px, py, pz, nx, ny, nz, cr, cg, cb]` per vertex (stride 36 bytes). Vertex colors are multiplied with the per-entity uniform color in the fragment shader — set vertex colors to white `[1,1,1]` for uniform-only coloring, or set uniform color to white for vertex-color-only coloring. GLB meshes loaded via `loadGlb()` support both uint16 and uint32 index buffers. `mergeGeometries()` merges multiple primitives into a single geometry, baking per-primitive colors into vertex RGB.
