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

All `src/` imports use explicit `.ts` extensions (`from './ecs.ts'`), enabled by `allowImportingTsExtensions` in tsconfig.

When passing TypedArrays to `device.queue.writeBuffer`, cast the buffer: `writeBuffer(buf, 0, arr.buffer as ArrayBuffer, arr.byteOffset, arr.byteLength)` — strict TS rejects `ArrayBufferLike` where `ArrayBuffer` is expected.

## Architecture

**Mana Engine** is a minimal WebGPU ECS game engine. Next.js serves a single page (`pages/index.tsx`) that mounts a fullscreen `<canvas>` and dynamically imports the engine (`src/main.ts`) to avoid SSR issues with browser APIs.

The engine is pure TypeScript with no React dependency. Engine code is split into two independent libraries under `src/`:

- **`src/engine/`** — Graphics library (like three.js). Zero dependency on ECS. Contains: `math.ts`, `geometry.ts`, `shaders.ts`, `gpu.ts` (WebGPU init), `renderer.ts` (decoupled via `RenderScene` interface), `orbit-controls.ts`, `gltf.ts`, `skin.ts`, `index.ts` (barrel export).
- **`src/ecs/`** — Optional ECS library. Zero dependency on engine. Contains: `world.ts` (renamed from `ecs.ts`), `input.ts`, `index.ts` (barrel export).
- **`src/main.ts`** — Demo app that consumes both libraries and bridges them via the `RenderScene` interface.

The **`RenderScene` bridge pattern** (`renderer.ts`): The renderer accepts a `RenderScene` interface with pre-computed `renderMask`/`skinnedMask` Uint8Arrays and direct references to the World's typed arrays (zero-copy). `main.ts` builds the `RenderScene` each frame from ECS bitmask checks, keeping the renderer completely ECS-agnostic. `gpu.ts` exports the standalone `initGPU()` factory extracted from the renderer.

### Per-frame system execution order (game loop in `main.ts`):

1. **Input system** — WASD keyboard state applied to entities with `TRANSFORM | INPUT_RECEIVER`
2. **Animation system** — `updateSkinInstance()` for each `SkinInstance`: advance time, sample keyframes, compute joint matrices
3. **Transform system** — Compute `worldMatrix` from position/rotation/scale via `m4FromTRS`
4. **Camera system** — Compute view matrix from OrbitControls' `eye`/`target` Float32Arrays, and projection matrix
5. **Render system** — Frustum cull, upload uniforms, issue draw calls (static pass then skinned pass)

### Key design patterns:

- **Zero-allocation math** (`engine/math.ts`): All vec3/mat4 functions write to a pre-allocated `Float32Array` at a given offset. No temp objects, no GC pressure in the hot path. Includes `quatToEulerYXZ` for glTF quaternion → Euler conversion matching the Y×X×Z rotation order. Animation math: `v3Lerp`, `quatSlerp`, `m4FromQuatTRS` (builds mat4 from quaternion + TRS).
- **Structure-of-Arrays ECS** (`ecs/world.ts`): Component data stored as parallel TypedArrays (e.g., `positions: Float32Array[MAX*3]`), not per-entity objects. Component presence tracked via bitmasks (`TRANSFORM = 1 << 0`, `SKINNED = 1 << 4`, etc.) with inline iteration — no query abstraction. Skinned entities store a `skinInstanceIds` index into an external `SkinInstance[]` array.
- **Dynamic uniform buffer** (`engine/renderer.ts`): Per-entity model data packed into 256-byte aligned slots in a single GPU buffer, bound via dynamic offsets to avoid per-entity bind group allocation.
- **Frustum culling** (`engine/renderer.ts` + `engine/math.ts`): Gribb-Hartmann plane extraction from the VP matrix, bounding sphere test per entity before upload and draw.
- **OrbitControls** (`engine/orbit-controls.ts`): Spherical coordinates (theta/phi/radius) around a target point. Outputs `eye` and `target` Float32Arrays consumed by the camera system's `m4LookAt`. Left-drag orbits, right-drag pans, scroll zooms.
- **GLB/glTF loader** (`engine/gltf.ts`): Loads `.glb` files with Draco mesh compression (`KHR_draco_mesh_compression`). Draco WASM decoder loaded at runtime from `public/draco-1.5.7/` via `<script>` tag (singleton, no npm package). Supports reading custom `_MATERIALINDEX` vertex attribute from Draco data to assign per-vertex colors via a `materialColors` map. Outputs interleaved vertices matching the engine's format. Falls back to standard accessor-based decoding for non-Draco primitives. Returns `GlbResult { meshes, skins, animations, nodeTransforms }` — parses node hierarchy, skins (joints + inverseBindMatrices), animations (channels with keyframe samplers), and JOINTS_0/WEIGHTS_0 attributes for skinned meshes.
- **GPU Skinning** (`engine/skin.ts` + `engine/renderer.ts`): `Skeleton` holds joint node indices + inverse bind matrices. `SkinInstance` holds per-instance animation state + pre-allocated scratch buffers. `updateSkinInstance()` samples animation keyframes (binary search + lerp/slerp), traverses node hierarchy parent-first to compute global matrices, then multiplies by inverse bind matrices. Renderer uses a separate `skinnedPipeline` with two vertex buffers (buffer 0: pos/norm/color stride 36, buffer 1: joints uint8x4 + weights float32x4 stride 20) and a `read-only-storage` joint matrices buffer (group 3, dynamic offset, 128 joints × 64 bytes = 8192 per slot).

### Demo scene (`main.ts`):

The demo scene contains a skinned player (Body mesh from `player-bundle.glb` with Run animation looping), a single megaxe, an Eden environment (merged into 1 draw call), and a 7×5 sphere grid. The player entity has `SKINNED` + `INPUT_RECEIVER` components and is movable with WASD. Renderer `maxEntities` is set to 10,000 (matching ECS limit). Stats overlay shows FPS and draw call count (updated every 500ms).

### Shader bind group layout (WGSL in `engine/shaders.ts`):

- Group 0: Camera (view + projection mat4x4f)
- Group 1: Model with dynamic offset (world mat4x4f + color vec4f)
- Group 2: Lighting (direction, dirColor, ambientColor — all vec4f for 16-byte alignment)
- Group 3: Joint matrices storage buffer with dynamic offset (skinned pipeline only — `array<mat4x4f>`, 128 joints per slot)

### Geometry format:

Interleaved vertex data: `[px, py, pz, nx, ny, nz, cr, cg, cb]` per vertex (stride 36 bytes). Vertex colors are multiplied with the per-entity uniform color in the fragment shader — set vertex colors to white `[1,1,1]` for uniform-only coloring, or set uniform color to white for vertex-color-only coloring. Cube is static data, sphere is procedurally generated via `createSphereGeometry(stacks, slices)`. GLB meshes loaded via `loadGlb()` support both uint16 and uint32 index buffers (auto-detected). `mergeGeometries()` in `engine/geometry.ts` merges multiple primitives into a single geometry, baking per-primitive colors into vertex RGB and reindexing — use this to collapse multi-material glTF meshes into one draw call with a white uniform.
