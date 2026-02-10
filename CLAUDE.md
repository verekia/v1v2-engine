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

The engine is pure TypeScript with no React dependency. All engine code lives in `src/`.

### Per-frame system execution order (game loop in `main.ts`):

1. **Input system** — WASD keyboard state applied to entities with `TRANSFORM | INPUT_RECEIVER`
2. **Transform system** — Compute `worldMatrix` from position/rotation/scale via `m4FromTRS`
3. **Camera system** — Compute view matrix from OrbitControls' `eye`/`target` Float32Arrays, and projection matrix
4. **Render system** — Frustum cull, upload uniforms, issue draw calls

### Key design patterns:

- **Zero-allocation math** (`math.ts`): All vec3/mat4 functions write to a pre-allocated `Float32Array` at a given offset. No temp objects, no GC pressure in the hot path. Includes `quatToEulerYXZ` for glTF quaternion → Euler conversion matching the Y×X×Z rotation order.
- **Structure-of-Arrays ECS** (`ecs.ts`): Component data stored as parallel TypedArrays (e.g., `positions: Float32Array[MAX*3]`), not per-entity objects. Component presence tracked via bitmasks (`TRANSFORM = 1 << 0`, etc.) with inline iteration — no query abstraction.
- **Dynamic uniform buffer** (`renderer.ts`): Per-entity model data packed into 256-byte aligned slots in a single GPU buffer, bound via dynamic offsets to avoid per-entity bind group allocation.
- **Frustum culling** (`renderer.ts` + `math.ts`): Gribb-Hartmann plane extraction from the VP matrix, bounding sphere test per entity before upload and draw.
- **OrbitControls** (`orbit-controls.ts`): Spherical coordinates (theta/phi/radius) around a target point. Outputs `eye` and `target` Float32Arrays consumed by the camera system's `m4LookAt`. Left-drag orbits, right-drag pans, scroll zooms.
- **GLB/glTF loader** (`gltf.ts`): Loads `.glb` files with Draco mesh compression (`KHR_draco_mesh_compression`). Draco WASM decoder loaded at runtime from `public/draco-1.5.7/` via `<script>` tag (singleton, no npm package). Supports reading custom `_MATERIALINDEX` vertex attribute from Draco data to assign per-vertex colors via a `materialColors` map. Outputs interleaved vertices matching the engine's format. Falls back to standard accessor-based decoding for non-Draco primitives.

### Demo scene (`main.ts`):

The demo scene contains a movable cube, a single megaxe, a 100×50 grid of 5,000 megaxes (at Z=-20, behind the spheres), and a 7×5 sphere grid. Renderer `maxEntities` is set to 10,000 (matching ECS limit). Stats overlay shows FPS and draw call count (updated every 500ms).

### Shader bind group layout (WGSL in `shaders.ts`):

- Group 0: Camera (view + projection mat4x4f)
- Group 1: Model with dynamic offset (world mat4x4f + color vec4f)
- Group 2: Lighting (direction, dirColor, ambientColor — all vec4f for 16-byte alignment)

### Geometry format:

Interleaved vertex data: `[px, py, pz, nx, ny, nz, cr, cg, cb]` per vertex (stride 36 bytes). Vertex colors are multiplied with the per-entity uniform color in the fragment shader — set vertex colors to white `[1,1,1]` for uniform-only coloring, or set uniform color to white for vertex-color-only coloring. Cube is static data, sphere is procedurally generated via `createSphereGeometry(stacks, slices)`. GLB meshes loaded via `loadGlb()` support both uint16 and uint32 index buffers (auto-detected).
