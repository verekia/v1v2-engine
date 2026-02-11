import { defineConfig } from 'tsup'

export default defineConfig({
  entry: { engine: 'src/engine/index.ts', ecs: 'src/ecs/index.ts' },
  clean: true,
  format: ['esm'],
  dts: true,
  splitting: false,
})
