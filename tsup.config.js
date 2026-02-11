import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/engine/index.ts'],
  clean: true,
  format: ['esm'],
  dts: true,
  splitting: false,
})
