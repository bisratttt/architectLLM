import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/architectLLM/',
  server: {
    port: parseInt(process.env.PORT) || 5173,
    strictPort: true,
  },
  build: {
    outDir: '../docs',
    emptyOutDir: true,
  },
})
