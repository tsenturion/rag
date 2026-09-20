import { loadEnv } from 'vite'
import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, process.cwd(), '')
  const target = environment.API_PROXY_TARGET || 'http://127.0.0.1:8000'
  const proxy = Object.fromEntries(
    ['/v1', '/health', '/ready'].map((path) => [path, { target, changeOrigin: false }]),
  )
  return {
    plugins: [vue(), tailwindcss()],
    server: {
      port: 5173,
      proxy,
      watch: { ignored: ['**/playwright-report/**', '**/coverage/**'] },
    },
    preview: { port: 4173, proxy },
    build: { sourcemap: false },
    test: {
      environment: 'jsdom',
      env: { VITE_API_BASE_URL: 'http://localhost' },
      include: ['tests/**/*.test.ts'],
      exclude: ['tests/e2e/**', 'node_modules/**', 'dist/**'],
      clearMocks: true,
    },
  }
})
