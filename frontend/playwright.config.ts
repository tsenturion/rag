import { defineConfig, devices } from '@playwright/test'

const liveUrl = process.env.WEB_E2E_URL

export default defineConfig({
  testDir: './tests/e2e',
  outputDir: 'playwright-report/results',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  ...(process.env.CI ? { workers: 1 } : {}),
  reporter: [['list'], ['html', { outputFolder: 'playwright-report/html', open: 'never' }]],
  use: {
    baseURL: liveUrl || 'http://127.0.0.1:4173',
    ...(process.env.CI ? {} : { channel: 'msedge' }),
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    {
      name: 'mobile',
      use: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, isMobile: true },
    },
  ],
  ...(liveUrl
    ? {}
    : {
        webServer: {
          command: 'npm run build && npm run preview -- --strictPort',
          url: 'http://127.0.0.1:4173',
          reuseExistingServer: !process.env.CI,
        },
      }),
})
