import { expect, test, type Page, type TestInfo } from '@playwright/test'

const liveUrl = process.env.WEB_E2E_URL
const fixtureUsername = 'engineer'
const fixturePassword = 'correct-password'
const allPermissions = [
  'session:read',
  'knowledge:read',
  'incident:read',
  'memory:read',
  'run:read',
  'orchestration:read',
  'operation:read',
  'review:read',
  'chat:write',
  'admin:write',
]

const appConfig = {
  api_version: 'v1',
  service: 'engineer-support-agent',
  provider: 'fixture',
  model: 'fixture',
  features: {
    chat: true,
    streaming: true,
    resource_management: true,
    operations: true,
    rag: true,
    multi_agent: true,
    orchestration: true,
    human_review: true,
    a2a: false,
    mcp: false,
  },
  authentication: {
    api_key_enabled: false,
    jwt_enabled: false,
    user_scope_enforced: true,
    api_key_header: 'X-API-Key',
    bearer_scheme: 'Bearer',
    browser_session_enabled: true,
    csrf_header: 'X-CSRF-Token',
  },
  limits: {
    request_max_chars: 10_000,
    max_history_messages: 50,
    rate_limit_enabled: false,
  },
  openapi_url: '/openapi.json',
  docs_url: '/docs',
}

async function installHttpFixtures(page: Page) {
  let authenticated = false

  await page.route('**/v1/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })

    if (path === '/v1/app/config') return json(appConfig)
    if (path === '/v1/auth/session') {
      if (!authenticated) return json({ detail: 'unauthorized' }, 401)
      return json({
        user: {
          username: fixtureUsername,
          display_name: 'Тестовый инженер',
          roles: ['engineer'],
          active: true,
        },
        csrf_token: 'fixture-csrf-token',
        expires_at: 4_102_444_800,
      })
    }
    if (path === '/v1/auth/login' && request.method() === 'POST') {
      const credentials = request.postDataJSON() as { username?: string; password?: string }
      if (credentials.username !== fixtureUsername || credentials.password !== fixturePassword) {
        return json({ detail: 'unauthorized' }, 401)
      }
      authenticated = true
      return json({
        user: {
          username: fixtureUsername,
          display_name: 'Тестовый инженер',
          roles: ['engineer'],
          active: true,
        },
        csrf_token: 'fixture-csrf-token',
        expires_at: 4_102_444_800,
      })
    }
    if (path === '/v1/auth/me' && authenticated) {
      return json({
        subject: fixtureUsername,
        roles: ['engineer'],
        permissions: allPermissions,
        auth_method: 'session',
      })
    }
    if (path === '/v1/auth/logout' && request.method() === 'POST') {
      authenticated = false
      return route.fulfill({ status: 204, body: '' })
    }
    if (path === '/v1/sessions') {
      return json({
        sessions: [{ session_id: 'incident-42', title: 'Диагностика обращения', status: 'active' }],
      })
    }
    if (authenticated && request.method() === 'GET') return json({ items: [] })
    return json({ detail: 'unauthorized' }, 401)
  })
}

async function login(page: Page, username = fixtureUsername, password = fixturePassword) {
  await page.goto('/login')
  await page.getByLabel('Имя пользователя').fill(username)
  await page.getByLabel('Пароль').fill(password)
  await page.getByRole('button', { name: 'Войти' }).click()
  await expect(page).toHaveURL(/\/conversations$/)
}

async function saveScreenshot(page: Page, testInfo: TestInfo, name: string) {
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: testInfo.outputPath(`${name}.png`), fullPage: true })
}

test.describe('каркас приложения с HTTP-фикстурами', () => {
  test.skip(Boolean(liveUrl), 'HTTP-фикстуры используются только для локального и CI-запуска')

  test.beforeEach(async ({ page }) => installHttpFixtures(page))

  test('вход, навигация, запрет маршрута, 404 и выход', async ({ page }, testInfo) => {
    await page.goto('/knowledge')
    await expect(page).toHaveURL(/\/login\?next=\/knowledge$/)

    await login(page)
    await expect(page.getByRole('heading', { name: 'Диалоги' })).toBeVisible()
    await expect(page.getByText('Диагностика обращения')).toBeVisible()

    await page.getByRole('link', { name: 'База знаний' }).click()
    await expect(page).toHaveURL(/\/knowledge$/)
    await expect(page.getByRole('heading', { name: 'База знаний' })).toBeVisible()

    await page.goto('/missing-page')
    await expect(page.getByRole('heading', { name: 'Страница не найдена' })).toBeVisible()

    await page.goto('/users')
    await expect(page.getByRole('heading', { name: 'Пользователи' })).toBeVisible()
    await page.getByRole('button', { name: 'Выйти' }).click()
    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByRole('button', { name: 'Войти' })).toBeVisible()
    await saveScreenshot(page, testInfo, 'logout')
  })

  test('скрывает запрещённый пункт и отклоняет прямой переход', async ({ page }) => {
    await page.route('**/v1/auth/me', (route) =>
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          subject: fixtureUsername,
          roles: ['engineer'],
          permissions: ['session:read'],
          auth_method: 'session',
        }),
      }),
    )
    await login(page)
    await expect(page.getByRole('link', { name: 'Пользователи' })).toHaveCount(0)

    await page.goto('/users')
    await expect(page).toHaveURL(/\/unavailable\?reason=forbidden$/)
    await expect(page.getByRole('heading', { name: 'Нет доступа' })).toBeVisible()
  })

  test('показывает предупреждение при сетевой ошибке logout', async ({ page }) => {
    await login(page)
    await page.unroute('**/v1/**')
    await page.route('**/v1/auth/logout', (route) => route.abort('failed'))

    await page.getByRole('button', { name: 'Выйти' }).click()

    await expect(page).toHaveURL(/\/login$/)
    await expect(page.getByRole('alert')).toHaveText(
      'Не удалось подтвердить выход на сервере. Локальные данные удалены.',
    )
    await expect(page.getByRole('button', { name: 'Войти' })).toBeVisible()
  })

  test('не создаёт горизонтальное переполнение', async ({ page }, testInfo) => {
    await login(page)
    const dimensions = await page.evaluate(() => ({
      viewport: document.documentElement.clientWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
    }))
    expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport)
    expect(dimensions.body).toBeLessThanOrEqual(dimensions.viewport)
    await saveScreenshot(page, testInfo, 'shell')
  })
})

test.describe('live smoke', () => {
  test.skip(!liveUrl, 'Для live-проверки задайте WEB_E2E_URL, WEB_E2E_USERNAME и WEB_E2E_PASSWORD')

  test('входит, переходит по меню и выходит', async ({ page }) => {
    const username = process.env.WEB_E2E_USERNAME
    const password = process.env.WEB_E2E_PASSWORD
    test.skip(!username || !password, 'Для live-проверки нужны WEB_E2E_USERNAME и WEB_E2E_PASSWORD')

    await login(page, username, password)
    const firstNavigationLink = page.getByRole('navigation').getByRole('link').first()
    await expect(firstNavigationLink).toBeVisible()
    await firstNavigationLink.click()
    await expect(page.locator('#main-content h1')).toBeVisible()
    await page.getByRole('button', { name: 'Выйти' }).click()
    await expect(page).toHaveURL(/\/login$/)
  })
})
