import { describe, expect, it } from 'vitest'
import type { components } from '../src/shared/api/schema'
import { canAccess, safeReturnPath, screens } from '../src/app/navigation'
import { resolveRouteAccess } from '../src/app/router'

const features = {
  chat: true,
  streaming: true,
  resource_management: true,
  operations: true,
  rag: true,
  multi_agent: true,
  orchestration: true,
  human_review: true,
  a2a: true,
  mcp: true,
} satisfies components['schemas']['AppFeatureFlags']

const config = {
  features,
} as components['schemas']['AppConfigResponse']

const principal = (permissions: string[]) =>
  ({
    subject: 'user',
    auth_method: 'session',
    permissions,
  }) satisfies components['schemas']['CurrentPrincipalResponse']

describe('доступ к разделам', () => {
  it('использует реальное разрешение chat:write для интеграций', () => {
    const integrations = screens.find((screen) => screen.path === '/integrations')

    expect(integrations?.permission).toBe('chat:write')
    expect(canAccess(integrations!, ['chat:write'], features)).toBe(true)
    expect(canAccess(integrations!, ['chat'], features)).toBe(false)
  })

  it('запрещает маршрут без необходимого разрешения', () => {
    expect(
      resolveRouteAccess(
        { path: '/operations' },
        { authenticated: true, principal: principal(['session:read']), config },
      ),
    ).toEqual({ path: '/unavailable', query: { reason: 'forbidden' } })
  })

  it('перенаправляет гостя на вход с локальным return URL', () => {
    expect(
      resolveRouteAccess({ path: '/knowledge' }, { authenticated: false, principal: null, config }),
    ).toEqual({ name: 'login', query: { next: '/knowledge' } })
  })
})

describe('return URL', () => {
  it.each(['/knowledge', '/conversations'])('принимает известный маршрут %s', (path) => {
    expect(safeReturnPath(path)).toBe(path)
  })

  it.each(['https://example.com', '//example.com', '/unknown', ['/knowledge'], undefined])(
    'заменяет небезопасное значение %j',
    (value) => {
      expect(safeReturnPath(value)).toBe('/conversations')
    },
  )
})
