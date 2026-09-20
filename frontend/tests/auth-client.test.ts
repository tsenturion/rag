import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, setCsrfToken, setUnauthorizedHandler, unwrap } from '../src/shared/api/client'

describe('API-клиент', () => {
  beforeEach(() => {
    setCsrfToken(null)
    setUnauthorizedHandler(null)
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => vi.unstubAllGlobals())

  it('добавляет CSRF только к изменяющим запросам', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }))
    setCsrfToken('csrf-value')

    await api.POST('/v1/auth/logout')

    const request = fetchMock.mock.calls[0]?.[0]
    expect(request).toBeInstanceOf(Request)
    expect((request as Request).headers.get('X-CSRF-Token')).toBe('csrf-value')
  })

  it('не добавляет CSRF к безопасным запросам', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockResolvedValue(Response.json({ api_version: 'v1' }, { status: 200 }))
    setCsrfToken('csrf-value')

    await api.GET('/v1/app/config')

    const request = fetchMock.mock.calls[0]?.[0] as Request
    expect(request.headers.has('X-CSRF-Token')).toBe(false)
  })

  it('сообщает об истёкшей сессии при ответе 401', async () => {
    const expired = vi.fn()
    vi.mocked(fetch).mockResolvedValue(new Response(null, { status: 401 }))
    setUnauthorizedHandler(expired)

    await api.GET('/v1/auth/me')

    expect(expired).toHaveBeenCalledOnce()
  })

  it('нормализует HTTP-ошибку без раскрытия тела ответа', () => {
    const response = new Response(JSON.stringify({ detail: 'internal error' }), {
      status: 429,
      headers: { 'X-Request-ID': 'request-42', 'Retry-After': '10' },
    })

    expect(() => unwrap({ response })).toThrowError(
      expect.objectContaining({
        message: 'Превышен лимит запросов. Повторите позже.',
        status: 429,
        requestId: 'request-42',
        retryAfter: '10',
      }),
    )
  })
})
