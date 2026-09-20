import createClient from 'openapi-fetch'
import type { paths } from './schema'

let csrfToken: string | null = null
let unauthorizedHandler: (() => void) | null = null

/** CSRF хранится только в памяти вкладки; HttpOnly cookie читает сам браузер. */
export function setCsrfToken(value: string | null) {
  csrfToken = value
}

export function setUnauthorizedHandler(handler: (() => void) | null) {
  unauthorizedHandler = handler
}

export const api = createClient<paths>({
  baseUrl: (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, ''),
  credentials: 'include',
  fetch: (...args) => globalThis.fetch(...args),
})

api.use({
  onRequest({ request }) {
    if (csrfToken && !['GET', 'HEAD', 'OPTIONS'].includes(request.method)) {
      request.headers.set('X-CSRF-Token', csrfToken)
    }
    return request
  },
  onResponse({ response }) {
    if (response.status === 401) unauthorizedHandler?.()
    return response
  },
})

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly requestId: string | null,
    public readonly retryAfter: string | null = null,
  ) {
    const labels: Record<number, string> = {
      401: 'Сессия завершена. Войдите снова.',
      403: 'Недостаточно прав для этого действия.',
      404: 'Ресурс не найден или недоступен.',
      409: 'Состояние изменилось. Обновите данные.',
      413: 'Превышен допустимый размер.',
      422: 'Проверьте заполненные поля.',
      429: 'Превышен лимит запросов. Повторите позже.',
      503: 'Компонент временно недоступен.',
    }
    super(labels[status] || 'Не удалось выполнить запрос.')
    this.name = 'ApiError'
  }
}

/** Не показывает внутренние исключения сервера; request_id связывает ошибку с логами. */
export function unwrap<T>(result: { data?: T; response: Response }): T {
  if (!result.response.ok) {
    throw new ApiError(
      result.response.status,
      result.response.headers.get('X-Request-ID'),
      result.response.headers.get('Retry-After'),
    )
  }
  return result.data as T
}
