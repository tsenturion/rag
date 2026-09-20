import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useAuthStore } from '../src/features/auth/store'
import { api } from '../src/shared/api/client'
import { queryClient } from '../src/shared/lib/queryClient'

describe('auth-store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    queryClient.clear()
  })

  it('удаляет данные пользователя при локальной очистке', () => {
    const auth = useAuthStore()
    auth.principal = { subject: 'first', auth_method: 'session', permissions: ['session:read'] }
    queryClient.setQueryData(['resources', 'first'], [{ id: 'private' }])

    auth.clear()

    expect(auth.authenticated).toBe(false)
    expect(queryClient.getQueryData(['resources', 'first'])).toBeUndefined()
  })

  it('очищает состояние и сохраняет предупреждение при ошибке logout на сервере', async () => {
    const auth = useAuthStore()
    auth.principal = { subject: 'first', auth_method: 'session', permissions: [] }
    queryClient.setQueryData(['resources', 'first'], [{ id: 'private' }])
    vi.spyOn(api, 'POST').mockResolvedValue({
      response: new Response(null, { status: 503 }),
      error: undefined,
    })

    await expect(auth.logout()).resolves.toBeUndefined()
    expect(auth.authenticated).toBe(false)
    expect(queryClient.getQueryCache().getAll()).toHaveLength(0)
    expect(auth.logoutWarning).toBe(
      'Не удалось подтвердить выход на сервере. Локальные данные удалены.',
    )
  })

  it('очищает состояние при сетевой ошибке logout', async () => {
    const auth = useAuthStore()
    auth.principal = { subject: 'first', auth_method: 'session', permissions: [] }
    vi.spyOn(api, 'POST').mockRejectedValue(new TypeError('network error'))

    await expect(auth.logout()).resolves.toBeUndefined()

    expect(auth.authenticated).toBe(false)
    expect(auth.logoutWarning).toBe(
      'Не удалось подтвердить выход на сервере. Локальные данные удалены.',
    )
  })
})
