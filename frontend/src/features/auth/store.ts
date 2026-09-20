import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { api, ApiError, setCsrfToken, unwrap } from '../../shared/api/client'
import type { components } from '../../shared/api/schema'
import { queryClient } from '../../shared/lib/queryClient'

export const useAuthStore = defineStore('auth', () => {
  const config = ref<components['schemas']['AppConfigResponse'] | null>(null)
  const principal = ref<components['schemas']['CurrentPrincipalResponse'] | null>(null)
  const displayName = ref('')
  const logoutWarning = ref('')
  const bootstrapped = ref(false)
  let initialization: Promise<void> | null = null
  const authenticated = computed(() => principal.value !== null)

  function clear() {
    principal.value = null
    displayName.value = ''
    setCsrfToken(null)
    queryClient.clear()
  }

  async function restore() {
    try {
      const session = unwrap(await api.GET('/v1/auth/session'))
      setCsrfToken(session.csrf_token)
      displayName.value = session.user.display_name
      principal.value = unwrap(await api.GET('/v1/auth/me'))
    } catch (error) {
      clear()
      if (!(error instanceof ApiError) || error.status !== 401) throw error
    }
  }

  /** Один bootstrap на навигацию исключает гонки повторного восстановления сессии. */
  async function initialize() {
    if (bootstrapped.value) return
    initialization ??= (async () => {
      config.value = unwrap(await api.GET('/v1/app/config'))
      if (config.value.authentication.browser_session_enabled) await restore()
      bootstrapped.value = true
    })().finally(() => {
      initialization = null
    })
    return initialization
  }

  async function login(username: string, password: string) {
    clear()
    try {
      const session = unwrap(await api.POST('/v1/auth/login', { body: { username, password } }))
      setCsrfToken(session.csrf_token)
      displayName.value = session.user.display_name
      principal.value = unwrap(await api.GET('/v1/auth/me'))
    } catch (error) {
      clear()
      throw error
    }
  }

  async function logout() {
    logoutWarning.value = ''
    try {
      const result = await api.POST('/v1/auth/logout')
      if (!result.response.ok && result.response.status !== 401) unwrap(result)
    } catch {
      logoutWarning.value = 'Не удалось подтвердить выход на сервере. Локальные данные удалены.'
    } finally {
      clear()
    }
  }

  return {
    config,
    principal,
    displayName,
    logoutWarning,
    bootstrapped,
    authenticated,
    initialize,
    login,
    logout,
    clear,
  }
})
