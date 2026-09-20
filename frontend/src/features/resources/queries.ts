import { computed } from 'vue'
import { useQuery } from '@tanstack/vue-query'
import { useAuthStore } from '../auth/store'
import { api, unwrap } from '../../shared/api/client'

export type ResourceKind =
  | 'conversations'
  | 'knowledge'
  | 'incidents'
  | 'memory'
  | 'projects'
  | 'runs'
  | 'orchestration'
  | 'operations'
  | 'reviews'
  | 'users'
  | 'integrations'

/** Запросы используют сгенерированный OpenAPI-контракт; API-адреса не строятся из ввода. */
export async function loadResource(kind: ResourceKind, userId: string, signal?: AbortSignal) {
  const options = signal ? { signal } : {}
  switch (kind) {
    case 'conversations':
      return unwrap(
        await api.GET('/v1/sessions', { ...options, params: { query: { user_id: userId } } }),
      )
    case 'knowledge':
      return unwrap(await api.GET('/v1/knowledge/sources', options))
    case 'incidents':
      return unwrap(await api.GET('/v1/incidents', options))
    case 'memory':
      return unwrap(await api.GET('/v1/memories', options))
    case 'projects':
      return unwrap(await api.GET('/v1/projects', options))
    case 'runs':
      return unwrap(await api.GET('/v1/multi-agent/runs', options))
    case 'orchestration':
      return unwrap(await api.GET('/v1/orchestration/jobs', options))
    case 'operations':
      return unwrap(await api.GET('/v1/operations', options))
    case 'reviews':
      return unwrap(await api.GET('/v1/reviews', options))
    case 'users':
      return unwrap(await api.GET('/v1/admin/users', options))
    case 'integrations':
      return unwrap(await api.GET('/v1/integrations', options))
  }
}

export function useResource(kind: ResourceKind) {
  const auth = useAuthStore()
  return useQuery({
    queryKey: computed(() => ['resources', auth.principal?.subject, kind]),
    enabled: computed(() => auth.authenticated),
    queryFn: ({ signal }) => loadResource(kind, auth.principal?.subject || '', signal),
    retry: false,
    staleTime: 15_000,
  })
}

/** Списки старых и новых маршрутов приводятся только на границе представления. */
export function resourceRows(data: unknown): Record<string, unknown>[] {
  if (Array.isArray(data)) return data.filter(isRecord)
  if (!isRecord(data)) return []
  for (const key of ['items', 'sessions', 'reviews']) {
    if (Array.isArray(data[key])) return data[key].filter(isRecord)
  }
  return Object.entries(data).map(([key, value]) => ({ id: key, title: key, value }))
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
