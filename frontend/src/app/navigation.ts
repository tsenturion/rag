import type { components } from '../shared/api/schema'

type Feature = keyof components['schemas']['AppFeatureFlags']
export type Screen = { path: string; title: string; permission: string; feature?: Feature }

export const screens: Screen[] = [
  { path: '/conversations', title: 'Диалоги', permission: 'session:read' },
  {
    path: '/knowledge',
    title: 'База знаний',
    permission: 'knowledge:read',
    feature: 'resource_management',
  },
  {
    path: '/incidents',
    title: 'Инциденты',
    permission: 'incident:read',
    feature: 'resource_management',
  },
  { path: '/memory', title: 'Память', permission: 'memory:read', feature: 'resource_management' },
  {
    path: '/projects',
    title: 'Проекты',
    permission: 'memory:read',
    feature: 'resource_management',
  },
  { path: '/runs', title: 'Запуски агентов', permission: 'run:read', feature: 'multi_agent' },
  {
    path: '/orchestration',
    title: 'Оркестрация',
    permission: 'orchestration:read',
    feature: 'orchestration',
  },
  { path: '/operations', title: 'Операции', permission: 'operation:read', feature: 'operations' },
  {
    path: '/reviews',
    title: 'Проверка ответов',
    permission: 'review:read',
    feature: 'human_review',
  },
  { path: '/integrations', title: 'Интеграции', permission: 'chat:write' },
  {
    path: '/users',
    title: 'Пользователи',
    permission: 'admin:write',
    feature: 'resource_management',
  },
]

/** Видимость не заменяет серверный RBAC: один predicate используется меню и маршрутизатором. */
export function canAccess(
  screen: Screen,
  permissions: readonly string[],
  features: Partial<Record<Feature, boolean>>,
) {
  return (
    permissions.includes(screen.permission) &&
    (!screen.feature || features[screen.feature] === true)
  )
}

export function safeReturnPath(value: unknown) {
  return typeof value === 'string' && screens.some((screen) => screen.path === value)
    ? value
    : '/conversations'
}
