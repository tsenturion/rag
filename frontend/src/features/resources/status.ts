import type { components } from '../../shared/api/schema'

type OperationStatus = components['schemas']['OperationRecord']['status']
type IncidentStatus = components['schemas']['IncidentRecord']['status']
type RunStatus = components['schemas']['AgentRunState']

type KnownStatus = OperationStatus | IncidentStatus | RunStatus | 'active'

const statusLabels: Record<KnownStatus, string> = {
  active: 'Активен',
  queued: 'В очереди',
  received: 'Получен',
  decomposed: 'Декомпозирован',
  delegated: 'Назначен',
  running: 'Выполняется',
  reviewing: 'Проверяется',
  cancel_requested: 'Запрошена отмена',
  cancelled: 'Отменена',
  completed: 'Завершена',
  failed: 'Не выполнена',
  interrupted: 'Прервана',
  open: 'Открыт',
  in_progress: 'В работе',
  resolved: 'Решён',
  closed: 'Закрыт',
}

export function resourceStatusLabel(status: unknown, active: unknown): string {
  if (typeof status === 'string') return statusLabels[status as KnownStatus] || status
  if (active === true) return 'Активен'
  if (active === false) return 'Отключён'
  return '—'
}
