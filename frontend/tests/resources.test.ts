import { describe, expect, it } from 'vitest'
import { resourceRows } from '../src/features/resources/queries'
import { resourceStatusLabel } from '../src/features/resources/status'

describe('нормализация ресурсов', () => {
  it('извлекает записи из именованных списков', () => {
    expect(resourceRows({ items: [{ id: 'one' }, null, 'invalid'] })).toEqual([{ id: 'one' }])
    expect(resourceRows({ sessions: [{ session_id: 'session-1' }] })).toEqual([
      { session_id: 'session-1' },
    ])
  })

  it('нормализует словарь и отбрасывает скалярный ответ', () => {
    expect(resourceRows({ alpha: 1, beta: { active: true } })).toEqual([
      { id: 'alpha', title: 'alpha', value: 1 },
      { id: 'beta', title: 'beta', value: { active: true } },
    ])
    expect(resourceRows('invalid')).toEqual([])
  })

  it('переводит известные состояния и явно показывает флаг активности', () => {
    expect(resourceStatusLabel('active', undefined)).toBe('Активен')
    expect(resourceStatusLabel('queued', undefined)).toBe('В очереди')
    expect(resourceStatusLabel('cancel_requested', undefined)).toBe('Запрошена отмена')
    expect(resourceStatusLabel('resolved', undefined)).toBe('Решён')
    expect(resourceStatusLabel('decomposed', undefined)).toBe('Декомпозирован')
    expect(resourceStatusLabel(undefined, true)).toBe('Активен')
    expect(resourceStatusLabel(undefined, false)).toBe('Отключён')
    expect(resourceStatusLabel('custom_status', undefined)).toBe('custom_status')
  })
})
