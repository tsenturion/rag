<script setup lang="ts">
import { computed } from 'vue'
import { RefreshCw } from 'lucide-vue-next'
import { ApiError } from '../../shared/api/client'
import { useResource, resourceRows, type ResourceKind } from './queries'
import { resourceStatusLabel } from './status'

const props = defineProps<{ kind: ResourceKind; title: string }>()
const query = useResource(props.kind)
const rows = computed(() => resourceRows(query.data.value))
const label = (row: Record<string, unknown>) =>
  String(
    row.title ||
      row.name ||
      row.display_name ||
      row.key ||
      row.profile ||
      row.session_id ||
      row.id ||
      row.run_id ||
      row.username ||
      '',
  )
</script>

<template>
  <section :aria-label="title">
    <header class="page-heading">
      <h1>{{ title }}</h1>
      <button
        type="button"
        class="icon-button"
        title="Обновить"
        aria-label="Обновить"
        :disabled="query.isFetching.value"
        @click="query.refetch()"
      >
        <RefreshCw :size="18" />
      </button>
    </header>
    <p v-if="query.isPending.value" role="status">Загрузка…</p>
    <div v-else-if="query.error.value" role="alert" class="notice">
      <p>
        {{
          query.error.value instanceof ApiError ? query.error.value.message : 'Сервер недоступен.'
        }}
      </p>
      <p v-if="query.error.value instanceof ApiError && query.error.value.requestId">
        Запрос: {{ query.error.value.requestId }}
      </p>
      <RouterLink
        v-if="query.error.value instanceof ApiError && query.error.value.status === 401"
        to="/login"
      >
        Войти снова
      </RouterLink>
    </div>
    <p v-else-if="!rows.length" class="empty-state">Записей пока нет.</p>
    <div v-else class="table-scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">Название</th>
            <th scope="col">Состояние</th>
            <th scope="col">Идентификатор</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(row, index) in rows"
            :key="String(row.id || row.run_id || row.session_id || index)"
          >
            <td>{{ label(row) }}</td>
            <td>{{ resourceStatusLabel(row.status, row.active) }}</td>
            <td class="identifier">
              {{ row.id || row.run_id || row.session_id || row.username || '—' }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
