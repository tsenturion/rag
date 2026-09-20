<script setup lang="ts">
import { useResource } from '../features/resources/queries'
const query = useResource('integrations')
</script>
<template>
  <section>
    <h1>Интеграции</h1>
    <p v-if="query.isPending.value" role="status">Загрузка…</p>
    <p v-else-if="query.error.value" role="alert">Не удалось загрузить подключения.</p>
    <pre v-else>{{ JSON.stringify(query.data.value, null, 2) }}</pre>
  </section>
</template>
