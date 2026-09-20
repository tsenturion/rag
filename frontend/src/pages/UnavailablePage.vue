<script setup lang="ts">
import { useRoute } from 'vue-router'
const route = useRoute()
</script>
<template>
  <section class="notice">
    <h1>{{ route.query.reason === 'forbidden' ? 'Нет доступа' : 'Сервис недоступен' }}</h1>
    <p>Проверьте доступность сервера и разрешения учётной записи.</p>
    <a href="/">Повторить</a>
  </section>
</template>
