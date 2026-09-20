<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { LogIn } from 'lucide-vue-next'
import { useAuthStore } from '../features/auth/store'
import { ApiError } from '../shared/api/client'
import { safeReturnPath } from '../app/navigation'

const auth = useAuthStore()
const route = useRoute()
const router = useRouter()
const username = ref('')
const password = ref('')
const busy = ref(false)
const error = ref('')

async function submit() {
  if (busy.value) return
  busy.value = true
  error.value = ''
  try {
    await auth.login(username.value, password.value)
    await router.replace(safeReturnPath(route.query.next))
  } catch (caught) {
    error.value =
      caught instanceof ApiError && caught.status === 401
        ? 'Неверное имя пользователя или пароль.'
        : caught instanceof ApiError
          ? caught.message
          : 'Не удалось подключиться к серверу.'
  } finally {
    password.value = ''
    busy.value = false
  }
}
</script>

<template>
  <main class="login-layout">
    <h1>Инженерная поддержка</h1>
    <p v-if="auth.logoutWarning" role="alert" class="notice">{{ auth.logoutWarning }}</p>
    <form
      v-if="auth.config?.authentication.browser_session_enabled"
      class="login-form"
      @submit.prevent="submit"
    >
      <h2>Вход</h2>
      <label for="username">Имя пользователя</label>
      <input
        id="username"
        v-model="username"
        name="username"
        autocomplete="username"
        required
        maxlength="128"
        :disabled="busy"
      />
      <label for="password">Пароль</label>
      <input
        id="password"
        v-model="password"
        name="password"
        type="password"
        autocomplete="current-password"
        required
        maxlength="1024"
        :disabled="busy"
      />
      <p v-if="error" role="alert" class="error-text">{{ error }}</p>
      <button type="submit" class="primary-button" :disabled="busy">
        <LogIn :size="18" />{{ busy ? 'Вход…' : 'Войти' }}
      </button>
    </form>
    <p v-else role="alert">Браузерный вход отключён администратором.</p>
  </main>
</template>
