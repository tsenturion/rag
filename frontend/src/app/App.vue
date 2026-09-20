<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { LogOut, PanelsTopLeft } from 'lucide-vue-next'
import { useAuthStore } from '../features/auth/store'
import { canAccess, screens } from './navigation'

const auth = useAuthStore()
const route = useRoute()
const router = useRouter()
const navigation = computed(() =>
  screens.filter((screen) =>
    canAccess(screen, auth.principal?.permissions || [], auth.config?.features || {}),
  ),
)
async function logout() {
  await auth.logout()
  await router.replace('/login')
}
</script>

<template>
  <a href="#main-content" class="skip-link">К содержимому</a>
  <div v-if="auth.authenticated && route.path !== '/login'" class="workspace">
    <aside class="sidebar">
      <div class="brand"><PanelsTopLeft :size="22" /><span>Инженерная поддержка</span></div>
      <nav aria-label="Основные разделы">
        <RouterLink v-for="screen in navigation" :key="screen.path" :to="screen.path">
          {{ screen.title }}
        </RouterLink>
      </nav>
      <div class="identity">
        <span>{{ auth.displayName || auth.principal?.subject }}</span
        ><button class="icon-button" type="button" aria-label="Выйти" title="Выйти" @click="logout">
          <LogOut :size="18" />
        </button>
      </div>
    </aside>
    <main id="main-content" class="content"><RouterView :key="route.path" /></main>
  </div>
  <div v-else id="main-content"><RouterView /></div>
</template>
