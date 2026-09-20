import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { VueQueryPlugin } from '@tanstack/vue-query'
import App from './App.vue'
import { router } from './router'
import { useAuthStore } from '../features/auth/store'
import { setUnauthorizedHandler } from '../shared/api/client'
import { queryClient } from '../shared/lib/queryClient'
import './styles.css'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)
app.use(VueQueryPlugin, { queryClient })
app.use(router)

const auth = useAuthStore(pinia)
setUnauthorizedHandler(() => {
  auth.clear()
  const route = router.currentRoute.value
  if (route.name !== 'login' && route.path !== '/unavailable') {
    void router.replace({ name: 'login', query: { next: route.path } })
  }
})

app.mount('#app')
