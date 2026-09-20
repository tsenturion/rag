import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../features/auth/store'
import type { components } from '../shared/api/schema'
import { canAccess, screens } from './navigation'

type RouteTarget = { path: string; name?: string | symbol | null | undefined }
type AuthState = {
  authenticated: boolean
  principal: components['schemas']['CurrentPrincipalResponse'] | null
  config: components['schemas']['AppConfigResponse'] | null
}

export function resolveRouteAccess(to: RouteTarget, auth: AuthState) {
  if (to.path === '/unavailable') return true
  if (to.name === 'login') return auth.authenticated ? '/conversations' : true
  if (!auth.authenticated) return { name: 'login', query: { next: to.path } }
  const screen = screens.find((item) => item.path === to.path)
  if (
    screen &&
    !canAccess(screen, auth.principal?.permissions || [], auth.config?.features || {})
  ) {
    return { path: '/unavailable', query: { reason: 'forbidden' } }
  }
  return true
}

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/conversations' },
    { path: '/login', name: 'login', component: () => import('../pages/LoginPage.vue') },
    { path: '/conversations', component: () => import('../pages/ConversationsPage.vue') },
    { path: '/knowledge', component: () => import('../pages/KnowledgePage.vue') },
    { path: '/incidents', component: () => import('../pages/IncidentsPage.vue') },
    { path: '/memory', component: () => import('../pages/MemoryPage.vue') },
    { path: '/projects', component: () => import('../pages/ProjectsPage.vue') },
    { path: '/runs', component: () => import('../pages/RunsPage.vue') },
    { path: '/orchestration', component: () => import('../pages/OrchestrationPage.vue') },
    { path: '/operations', component: () => import('../pages/OperationsPage.vue') },
    { path: '/reviews', component: () => import('../pages/ReviewsPage.vue') },
    { path: '/integrations', component: () => import('../pages/IntegrationsPage.vue') },
    { path: '/users', component: () => import('../pages/UsersPage.vue') },
    { path: '/unavailable', component: () => import('../pages/UnavailablePage.vue') },
    { path: '/:pathMatch(.*)*', component: () => import('../pages/NotFoundPage.vue') },
  ],
})

router.beforeEach(async (to) => {
  if (to.path === '/unavailable') return true
  const auth = useAuthStore()
  try {
    await auth.initialize()
  } catch {
    return '/unavailable'
  }
  return resolveRouteAccess(to, auth)
})
