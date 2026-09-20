// Сгенерировано из OpenAPI; изменения выполняются через npm run api:generate.
export interface paths {
    "/v1/app/config": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить bootstrap-конфигурацию web-клиента
         * @description Публично возвращает только возможности, способы авторизации и лимиты API. Секреты и внутренние пути в ответ не включаются.
         */
        get: operations["app_config_v1_app_config_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/auth/me": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить текущую identity и разрешения
         * @description Возвращает только проверенные claims, а не содержимое JWT или API key.
         */
        get: operations["auth_me_v1_auth_me_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/chat": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Отправить запрос агенту
         * @description Выполняет один ход агента. В зависимости от запроса LLM использует RAG, инженерные tools и память. Ответ содержит trace, citations и retrieval diagnostics.
         */
        post: operations["chat_v1_chat_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/multi-agent/chat": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Выполнить запрос через supervisor-граф
         * @description Декомпозирует запрос, делегирует подзадачи профильным агентам, проверяет отчёты критиком и возвращает итог с lifecycle и usage.
         */
        post: operations["multi_agent_chat_v1_multi_agent_chat_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/multi-agent/compare": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Сравнить single-agent и multi-agent
         * @description Выполняет один запрос в обоих режимах при общей модели и возвращает качество, latency, tokens, tool calls и настраиваемую стоимость.
         */
        post: operations["compare_agents_v1_multi_agent_compare_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/multi-agent/runs/{run_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить сохранённый результат запуска
         * @description Возвращает сохранённые данные мультиагентного запуска по идентификатору, гарантируя ошибку 404 при отсутствии записи.
         */
        get: operations["get_multi_agent_run_v1_multi_agent_runs__run_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/orchestration/jobs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Jobs
         * @description Отображает также задания, поставленные через CLI, с текущими broker-статусами.
         */
        get: operations["jobs_v1_orchestration_jobs_get"];
        put?: never;
        /**
         * Поставить задание в оркестратор
         * @description Создаёт воспроизводимое задание с выбранным паттерном. В inline режиме оно выполняется до возврата ответа; в Celery-режиме попадает в приоритетную RabbitMQ-очередь.
         */
        post: operations["submit_orchestration_job_v1_orchestration_jobs_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/orchestration/jobs/{job_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить состояние задания
         * @description Возвращает запись оркестрационного задания по идентификатору с гарантией ошибки 404 при отсутствии, обеспечивая надёжный доступ к данным.
         */
        get: operations["get_orchestration_job_v1_orchestration_jobs__job_id__get"];
        put?: never;
        post?: never;
        /**
         * Отменить незавершённое задание
         * @description Гарантирует корректное завершение или отмену оркестрационной задачи с явной ошибкой 404 при отсутствии указанной задачи.
         */
        delete: operations["cancel_orchestration_job_v1_orchestration_jobs__job_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/orchestration/jobs/{job_id}/events": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить журнал событий задания
         * @description Возвращает список событий оркестрационного задания с гарантией ошибки 404 при отсутствии задания, обеспечивая целостность истории событий.
         */
        get: operations["get_orchestration_events_v1_orchestration_jobs__job_id__events_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/orchestration/queues/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Проверить очередь и workers
         * @description Обеспечивает вызывающему коду актуальное состояние очереди оркестрации для мониторинга и диагностики.
         */
        get: operations["orchestration_queue_status_v1_orchestration_queues_status_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/chat/stream": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Отправить запрос и получить SSE-события
         * @description Возвращает событие `started`, затем `result` с полным ChatResponse. Это поток этапов выполнения, а не token-by-token streaming LLM.
         */
        post: operations["chat_stream_v1_chat_stream_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/sessions": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить список сохранённых диалогов
         * @description Возвращает диалоги текущего user_id в порядке последней активности. Для следующей страницы передайте непрозрачный next_cursor.
         */
        get: operations["list_sessions_v1_sessions_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/sessions/{session_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить состояние сессии
         * @description Возвращает доступные текущему user_id записи памяти и связанные инциденты. Чужие user-scoped данные не возвращаются.
         */
        get: operations["get_session_v1_sessions__session_id__get"];
        put?: never;
        post?: never;
        /**
         * Очистить сессию
         * @description Удаляет session-scoped память и AgentRunner из локального кэша. Глобальная долговременная память пользователя сохраняется.
         */
        delete: operations["delete_session_v1_sessions__session_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/reviews": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить очередь human review
         * @description Гарантирует возврат ограниченного списка заявок на ручную модерацию для последующей обработки или аудита.
         */
        get: operations["list_reviews_v1_reviews_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/reviews/{review_id}/decision": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Подтвердить или отклонить ответ
         * @description Гарантирует принятие решения по заявке на ручную модерацию с аудированием события и ошибкой 404 при невозможности обработки.
         */
        post: operations["decide_review_v1_reviews__review_id__decision_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/security/audit": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить журнал решений безопасности
         * @description Гарантирует возврат последних событий аудита безопасности для анализа действий пользователей и администраторов.
         */
        get: operations["security_audit_v1_security_audit_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/health": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Проверить liveness
         * @description Проверяет, что HTTP-процесс запущен.
         */
        get: operations["health_health_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/ready": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Проверить readiness
         * @description Проверяет конфигурацию безопасности, LLM runtime, Qdrant collection, embedding provider и размерность vectors без платного LLM-запроса.
         */
        get: operations["ready_ready_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/metrics": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Получить Prometheus metrics
         * @description Возвращает счётчики запросов, latency и статусы retrieval субъекту с разрешением metrics:read.
         */
        get: operations["metrics_metrics_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/auth/login": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Login
         * @description Проверяет origin до выдачи host-only HttpOnly cookie с фиксированным сроком.
         */
        post: operations["login_v1_auth_login_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/auth/session": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Session
         * @description Восстанавливает identity и CSRF после перезагрузки страницы без продления TTL.
         */
        get: operations["session_v1_auth_session_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/auth/logout": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Logout
         * @description Отзывает cookie в БД и очищает её в браузере.
         */
        post: operations["logout_v1_auth_logout_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/admin/users": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Users
         * @description Возвращает следующую страницу пользователей без хешей паролей.
         */
        get: operations["users_v1_admin_users_get"];
        put?: never;
        /**
         * Create User
         * @description Саморегистрация отсутствует: аккаунт создаёт проверенный администратор.
         */
        post: operations["create_user_v1_admin_users_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/admin/users/{username}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /**
         * Update User
         * @description Блокировка и сброс пароля немедленно отзывают активные сессии.
         */
        patch: operations["update_user_v1_admin_users__username__patch"];
        trace?: never;
    };
    "/v1/memories": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Memories
         * @description Пагинация выполняется по идентификатору; истёкшая память исключается store-политикой.
         */
        get: operations["memories_v1_memories_get"];
        put?: never;
        /**
         * Save Memory
         * @description Отклоняет секреты перед сохранением факта или предпочтения пользователя.
         */
        post: operations["save_memory_v1_memories_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/memories/{memory_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Memory Detail
         * @description Карточка памяти доступна владельцу; чужая и истёкшая запись дают одинаковый 404.
         */
        get: operations["memory_detail_v1_memories__memory_id__get"];
        put?: never;
        post?: never;
        /**
         * Delete Memory
         * @description Чужой UUID не позволяет удалить запись другого пользователя.
         */
        delete: operations["delete_memory_v1_memories__memory_id__delete"];
        options?: never;
        head?: never;
        /**
         * Update Memory
         * @description Изменяет только запись проверенного владельца.
         */
        patch: operations["update_memory_v1_memories__memory_id__patch"];
        trace?: never;
    };
    "/v1/incidents": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Incidents
         * @description Возвращает стабильную страницу инцидентов текущего инженера.
         */
        get: operations["incidents_v1_incidents_get"];
        put?: never;
        /**
         * Create Incident
         * @description Создаёт инцидент тем же доменным сервисом, что инженерный tool.
         */
        post: operations["create_incident_v1_incidents_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/incidents/{incident_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Incident
         * @description Карточка инцидента проверяет владельца на сервере.
         */
        get: operations["incident_v1_incidents__incident_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /**
         * Update Incident
         * @description Не создаёт отсутствующую запись при ошибке идентификатора.
         */
        patch: operations["update_incident_v1_incidents__incident_id__patch"];
        trace?: never;
    };
    "/v1/projects": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Projects
         * @description Проекты и задачи представлены типизированными записями долговременной памяти.
         */
        get: operations["projects_v1_projects_get"];
        put?: never;
        /**
         * Create Project
         * @description Создаёт проект, доступный в следующих диалогах пользователя.
         */
        post: operations["create_project_v1_projects_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/projects/tasks": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Create Task
         * @description Сохраняет задачу через существующий project tool.
         */
        post: operations["create_task_v1_projects_tasks_post"];
        delete?: never;
        options?: never;
        head?: never;
        /**
         * Update Task
         * @description Обновляет статус по доменному ключу проекта и задачи.
         */
        patch: operations["update_task_v1_projects_tasks_patch"];
        trace?: never;
    };
    "/v1/multi-agent/runs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Runs
         * @description Находит сохранённые запуски после перезапуска API с фильтрацией владельца.
         */
        get: operations["runs_v1_multi_agent_runs_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/operations/profiles": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Profiles
         * @description Возвращает зарегистрированные действия и реальную готовность worker.
         */
        get: operations["profiles_v1_operations_profiles_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/operations": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Operations
         * @description Список операций сохраняется независимо от вкладки браузера.
         */
        get: operations["operations_v1_operations_get"];
        put?: never;
        /**
         * Submit
         * @description Создаёт ровно одну операцию для пары владелец/ключ идемпотентности.
         */
        post: operations["submit_v1_operations_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/operations/{operation_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Operation
         * @description Состояние читается из БД, а не из памяти HTTP worker.
         */
        get: operations["get_operation_v1_operations__operation_id__get"];
        put?: never;
        post?: never;
        /**
         * Cancel
         * @description Отмена отмечается в БД и обрабатывается владельцем процесса.
         */
        delete: operations["cancel_v1_operations__operation_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/operations/{operation_id}/artifacts": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Artifacts
         * @description Список опубликованных результатов доступен только владельцу запуска.
         */
        get: operations["artifacts_v1_operations__operation_id__artifacts_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/operations/{operation_id}/artifacts/{artifact_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Download Artifact
         * @description Артефакт отдаётся как скачивание, а HTML не исполняется на origin приложения.
         */
        get: operations["download_artifact_v1_operations__operation_id__artifacts__artifact_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/operations/{operation_id}/preview/{artifact_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Preview Artifact
         * @description JSONL читается построчно, поэтому просмотр embedding-корпуса не загружает его целиком.
         */
        get: operations["preview_artifact_v1_operations__operation_id__preview__artifact_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/operations/{operation_id}/logs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Logs
         * @description Отдаёт ограниченный блок сохранённого журнала с курсором в байтах.
         */
        get: operations["logs_v1_operations__operation_id__logs_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/knowledge/sources": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Sources
         * @description Возвращает только исходники текущего пользователя.
         */
        get: operations["sources_v1_knowledge_sources_get"];
        put?: never;
        /**
         * Upload
         * @description Лимит проверяется при чтении потока, включая запрос без Content-Length.
         */
        post: operations["upload_v1_knowledge_sources_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/knowledge/sources/{source_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Download Source
         * @description Проверяет hash источника перед скачиванием.
         */
        get: operations["download_source_v1_knowledge_sources__source_id__get"];
        put?: never;
        post?: never;
        /**
         * Delete Source
         * @description Не удаляет исходник, пока worker использует его для подготовки корпуса.
         */
        delete: operations["delete_source_v1_knowledge_sources__source_id__delete"];
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/knowledge/search": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Search
         * @description Поиск вызывает только configured embedding-провайдер и проходит лимит платных запросов.
         */
        post: operations["search_v1_knowledge_search_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/integrations": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Integrations
         * @description Возвращает безопасный каталог подключений без адресов с credentials.
         */
        get: operations["integrations_v1_integrations_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/v1/telemetry/browser": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Browser Error
         * @description Сохраняет диагностический код и связь с запросом без произвольного текста клиента.
         */
        post: operations["browser_error_v1_telemetry_browser_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /**
         * AgentModeResult
         * @description Фиксирует все существенные метрики и параметры ответа агента в выбранном режиме для последующего анализа и сравнения.
         */
        AgentModeResult: {
            /**
             * Mode
             * @enum {string}
             */
            mode: "single" | "multi";
            /** Answer */
            answer: string;
            /**
             * Citations Count
             * @default 0
             */
            citations_count: number;
            /** Tool Calls */
            tool_calls?: string[];
            /** Selected Agents */
            selected_agents?: string[];
            quality: components["schemas"]["QualityAssessment"];
            usage: components["schemas"]["UsageMetrics"];
            /** Run Id */
            run_id?: string | null;
        };
        /**
         * AgentRetrievalInfo
         * @description Гарантирует вызывающему коду прозрачный контракт о статусе, объёме и источнике извлечённых данных для трассировки и аудита работы retrieval-подсистемы.
         */
        AgentRetrievalInfo: {
            /** Status */
            status: string;
            /**
             * Retrieved Count
             * @default 0
             */
            retrieved_count: number;
            /**
             * Used Count
             * @default 0
             */
            used_count: number;
            /**
             * Context Tokens
             * @default 0
             */
            context_tokens: number;
            /** Provider */
            provider?: string | null;
            /** Model */
            model?: string | null;
            /** Collection Name */
            collection_name?: string | null;
            /** Error */
            error?: string | null;
        };
        /**
         * AgentRunState
         * @description Определяет стадии жизненного цикла агента в мультиагентной системе, обеспечивая контроль и управление состоянием выполнения задач.
         * @enum {string}
         */
        AgentRunState: "received" | "decomposed" | "delegated" | "running" | "reviewing" | "completed" | "failed";
        /**
         * AgentTask
         * @description Моделирует задачу агента с необходимыми атрибутами для управления её жизненным циклом и распределения в мультиагентной системе.
         */
        AgentTask: {
            /** Id */
            id?: string;
            /** Capability */
            capability: string;
            /** Title */
            title: string;
            /** Instruction */
            instruction: string;
            /** Required Tools */
            required_tools?: string[];
            /** Assigned To */
            assigned_to?: string | null;
            /** @default pending */
            state: components["schemas"]["TaskExecutionState"];
            /**
             * Position
             * @default 0
             */
            position: number;
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
        };
        /**
         * AgentTaskResult
         * @description Гарантирует воспроизводимость и трассируемость результата выполнения задачи агентом, включая состояние, метрики и ошибки.
         */
        AgentTaskResult: {
            /** Task Id */
            task_id: string;
            /** Agent Name */
            agent_name: string;
            /** Capability */
            capability: string;
            state: components["schemas"]["TaskExecutionState"];
            /** Content */
            content: string;
            /** Tool Calls */
            tool_calls?: string[];
            /** Citations */
            citations?: components["schemas"]["RagCitation"][];
            usage?: components["schemas"]["UsageMetrics"];
            /** Error */
            error?: string | null;
            /**
             * Started At
             * Format: date-time
             */
            started_at?: string;
            /**
             * Finished At
             * Format: date-time
             */
            finished_at?: string;
        };
        /**
         * AgentToolResult
         * @description Гарантирует однозначное представление результата вызова инструмента агента с признаком ошибки для трассировки.
         */
        AgentToolResult: {
            /** Name */
            name?: string | null;
            /** Content */
            content: string;
            /**
             * Is Error
             * @default false
             */
            is_error: boolean;
        };
        /**
         * AgentTrace
         * @description Гарантирует полную трассировку принятия решений и изменений состояния агента в ходе обработки пользовательского запроса.
         */
        AgentTrace: {
            /** User Request */
            user_request: string;
            start_state: components["schemas"]["AgentTraceState"];
            /** Intermediate States */
            intermediate_states?: components["schemas"]["AgentTraceState"][];
            final_state: components["schemas"]["AgentTraceState"];
            /** Transition Rules */
            transition_rules?: string[];
            /** Decision Points */
            decision_points?: string[];
            /** Tool Calls */
            tool_calls?: string[];
            /** Tool Results */
            tool_results?: components["schemas"]["AgentToolResult"][];
            /** Memory Created Ids */
            memory_created_ids?: string[];
            /** Memory Updated Ids */
            memory_updated_ids?: string[];
            /** Memory Deleted Ids */
            memory_deleted_ids?: string[];
            /**
             * Loop Guard Triggered
             * @default false
             */
            loop_guard_triggered: boolean;
            /** Recursion Limit */
            recursion_limit: number;
        };
        /**
         * AgentTraceState
         * @description Хранит состояние трассировки агента с произвольными данными, обеспечивая сохранение и передачу контекста выполнения.
         */
        AgentTraceState: {
            /** Name */
            name: string;
            /** Data */
            data?: {
                [key: string]: unknown;
            };
        };
        /**
         * ApiError
         * @description Структурирует информацию об ошибках API, предоставляя машиночитаемый код, описание и корреляционный идентификатор для отладки.
         * @example {
         *       "error": "http_error",
         *       "message": "Некорректный API key.",
         *       "request_id": "f9c85fd1-59b0-4c48-a57e-b67d4019aa19"
         *     }
         */
        ApiError: {
            /**
             * Error
             * @description Машиночитаемый код ошибки.
             */
            error: string;
            /**
             * Message
             * @description Безопасное описание ошибки.
             */
            message: string;
            /**
             * Request Id
             * @description Корреляционный идентификатор запроса.
             */
            request_id?: string | null;
            /**
             * Details
             * @description Ошибки отдельных полей; заполнены для validation_error.
             */
            details?: components["schemas"]["ApiValidationDetail"][];
        };
        /**
         * ApiValidationDetail
         * @description Описывает одно поле некорректного HTTP-запроса без исходного значения.
         */
        ApiValidationDetail: {
            /** Field */
            field: string;
            /** Message */
            message: string;
            /** Type */
            type: string;
        };
        /**
         * AppAuthenticationConfig
         * @description Сообщает frontend допустимые способы входа без раскрытия секретов.
         */
        AppAuthenticationConfig: {
            /** Api Key Enabled */
            api_key_enabled: boolean;
            /** Jwt Enabled */
            jwt_enabled: boolean;
            /** User Scope Enforced */
            user_scope_enforced: boolean;
            /**
             * Api Key Header
             * @default X-API-Key
             */
            api_key_header: string;
            /**
             * Bearer Scheme
             * @default Bearer
             */
            bearer_scheme: string;
            /**
             * Browser Session Enabled
             * @default false
             */
            browser_session_enabled: boolean;
            /**
             * Csrf Header
             * @default X-CSRF-Token
             */
            csrf_header: string;
        };
        /**
         * AppConfigResponse
         * @description Предоставляет безопасный bootstrap-конфиг для web-приложения.
         */
        AppConfigResponse: {
            /**
             * Api Version
             * @default v1
             */
            api_version: string;
            /**
             * Service
             * @default engineer-support-agent
             */
            service: string;
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            features: components["schemas"]["AppFeatureFlags"];
            authentication: components["schemas"]["AppAuthenticationConfig"];
            limits: components["schemas"]["AppLimitsConfig"];
            /**
             * Openapi Url
             * @default /openapi.json
             */
            openapi_url: string;
            /**
             * Docs Url
             * @default /docs
             */
            docs_url: string;
        };
        /**
         * AppFeatureFlags
         * @description Описывает включённые backend-возможности для построения интерфейса.
         */
        AppFeatureFlags: {
            /**
             * Chat
             * @default true
             */
            chat: boolean;
            /**
             * Streaming
             * @default true
             */
            streaming: boolean;
            /**
             * Resource Management
             * @default false
             */
            resource_management: boolean;
            /**
             * Operations
             * @default false
             */
            operations: boolean;
            /** Rag */
            rag: boolean;
            /** Multi Agent */
            multi_agent: boolean;
            /** Orchestration */
            orchestration: boolean;
            /** Human Review */
            human_review: boolean;
            /** A2A */
            a2a: boolean;
            /** Mcp */
            mcp: boolean;
        };
        /**
         * AppLimitsConfig
         * @description Публикует ограничения, необходимые клиентской валидации запросов.
         */
        AppLimitsConfig: {
            /** Request Max Chars */
            request_max_chars: number;
            /**
             * Upload Max Bytes
             * @default 20971520
             */
            upload_max_bytes: number;
            /** Upload Extensions */
            upload_extensions?: string[];
            /**
             * Operation Timeout Seconds
             * @default 3600
             */
            operation_timeout_seconds: number;
            /** Max History Messages */
            max_history_messages: number;
            /** Rate Limit Enabled */
            rate_limit_enabled: boolean;
            /** Rate Limit Requests Per Minute */
            rate_limit_requests_per_minute?: number | null;
            /** Rate Limit Burst */
            rate_limit_burst?: number | null;
        };
        /**
         * ArtifactRecord
         * @description Непрозрачный идентификатор заменяет доступ по пользовательскому пути.
         */
        ArtifactRecord: {
            /** Id */
            id: string;
            /** Name */
            name: string;
            /** Size Bytes */
            size_bytes: number;
        };
        /**
         * BrowserError
         * @description Телеметрия принимает классификацию ошибки, но не текст диалога или токены.
         */
        BrowserError: {
            /** Category */
            category: string;
            /** Route */
            route: string;
            /** Request Id */
            request_id?: string | null;
        };
        /**
         * BrowserSession
         * @description CSRF-токен связывается с HttpOnly-сессией, доступной только серверу.
         */
        BrowserSession: {
            user: components["schemas"]["UserAccount"];
            /** Csrf Token */
            csrf_token: string;
            /** Expires At */
            expires_at: number;
        };
        /**
         * ChatRequest
         * @description Валидирует и нормализует входные данные запроса к агенту, гарантируя корректность идентификаторов и содержимого сообщения для обработки.
         * @example {
         *       "message": "Какие обязательные поля нужно указать в заявке и что делать, если данных недостаточно?",
         *       "session_id": "incident-42",
         *       "user_id": "engineer-1"
         *     }
         */
        ChatRequest: {
            /**
             * Message
             * @description Запрос инженера к агенту.
             */
            message: string;
            /**
             * User Id
             * @description Проверенный идентификатор пользователя для изоляции памяти.
             * @example engineer-1
             */
            user_id: string;
            /**
             * Session Id
             * @description Идентификатор текущего диалога или расследования.
             * @example incident-42
             */
            session_id: string;
        };
        /**
         * ChatResponse
         * @description Гарантирует вызывающему коду воспроизводимый результат диалога с агентом с трассировкой времени, идентификатором запроса и статусом guardrail.
         */
        ChatResponse: {
            /** Answer */
            answer: string;
            /** User Id */
            user_id: string;
            /** Session Id */
            session_id: string;
            /** Tool Calls */
            tool_calls?: string[];
            /** Citations */
            citations?: components["schemas"]["RagCitation"][];
            retrieval?: components["schemas"]["AgentRetrievalInfo"] | null;
            trace?: components["schemas"]["AgentTrace"] | null;
            /**
             * Request Id
             * @description Корреляционный идентификатор HTTP-запроса.
             */
            request_id: string;
            /**
             * Duration Ms
             * @description Полная длительность обработки запроса в миллисекундах.
             */
            duration_ms: number;
            /**
             * Guardrail Action
             * @description Решение выходного guardrail.
             * @default allow
             */
            guardrail_action: string;
            /**
             * Review Id
             * @description Human-review задача, если ответ требует проверки.
             */
            review_id?: string | null;
        };
        /**
         * ComparisonCaseResult
         * @description Гарантирует целостное сравнение работы агентов в разных режимах по качеству, времени, токенам и стоимости для одного запроса.
         */
        ComparisonCaseResult: {
            /** Id */
            id: string;
            /** Title */
            title: string;
            /** Request */
            request: string;
            single: components["schemas"]["AgentModeResult"];
            multi: components["schemas"]["AgentModeResult"];
            /** Quality Delta */
            quality_delta: number;
            /** Duration Delta Ms */
            duration_delta_ms: number;
            /** Token Delta */
            token_delta: number;
            /** Cost Delta */
            cost_delta: number;
            /** Cost Delta Rub */
            cost_delta_rub?: number | null;
        };
        /**
         * ConversationMessage
         * @description Представляет один безопасный для UI ход сохранённого диалога.
         */
        ConversationMessage: {
            /**
             * Role
             * @enum {string}
             */
            role: "user" | "assistant";
            /** Content */
            content: string;
        };
        /**
         * CreateProjectInput
         * @description Содержит проверенные данные для создания проекта, включая обязательные поля и необязательный срок, обеспечивая корректность и полноту информации при инициализации проекта.
         */
        CreateProjectInput: {
            /**
             * Project Name
             * @description Название проекта.
             */
            project_name: string;
            /**
             * Goal
             * @description Цель проекта.
             */
            goal: string;
            /**
             * Deadline
             * @description Срок проекта в свободной форме.
             */
            deadline?: string | null;
        };
        /**
         * CreateTaskInput
         * @description Определяет валидированные параметры для создания задачи в проекте, включая статус, сроки и ответственного, что гарантирует согласованность данных задачи.
         */
        CreateTaskInput: {
            /**
             * Project Name
             * @description Название проекта.
             */
            project_name: string;
            /**
             * Task Title
             * @description Название задачи.
             */
            task_title: string;
            /**
             * Status
             * @description Статус задачи.
             * @default todo
             * @enum {string}
             */
            status: "todo" | "in_progress" | "blocked" | "done";
            /**
             * Due Date
             * @description Срок задачи в свободной форме.
             */
            due_date?: string | null;
            /**
             * Owner
             * @description Ответственный.
             */
            owner?: string | null;
        };
        /**
         * CurrentPrincipalResponse
         * @description Возвращает проверенную identity и права текущего HTTP-клиента.
         */
        CurrentPrincipalResponse: {
            /** Subject */
            subject: string;
            /** Roles */
            roles?: string[];
            /** Permissions */
            permissions?: string[];
            /** Auth Method */
            auth_method: string;
        };
        /**
         * DeleteSessionResponse
         * @description Гарантирует вызывающему коду подтверждение удаления пользовательской сессии с деталями по памяти и состоянию multi-agent checkpoint.
         */
        DeleteSessionResponse: {
            /**
             * User Id
             * @description Владелец очищенной сессии.
             */
            user_id: string;
            /**
             * Session Id
             * @description Идентификатор очищенной сессии.
             */
            session_id: string;
            /**
             * Deleted Memory Count
             * @description Количество удалённых session-scoped записей памяти.
             */
            deleted_memory_count: number;
            /**
             * Runner Removed
             * @description Удалён ли AgentRunner из in-process session cache.
             */
            runner_removed: boolean;
            /**
             * Multi Agent Checkpoint Deleted
             * @description Удалён ли persistent checkpoint мультиагентного диалога.
             * @default false
             */
            multi_agent_checkpoint_deleted: boolean;
        };
        /**
         * ExecutionPlan
         * @description Гарантирует непротиворечивую и воспроизводимую структуру зависимостей шагов для корректного исполнения распределённого плана.
         */
        ExecutionPlan: {
            /**
             * Version
             * @default 1
             */
            version: number;
            pattern: components["schemas"]["OrchestrationPattern"];
            /** Steps */
            steps: components["schemas"]["PlanStep"][];
            /**
             * Reason
             * @default Первичный план
             */
            reason: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /**
         * HealthResponse
         * @description Гарантирует вызывающему коду актуальное состояние сервиса и диагностику компонентов для health-check и мониторинга.
         */
        HealthResponse: {
            /**
             * Status
             * @description Текущее состояние сервиса.
             */
            status: string;
            /**
             * Service
             * @default engineer-support-agent
             */
            service: string;
            /**
             * Details
             * @description Диагностика компонентов; заполняется readiness-проверкой.
             */
            details?: {
                [key: string]: unknown;
            };
        };
        /**
         * HumanReviewDecisionRequest
         * @description Обеспечивает проверку решения человека по обзору с обязательным статусом одобрения и опциональным комментарием для прозрачности.
         */
        HumanReviewDecisionRequest: {
            /** Approved */
            approved: boolean;
            /** Comment */
            comment?: string | null;
        };
        /**
         * HumanReviewResponse
         * @description Гарантирует вызывающему коду доступ к результату human-review задачи в согласованном формате.
         */
        HumanReviewResponse: {
            /** Id */
            id?: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at?: string;
            /**
             * Status
             * @default pending
             * @enum {string}
             */
            status: "pending" | "approved" | "rejected";
            /** User Id */
            user_id: string;
            /** Session Id */
            session_id: string;
            /** Request Id */
            request_id?: string | null;
            /** Trace Id */
            trace_id?: string | null;
            /** Prompt */
            prompt: string;
            /** Answer */
            answer: string;
            /** Reason */
            reason: string;
            /** Reviewer Id */
            reviewer_id?: string | null;
            /** Comment */
            comment?: string | null;
        };
        /**
         * IncidentInput
         * @description Инцидент привязан к текущему пользователю и выбранному диалогу.
         */
        IncidentInput: {
            /** Session Id */
            session_id: string;
            /** Title */
            title: string;
            /** Description */
            description: string;
            /**
             * Priority
             * @default medium
             * @enum {string}
             */
            priority: "low" | "medium" | "high" | "critical";
            /** Component */
            component?: string | null;
        };
        /**
         * IncidentRecord
         * @description Гарантирует целостность и валидацию данных инцидента для передачи между слоями поддержки.
         */
        IncidentRecord: {
            /** Id */
            id: string;
            /** User Id */
            user_id: string;
            /** Session Id */
            session_id: string;
            /** Title */
            title: string;
            /** Description */
            description: string;
            /**
             * Status
             * @default open
             * @enum {string}
             */
            status: "open" | "in_progress" | "resolved" | "closed";
            /**
             * Priority
             * @default medium
             * @enum {string}
             */
            priority: "low" | "medium" | "high" | "critical";
            /** Component */
            component?: string | null;
            /** Metadata */
            metadata?: {
                [key: string]: unknown;
            };
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at?: string;
        };
        /**
         * IncidentUpdate
         * @description Ограничивает переход известными статусами инженерного инцидента.
         */
        IncidentUpdate: {
            /**
             * Status
             * @enum {string}
             */
            status: "open" | "in_progress" | "resolved" | "closed";
        };
        /**
         * JobEvent
         * @description Гарантирует последовательную и воспроизводимую фиксацию событий жизненного цикла задания для аудита и реактивных обработчиков.
         */
        JobEvent: {
            /**
             * Sequence
             * @default 0
             */
            sequence: number;
            /** Job Id */
            job_id: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "submitted" | "started" | "step" | "retry" | "replanned" | "completed" | "failed" | "cancelled" | "expired" | "lease_lost";
            status: components["schemas"]["JobStatus"];
            /** Message */
            message: string;
            /** Payload */
            payload?: {
                [key: string]: unknown;
            };
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
        };
        /**
         * JobPriority
         * @description Определяет уровни приоритетов заданий, обеспечивая корректное сопоставление с числовыми значениями для управления порядком обработки в брокере очередей.
         * @enum {string}
         */
        JobPriority: "low" | "normal" | "high";
        /**
         * JobRecord
         * @description Фиксирует жизненный цикл задания с полной историей статусов, попыток, результатов и ошибок для мониторинга и восстановления.
         */
        JobRecord: {
            job: components["schemas"]["OrchestrationJob"];
            /** @default queued */
            status: components["schemas"]["JobStatus"];
            /** Task Id */
            task_id?: string | null;
            /**
             * Attempts
             * @default 0
             */
            attempts: number;
            result?: components["schemas"]["OrchestrationResult"] | null;
            /** Error */
            error?: string | null;
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at?: string;
            /** Started At */
            started_at?: string | null;
            /** Finished At */
            finished_at?: string | null;
        };
        /**
         * JobStatus
         * @description Обозначает жизненный цикл задания, позволяя определить, когда задание завершено и ресурсы можно освободить.
         * @enum {string}
         */
        JobStatus: "queued" | "running" | "retrying" | "completed" | "failed" | "cancelled" | "expired";
        /**
         * JobSubmission
         * @description Передаёт зарегистрированное задание и признак повторного запроса с тем же idempotency key.
         */
        JobSubmission: {
            record: components["schemas"]["JobRecord"];
            /**
             * Deduplicated
             * @default false
             */
            deduplicated: boolean;
        };
        /**
         * LLMRouteInfo
         * @description Гарантирует однозначную идентификацию маршрута LLM для каждой роли в мультиагентной сессии.
         */
        LLMRouteInfo: {
            /** Role */
            role: string;
            /** Profile */
            profile: string;
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /**
             * Cost Currency
             * @default RUB
             */
            cost_currency: string;
        };
        /**
         * LifecycleEvent
         * @description Гарантирует фиксирование значимых изменений состояния мультиагентного процесса с возможностью трассировки деталей.
         */
        LifecycleEvent: {
            state: components["schemas"]["AgentRunState"];
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
            /** Details */
            details?: {
                [key: string]: unknown;
            };
        };
        /**
         * LoginRequest
         * @description Ограничивает размер credentials до дорогостоящей проверки Argon2.
         */
        LoginRequest: {
            /** Username */
            username: string;
            /**
             * Password
             * Format: password
             */
            password: string;
        };
        /**
         * MemoryInput
         * @description Сохраняет происхождение и область памяти без возможности сменить владельца.
         */
        MemoryInput: {
            /** Key */
            key: string;
            /** Value */
            value: string;
            /**
             * Memory Type
             * @default fact
             * @enum {string}
             */
            memory_type: "fact" | "preference" | "task" | "summary" | "note";
            /** Session Id */
            session_id?: string | null;
            /** Tags */
            tags?: string[];
            /**
             * Importance
             * @default 3
             */
            importance: number;
            /** Ttl Seconds */
            ttl_seconds?: number | null;
        };
        /**
         * MemoryRecord
         * @description Гарантирует целостность и отслеживаемость единицы памяти агента с учётом типа, источника, важности и сроков хранения.
         */
        MemoryRecord: {
            /** Id */
            id: string;
            /** User Id */
            user_id: string;
            /** Session Id */
            session_id?: string | null;
            /**
             * Memory Type
             * @default fact
             * @enum {string}
             */
            memory_type: "fact" | "preference" | "task" | "summary" | "note";
            /** Key */
            key: string;
            /** Value */
            value: string;
            /** Tags */
            tags?: string[];
            /**
             * Importance
             * @default 3
             */
            importance: number;
            /**
             * Source
             * @default user
             * @enum {string}
             */
            source: "user" | "assistant" | "tool" | "system";
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at?: string;
            /** Last Accessed At */
            last_accessed_at?: string | null;
            /**
             * Access Count
             * @default 0
             */
            access_count: number;
            /** Ttl Seconds */
            ttl_seconds?: number | null;
            /** Metadata */
            metadata?: {
                [key: string]: unknown;
            };
        };
        /**
         * MemoryUpdate
         * @description Редактирование не позволяет менять scope или обходить policy памяти.
         */
        MemoryUpdate: {
            /** Value */
            value: string;
            /**
             * Importance
             * @default 3
             */
            importance: number;
        };
        /**
         * MultiAgentChatResponse
         * @description Гарантирует вызывающему коду полный отчёт о запуске supervisor-графа с идентификатором запроса, временем выполнения и артефактами.
         */
        MultiAgentChatResponse: {
            /** Run Id */
            run_id: string;
            /** Answer */
            answer: string;
            /** User Id */
            user_id: string;
            /** Session Id */
            session_id: string;
            /** Selected Agents */
            selected_agents?: string[];
            /** Tasks */
            tasks?: components["schemas"]["AgentTask"][];
            /** Task Results */
            task_results?: components["schemas"]["AgentTaskResult"][];
            /** Citations */
            citations?: components["schemas"]["RagCitation"][];
            /**
             * Review
             * @default
             */
            review: string;
            /**
             * History Messages Used
             * @default 0
             */
            history_messages_used: number;
            /**
             * Summary Used
             * @default false
             */
            summary_used: boolean;
            /** Llm Routes */
            llm_routes?: components["schemas"]["LLMRouteInfo"][];
            /** Lifecycle */
            lifecycle?: components["schemas"]["LifecycleEvent"][];
            usage?: components["schemas"]["UsageMetrics"];
            quality?: components["schemas"]["QualityAssessment"] | null;
            /**
             * Execution Mode
             * @default sequential
             * @enum {string}
             */
            execution_mode: "sequential" | "parallel";
            /**
             * Degraded
             * @default false
             */
            degraded: boolean;
            /**
             * Request Id
             * @description Корреляционный идентификатор HTTP-запроса.
             */
            request_id: string;
            /**
             * Duration Ms
             * @description Полная длительность supervisor-графа в миллисекундах.
             */
            duration_ms: number;
            /**
             * Run Dir
             * @description Публичный идентификатор каталога артефактов без локального пути сервера.
             */
            run_dir?: string | null;
            /**
             * Guardrail Action
             * @default allow
             */
            guardrail_action: string;
            /** Review Id */
            review_id?: string | null;
        };
        /**
         * MultiAgentCompareRequest
         * @description Расширяет запрос чата для сравнения ответов нескольких агентов, гарантируя наличие критериев оценки и требований к цитированию.
         * @example {
         *       "message": "Какие обязательные поля нужно указать в заявке и что делать, если данных недостаточно?",
         *       "session_id": "incident-42",
         *       "user_id": "engineer-1"
         *     }
         */
        MultiAgentCompareRequest: {
            /**
             * Message
             * @description Запрос инженера к агенту.
             */
            message: string;
            /**
             * User Id
             * @description Проверенный идентификатор пользователя для изоляции памяти.
             * @example engineer-1
             */
            user_id: string;
            /**
             * Session Id
             * @description Идентификатор текущего диалога или расследования.
             * @example incident-42
             */
            session_id: string;
            /**
             * Expected Terms
             * @description Термины для детерминированной оценки качества обоих режимов.
             */
            expected_terms?: string[];
            /**
             * Expected Tools
             * @description Tools, которые должен вызвать multi-agent режим.
             */
            expected_tools?: string[];
            /**
             * Require Citations
             * @description Требовать citations в single- и multi-agent ответах.
             * @default false
             */
            require_citations: boolean;
        };
        /**
         * MultiAgentCompareResponse
         * @description Гарантирует вызывающему коду воспроизводимый отчёт о сравнении двух запусков multi-agent сценариев с идентификатором запроса и временем.
         */
        MultiAgentCompareResponse: {
            /** Run Id */
            run_id?: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
            /** Provider */
            provider: string;
            /** Model */
            model: string;
            /** Cases */
            cases: components["schemas"]["ComparisonCaseResult"][];
            /** Average Single Quality */
            average_single_quality: number;
            /** Average Multi Quality */
            average_multi_quality: number;
            /** Quality Delta */
            quality_delta: number;
            /** Total Single Cost */
            total_single_cost: number;
            /** Total Multi Cost */
            total_multi_cost: number;
            /** Total Cost Delta */
            total_cost_delta: number;
            /**
             * Cost Currency
             * @default RUB
             */
            cost_currency: string;
            /** Total Single Costs By Currency */
            total_single_costs_by_currency?: {
                [key: string]: number;
            };
            /** Total Multi Costs By Currency */
            total_multi_costs_by_currency?: {
                [key: string]: number;
            };
            /** Total Single Cost Rub */
            total_single_cost_rub?: number | null;
            /** Total Multi Cost Rub */
            total_multi_cost_rub?: number | null;
            /** Total Cost Delta Rub */
            total_cost_delta_rub?: number | null;
            /** Llm Routes */
            llm_routes?: components["schemas"]["LLMRouteInfo"][];
            /** Run Dir */
            run_dir?: string | null;
            /**
             * Request Id
             * @description Корреляционный идентификатор HTTP-запроса.
             */
            request_id: string;
            /**
             * Duration Ms
             * @description Длительность двух запусков в миллисекундах.
             */
            duration_ms: number;
        };
        /**
         * OperationPage
         * @description Продолжение страницы привязано к монотонному порядку UUID внутри времени.
         */
        OperationPage: {
            /** Items */
            items: components["schemas"]["OperationRecord"][];
            /** Next Cursor */
            next_cursor?: string | null;
        };
        /**
         * OperationRecord
         * @description Публичная запись задания: параметры, состояние и безопасная диагностика.
         */
        OperationRecord: {
            /** Id */
            id: string;
            /** User Id */
            user_id: string;
            /** Profile */
            profile: string;
            /**
             * Status
             * @enum {string}
             */
            status: "queued" | "running" | "cancel_requested" | "cancelled" | "completed" | "failed" | "interrupted";
            /** Created At */
            created_at: number;
            /** Updated At */
            updated_at: number;
            request: components["schemas"]["OperationRequest"];
            /** Error */
            error?: string | null;
            /** Exit Code */
            exit_code?: number | null;
        };
        /**
         * OperationRequest
         * @description Клиент выбирает зарегистрированный профиль и идентификаторы входов.
         */
        OperationRequest: {
            /** Profile */
            profile: string;
            /** Idempotency Key */
            idempotency_key: string;
            /** Upstream Id */
            upstream_id?: string | null;
            /** Baseline Id */
            baseline_id?: string | null;
            /** Source Ids */
            source_ids?: string[];
            /** Parameters */
            parameters?: {
                [key: string]: number | string | boolean;
            };
        };
        /**
         * OrchestrationJob
         * @description Определяет контракт задания оркестрации с валидацией и нормализацией, гарантируя корректность и полноту данных для управления процессом.
         */
        OrchestrationJob: {
            /** Id */
            id?: string;
            /** User Id */
            user_id: string;
            /** Session Id */
            session_id: string;
            /** Message */
            message: string;
            /** @default sequential */
            pattern: components["schemas"]["OrchestrationPattern"];
            /** @default normal */
            priority: components["schemas"]["JobPriority"];
            /**
             * Risk Level
             * @default medium
             * @enum {string}
             */
            risk_level: "low" | "medium" | "high";
            /**
             * Quorum Size
             * @default 2
             */
            quorum_size: number;
            /** Idempotency Key */
            idempotency_key?: string | null;
            /** Deadline At */
            deadline_at?: string | null;
            /**
             * Max Plan Revisions
             * @default 2
             */
            max_plan_revisions: number;
            /** Metadata */
            metadata?: {
                [key: string]: unknown;
            };
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
        };
        /**
         * OrchestrationJobRequest
         * @description Определяет параметры задания оркестрации с валидацией приоритетов, паттернов и ограничений, обеспечивая корректное создание и управление задачами.
         * @example {
         *       "deadline_seconds": 300,
         *       "idempotency_key": "incident-42-analysis-v1",
         *       "message": "Проверь инцидент с недоступностью API, оцени риски и предложи порядок восстановления.",
         *       "pattern": "parallel",
         *       "priority": "high",
         *       "risk_level": "high",
         *       "session_id": "incident-42",
         *       "user_id": "engineer-1"
         *     }
         */
        OrchestrationJobRequest: {
            /**
             * Message
             * @description Запрос инженера к агенту.
             */
            message: string;
            /**
             * User Id
             * @description Проверенный идентификатор пользователя для изоляции памяти.
             * @example engineer-1
             */
            user_id: string;
            /**
             * Session Id
             * @description Идентификатор текущего диалога или расследования.
             * @example incident-42
             */
            session_id: string;
            /**
             * @description Паттерн выполнения: последовательный, параллельный, условный, кворум или динамическое перепланирование.
             * @default sequential
             */
            pattern: components["schemas"]["OrchestrationPattern"];
            /**
             * @description Приоритет broker-очереди.
             * @default normal
             */
            priority: components["schemas"]["JobPriority"];
            /**
             * Risk Level
             * @description Детерминированный вход для условной ветки.
             * @default medium
             * @enum {string}
             */
            risk_level: "low" | "medium" | "high";
            /**
             * Quorum Size
             * @description Число успешных голосов для кворума из трёх агентов.
             * @default 2
             */
            quorum_size: number;
            /**
             * Idempotency Key
             * @description Ключ защиты от повторной постановки одного задания.
             */
            idempotency_key?: string | null;
            /**
             * Deadline Seconds
             * @description Срок выполнения относительно момента постановки.
             */
            deadline_seconds?: number | null;
            /**
             * Max Plan Revisions
             * @description Максимальное число динамических перепланирований.
             * @default 2
             */
            max_plan_revisions: number;
            /**
             * Metadata
             * @description Дополнительный контекст задания без секретов.
             */
            metadata?: {
                [key: string]: unknown;
            };
        };
        /**
         * OrchestrationPattern
         * @description Определяет способы организации выполнения задач в оркестрации, обеспечивая гибкость и адаптивность процесса.
         * @enum {string}
         */
        OrchestrationPattern: "sequential" | "parallel" | "conditional" | "quorum" | "dynamic";
        /**
         * OrchestrationResult
         * @description Гарантирует целостное описание итогового состояния задания, включая план, шаги, ревизии, синхронизацию и ошибки.
         */
        OrchestrationResult: {
            /** Job Id */
            job_id: string;
            status: components["schemas"]["JobStatus"];
            /**
             * Answer
             * @default
             */
            answer: string;
            plan: components["schemas"]["ExecutionPlan"];
            /** Step Results */
            step_results?: components["schemas"]["StepResult"][];
            /** Revisions */
            revisions?: components["schemas"]["PlanRevision"][];
            synchronization?: components["schemas"]["SynchronizationResult"];
            /**
             * Duration Ms
             * @default 0
             */
            duration_ms: number;
            /** Error */
            error?: string | null;
            /**
             * Completed At
             * Format: date-time
             */
            completed_at?: string;
        };
        /**
         * PlanRevision
         * @description Обеспечивает прозрачную историю изменений плана с указанием причин и затронутых ролей для аудита и отката.
         */
        PlanRevision: {
            /** From Version */
            from_version: number;
            /** To Version */
            to_version: number;
            /** Reason */
            reason: string;
            /** Changed Roles */
            changed_roles?: {
                [key: string]: string;
            };
            /**
             * Created At
             * Format: date-time
             */
            created_at?: string;
        };
        /**
         * PlanStep
         * @description Моделирует шаг плана с валидацией параметров, обеспечивая корректное описание и управление зависимостями в оркестрации.
         */
        PlanStep: {
            /** Id */
            id: string;
            /** Title */
            title: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "validate" | "decision" | "agent" | "aggregate";
            /**
             * Prompt
             * @default
             */
            prompt: string;
            /** Assigned Role */
            assigned_role?: string | null;
            /** Fallback Roles */
            fallback_roles?: string[];
            /** Depends On */
            depends_on?: string[];
            /**
             * Condition
             * @default always
             * @enum {string}
             */
            condition: "always" | "low_or_medium_risk" | "high_risk";
            /**
             * Required
             * @default true
             */
            required: boolean;
            /**
             * Timeout Seconds
             * @default 60
             */
            timeout_seconds: number;
        };
        /**
         * PublicProfile
         * @description Браузер видит возможности и редактируемые поля, но не пути сервера.
         */
        PublicProfile: {
            /** Id */
            id: string;
            /** Title */
            title: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "prepare" | "chunk" | "embed" | "index" | "evaluation" | "scenarios" | "tuning-inspect" | "tuning-validate" | "tuning-baseline" | "tuning-train" | "tuning-evaluate" | "tuning-compare";
            /** Parameters */
            parameters: string[];
            /** Parameter Schema */
            parameter_schema?: {
                [key: string]: {
                    [key: string]: unknown;
                };
            };
            /** Worker Ready */
            worker_ready: boolean;
        };
        /**
         * QualityAssessment
         * @description Гарантирует прозрачную оценку качества результата мультиагентной работы с возможностью автоматизированной проверки критериев.
         */
        QualityAssessment: {
            /** Score */
            score: number;
            /** Checks */
            checks?: {
                [key: string]: boolean;
            };
            /** Notes */
            notes?: string[];
        };
        /**
         * QueueStatus
         * @description Хранит состояние очереди заданий и информацию о рабочих процессах для мониторинга и управления распределённой оркестрацией.
         */
        QueueStatus: {
            /**
             * Backend
             * @enum {string}
             */
            backend: "inline" | "celery";
            /** Ready */
            ready: boolean;
            /** Status Counts */
            status_counts?: {
                [key: string]: number;
            };
            /** Workers */
            workers?: {
                [key: string]: unknown;
            };
            /** Error */
            error?: string | null;
        };
        /**
         * RagCitation
         * @description Содержит метаданные и оценку релевантности фрагмента источника для обеспечения точной атрибуции в онлайн-RAG.
         */
        RagCitation: {
            /** Reference */
            reference: string;
            /** Point Id */
            point_id: string;
            /** Chunk Id */
            chunk_id: string;
            /** Document Id */
            document_id?: string | null;
            /** Source */
            source?: string | null;
            /** Section */
            section?: string | null;
            /** Position */
            position?: number | null;
            /** Score */
            score: number;
            /** Excerpt */
            excerpt: string;
        };
        /** ResourcePage[IncidentRecord] */
        ResourcePage_IncidentRecord_: {
            /** Items */
            items: components["schemas"]["IncidentRecord"][];
            /** Next Cursor */
            next_cursor?: string | null;
        };
        /** ResourcePage[JobRecord] */
        ResourcePage_JobRecord_: {
            /** Items */
            items: components["schemas"]["JobRecord"][];
            /** Next Cursor */
            next_cursor?: string | null;
        };
        /** ResourcePage[MemoryRecord] */
        ResourcePage_MemoryRecord_: {
            /** Items */
            items: components["schemas"]["MemoryRecord"][];
            /** Next Cursor */
            next_cursor?: string | null;
        };
        /** ResourcePage[dict[str, Any]] */
        ResourcePage_dict_str__Any__: {
            /** Items */
            items: {
                [key: string]: unknown;
            }[];
            /** Next Cursor */
            next_cursor?: string | null;
        };
        /**
         * SearchInput
         * @description Тестовый retrieval использует тот же embedding-профиль, что активный RAG.
         */
        SearchInput: {
            /** Query */
            query: string;
            /**
             * Top K
             * @default 5
             */
            top_k: number;
        };
        /**
         * SecurityAuditEvent
         * @description Гарантирует целостность и полноту данных о событии безопасности для аудита и расследования инцидентов.
         */
        SecurityAuditEvent: {
            /** Id */
            id?: string;
            /**
             * Occurred At
             * Format: date-time
             */
            occurred_at?: string;
            /** Event Type */
            event_type: string;
            /** Action */
            action: string;
            /** Principal Id */
            principal_id?: string | null;
            /** Role */
            role?: string | null;
            /** User Id */
            user_id?: string | null;
            /** Session Id */
            session_id?: string | null;
            /** Request Id */
            request_id?: string | null;
            /** Trace Id */
            trace_id?: string | null;
            /** Details */
            details?: {
                [key: string]: unknown;
            };
        };
        /**
         * SecurityAuditResponse
         * @description Гарантирует вызывающему коду полный список событий аудита безопасности для последующего анализа или отображения.
         */
        SecurityAuditResponse: {
            /** Events */
            events: components["schemas"]["SecurityAuditEvent"][];
        };
        /**
         * SessionListResponse
         * @description Возвращает страницу диалогов пользователя и непрозрачный cursor.
         */
        SessionListResponse: {
            /** User Id */
            user_id: string;
            /** Items */
            items?: components["schemas"]["SessionSummary"][];
            /** Next Cursor */
            next_cursor?: string | null;
            /** Limit */
            limit: number;
        };
        /**
         * SessionResponse
         * @description Гарантирует вызывающему коду согласованный снимок пользовательской сессии с памятью, инцидентами и историей multi-agent диалога.
         */
        SessionResponse: {
            /**
             * User Id
             * @description Владелец сессии.
             */
            user_id: string;
            /**
             * Session Id
             * @description Идентификатор сессии.
             */
            session_id: string;
            /**
             * Updated At
             * @description Время последнего сохранённого хода диалога.
             */
            updated_at?: string | null;
            /**
             * Messages
             * @description Сохранённые пользовательские и агентские сообщения.
             */
            messages?: components["schemas"]["ConversationMessage"][];
            /**
             * Memory
             * @description Доступные пользователю записи долговременной памяти.
             */
            memory?: {
                [key: string]: unknown;
            }[];
            /**
             * Incidents
             * @description Инциденты пользователя, связанные с этой сессией.
             */
            incidents?: {
                [key: string]: unknown;
            }[];
            /**
             * Multi Agent History
             * @description Последние сообщения persistent multi-agent checkpoint этой сессии.
             */
            multi_agent_history?: {
                [key: string]: unknown;
            }[];
        };
        /**
         * SessionSummary
         * @description Представляет одну строку списка диалогов для навигации frontend.
         */
        SessionSummary: {
            /** Session Id */
            session_id: string;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at: string;
            /** Message Count */
            message_count: number;
            /** Preview */
            preview: string;
        };
        /**
         * SourceRecord
         * @description Источник хранится отдельно от результатов преобразования и имеет SHA-256.
         */
        SourceRecord: {
            /** Id */
            id: string;
            /** Name */
            name: string;
            /** Sha256 */
            sha256: string;
            /** Size Bytes */
            size_bytes: number;
            /** Created At */
            created_at: number;
        };
        /**
         * StepResult
         * @description Фиксирует результат выполнения шага с инвариантом однозначного статуса и полной трассировкой попыток и ошибок.
         */
        StepResult: {
            /** Step Id */
            step_id: string;
            status: components["schemas"]["StepStatus"];
            /**
             * Output
             * @default
             */
            output: string;
            /** Error */
            error?: string | null;
            /** Assigned Role */
            assigned_role?: string | null;
            /** Vote */
            vote?: ("approve" | "reject" | "abstain") | null;
            /**
             * Retryable
             * @default false
             */
            retryable: boolean;
            /**
             * Attempt
             * @default 1
             */
            attempt: number;
            /**
             * Started At
             * Format: date-time
             */
            started_at?: string;
            /**
             * Finished At
             * Format: date-time
             */
            finished_at?: string;
            /** Metadata */
            metadata?: {
                [key: string]: unknown;
            };
        };
        /**
         * StepStatus
         * @description Отражает состояние отдельного шага в процессе оркестрации, влияя на логику выполнения и обработку ошибок.
         * @enum {string}
         */
        StepStatus: "pending" | "running" | "completed" | "failed" | "skipped" | "timed_out" | "cancelled";
        /**
         * SynchronizationResult
         * @description Гарантирует однозначную фиксацию достижения кворума и консенсуса между участниками распределённой оркестрации.
         */
        SynchronizationResult: {
            /**
             * Required
             * @default 0
             */
            required: number;
            /**
             * Received
             * @default 0
             */
            received: number;
            /**
             * Successful
             * @default 0
             */
            successful: number;
            /**
             * Quorum Reached
             * @default true
             */
            quorum_reached: boolean;
            /**
             * Consensus
             * @default undetermined
             * @enum {string}
             */
            consensus: "approve" | "reject" | "undetermined";
            /** Cancelled Steps */
            cancelled_steps?: string[];
        };
        /**
         * TaskExecutionState
         * @description Отражает текущее состояние выполнения задачи агентом, гарантируя прозрачность и отслеживаемость прогресса.
         * @enum {string}
         */
        TaskExecutionState: "pending" | "running" | "completed" | "failed" | "timed_out";
        /**
         * UpdateTaskStatusInput
         * @description Обеспечивает проверку данных для обновления статуса задачи, гарантируя корректное изменение состояния задачи в рамках проекта.
         */
        UpdateTaskStatusInput: {
            /**
             * Project Name
             * @description Название проекта.
             */
            project_name: string;
            /**
             * Task Title
             * @description Название задачи.
             */
            task_title: string;
            /**
             * Status
             * @description Новый статус задачи.
             * @enum {string}
             */
            status: "todo" | "in_progress" | "blocked" | "done";
        };
        /**
         * UsageMetrics
         * @description Хранит токены, исходные расходы по валютам и их эквивалент в RUB.
         */
        UsageMetrics: {
            /**
             * Llm Calls
             * @default 0
             */
            llm_calls: number;
            /**
             * Input Tokens
             * @default 0
             */
            input_tokens: number;
            /**
             * Output Tokens
             * @default 0
             */
            output_tokens: number;
            /**
             * Estimated Tokens
             * @default 0
             */
            estimated_tokens: number;
            /**
             * Tool Calls
             * @default 0
             */
            tool_calls: number;
            /**
             * Duration Ms
             * @default 0
             */
            duration_ms: number;
            /**
             * Estimated Cost
             * @default 0
             */
            estimated_cost: number;
            /**
             * Estimated Cost Currency
             * @default RUB
             */
            estimated_cost_currency: string;
            /** Costs By Currency */
            costs_by_currency?: {
                [key: string]: number;
            };
            /**
             * Estimated Cost Rub
             * @default 0
             */
            estimated_cost_rub: number | null;
            /** Exchange Rates To Rub */
            exchange_rates_to_rub?: {
                [key: string]: number;
            };
            /** Exchange Rate Dates */
            exchange_rate_dates?: {
                [key: string]: string;
            };
            /** Exchange Rate Source */
            exchange_rate_source?: string | null;
            /**
             * Exchange Rate Stale
             * @default false
             */
            exchange_rate_stale: boolean;
            /** Currency Conversion Errors */
            currency_conversion_errors?: string[];
        };
        /**
         * UserAccount
         * @description Публичная часть учётной записи без пароля и токенов сессии.
         */
        UserAccount: {
            /** Username */
            username: string;
            /** Display Name */
            display_name: string;
            /** Roles */
            roles: string[];
            /** Active */
            active: boolean;
        };
        /**
         * UserCreate
         * @description Создание пользователя доступно администратору, роли задаются явно.
         */
        UserCreate: {
            /** Username */
            username: string;
            /**
             * Password
             * Format: password
             */
            password: string;
            /** Display Name */
            display_name: string;
            /** Roles */
            roles?: string[];
        };
        /**
         * UserUpdate
         * @description Изменение ролей или пароля отзывает все ранее выданные сессии.
         */
        UserUpdate: {
            /** Roles */
            roles?: string[] | null;
            /** Active */
            active?: boolean | null;
            /** Password */
            password?: string | null;
        };
        /** ValidationError */
        ValidationError: {
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
            /** Input */
            input?: unknown;
            /** Context */
            ctx?: Record<string, never>;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    app_config_v1_app_config_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AppConfigResponse"];
                };
            };
        };
    };
    auth_me_v1_auth_me_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CurrentPrincipalResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Роль не имеет требуемого разрешения. */
            403: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    chat_v1_chat_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ChatRequest"];
            };
        };
        responses: {
            /** @description Итоговый ответ агента и диагностика выполнения. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ChatResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Размер запроса или сообщения превышает установленный предел. */
            413: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    multi_agent_chat_v1_multi_agent_chat_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ChatRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MultiAgentChatResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Размер запроса или сообщения превышает установленный предел. */
            413: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    compare_agents_v1_multi_agent_compare_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["MultiAgentCompareRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MultiAgentCompareResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Размер запроса или сообщения превышает установленный предел. */
            413: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    get_multi_agent_run_v1_multi_agent_runs__run_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    jobs_v1_orchestration_jobs_get: {
        parameters: {
            query?: {
                cursor?: string;
                limit?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ResourcePage_JobRecord_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    submit_orchestration_job_v1_orchestration_jobs_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["OrchestrationJobRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobSubmission"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Размер запроса или сообщения превышает установленный предел. */
            413: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Очередь заполнена: сработал backpressure. */
            429: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    get_orchestration_job_v1_orchestration_jobs__job_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobRecord"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Задание или ресурс не найден. */
            404: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    cancel_orchestration_job_v1_orchestration_jobs__job_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobRecord"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Задание или ресурс не найден. */
            404: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    get_orchestration_events_v1_orchestration_jobs__job_id__events_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                job_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["JobEvent"][];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Задание или ресурс не найден. */
            404: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    orchestration_queue_status_v1_orchestration_queues_status_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["QueueStatus"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    chat_stream_v1_chat_stream_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ChatRequest"];
            };
        };
        responses: {
            /** @description Поток Server-Sent Events. */
            200: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                    /**
                     * @example event: started
                     *     data: {"request_id":"..."}
                     *
                     *     event: result
                     *     data: {"answer":"...","request_id":"..."}
                     */
                    "text/event-stream": unknown;
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Размер запроса или сообщения превышает установленный предел. */
            413: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    list_sessions_v1_sessions_get: {
        parameters: {
            query: {
                user_id: string;
                limit?: number;
                cursor?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionListResponse"];
                };
            };
            /** @description Запрос отклонён guardrail-проверкой. */
            400: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    get_session_v1_sessions__session_id__get: {
        parameters: {
            query: {
                user_id: string;
            };
            header?: never;
            path: {
                session_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SessionResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    delete_session_v1_sessions__session_id__delete: {
        parameters: {
            query: {
                user_id: string;
            };
            header?: never;
            path: {
                session_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["DeleteSessionResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Запрос не соответствует OpenAPI-схеме. */
            422: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Непредвиденная ошибка выполнения агента. */
            500: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    list_reviews_v1_reviews_get: {
        parameters: {
            query?: {
                review_status?: string | null;
                limit?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HumanReviewResponse"][];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Роль не имеет требуемого разрешения. */
            403: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    decide_review_v1_reviews__review_id__decision_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                review_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["HumanReviewDecisionRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HumanReviewResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Роль не имеет требуемого разрешения. */
            403: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Задание или ресурс не найден. */
            404: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    security_audit_v1_security_audit_get: {
        parameters: {
            query?: {
                limit?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SecurityAuditResponse"];
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Роль не имеет требуемого разрешения. */
            403: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    health_health_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthResponse"];
                };
            };
        };
    };
    ready_ready_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthResponse"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    metrics_metrics_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Метрики в Prometheus text exposition format. */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                    /** @example support_agent_requests_total{status="200"} 1.0 */
                    "text/plain": unknown;
                };
            };
            /** @description Учётные данные отсутствуют, недействительны или срок сессии истёк. */
            401: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Роль не имеет требуемого разрешения. */
            403: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
            /** @description Сервис, LLM или RAG временно не готов к обработке запроса. */
            503: {
                headers: {
                    /** @description Корреляционный идентификатор запроса. */
                    "X-Request-ID"?: string;
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ApiError"];
                };
            };
        };
    };
    login_v1_auth_login_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["LoginRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BrowserSession"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    session_v1_auth_session_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BrowserSession"];
                };
            };
        };
    };
    logout_v1_auth_logout_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
        };
    };
    users_v1_admin_users_get: {
        parameters: {
            query?: {
                after?: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UserAccount"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_user_v1_admin_users_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UserCreate"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UserAccount"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_user_v1_admin_users__username__patch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                username: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UserUpdate"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["UserAccount"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    memories_v1_memories_get: {
        parameters: {
            query?: {
                user_id?: string | null;
                cursor?: string;
                limit?: number;
                session_id?: string | null;
                memory_type?: ("fact" | "preference" | "task" | "summary" | "note") | null;
                projects_only?: boolean;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ResourcePage_MemoryRecord_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    save_memory_v1_memories_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["MemoryInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MemoryRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    memory_detail_v1_memories__memory_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                memory_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MemoryRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_memory_v1_memories__memory_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                memory_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_memory_v1_memories__memory_id__patch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                memory_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["MemoryUpdate"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["MemoryRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    incidents_v1_incidents_get: {
        parameters: {
            query?: {
                cursor?: string;
                limit?: number;
                incident_status?: ("open" | "in_progress" | "resolved" | "closed") | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ResourcePage_IncidentRecord_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_incident_v1_incidents_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["IncidentInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IncidentRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    incident_v1_incidents__incident_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                incident_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IncidentRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_incident_v1_incidents__incident_id__patch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                incident_id: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["IncidentUpdate"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["IncidentRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    projects_v1_projects_get: {
        parameters: {
            query?: {
                cursor?: string;
                limit?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ResourcePage_MemoryRecord_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_project_v1_projects_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CreateProjectInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    create_task_v1_projects_tasks_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CreateTaskInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    update_task_v1_projects_tasks_patch: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["UpdateTaskStatusInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    runs_v1_multi_agent_runs_get: {
        parameters: {
            query?: {
                cursor?: string;
                limit?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ResourcePage_dict_str__Any__"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    profiles_v1_operations_profiles_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["PublicProfile"][];
                };
            };
        };
    };
    operations_v1_operations_get: {
        parameters: {
            query?: {
                cursor?: string | null;
                limit?: number;
                operation_status?: string | null;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OperationPage"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    submit_v1_operations_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["OperationRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OperationRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    get_operation_v1_operations__operation_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                operation_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OperationRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_v1_operations__operation_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                operation_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["OperationRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    artifacts_v1_operations__operation_id__artifacts_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                operation_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ArtifactRecord"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    download_artifact_v1_operations__operation_id__artifacts__artifact_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                operation_id: string;
                artifact_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    preview_artifact_v1_operations__operation_id__preview__artifact_id__get: {
        parameters: {
            query?: {
                offset?: number;
                limit?: number;
            };
            header?: never;
            path: {
                operation_id: string;
                artifact_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    logs_v1_operations__operation_id__logs_get: {
        parameters: {
            query?: {
                offset?: number;
            };
            header?: never;
            path: {
                operation_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    sources_v1_knowledge_sources_get: {
        parameters: {
            query?: {
                after?: string;
                limit?: number;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SourceRecord"][];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    upload_v1_knowledge_sources_post: {
        parameters: {
            query: {
                filename: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/octet-stream": string;
            };
        };
        responses: {
            /** @description Successful Response */
            201: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["SourceRecord"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    download_source_v1_knowledge_sources__source_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                source_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    delete_source_v1_knowledge_sources__source_id__delete: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                source_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    search_v1_knowledge_search_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["SearchInput"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    integrations_v1_integrations_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": {
                        [key: string]: unknown;
                    };
                };
            };
        };
    };
    browser_error_v1_telemetry_browser_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["BrowserError"];
            };
        };
        responses: {
            /** @description Successful Response */
            204: {
                headers: {
                    [name: string]: unknown;
                };
                content?: never;
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
