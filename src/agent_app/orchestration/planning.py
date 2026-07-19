"""Построение и пересмотр плана для распределённой оркестрации."""

from __future__ import annotations

from typing import Literal

from agent_app.orchestration.models import (
    ExecutionPlan,
    OrchestrationJob,
    OrchestrationPattern,
    PlanRevision,
    PlanStep,
    StepResult,
    StepStatus,
)


class OrchestrationPlanBuilder:
    """Строит ограниченные планы из разрешённых сервером сценариев."""

    def __init__(self, *, step_timeout_seconds: float = 60.0):
        """Настраивает параметры построителя плана, обеспечивая готовность к созданию планов с заданным таймаутом на шаг."""
        self.step_timeout_seconds = step_timeout_seconds

    def build(self, job: OrchestrationJob) -> ExecutionPlan:
        """Формирует план исполнения на основе шаблона задания, обеспечивая соответствие структуры плана выбранной оркестрационной стратегии."""
        builders = {
            OrchestrationPattern.SEQUENTIAL: self._sequential,
            OrchestrationPattern.PARALLEL: self._parallel,
            OrchestrationPattern.CONDITIONAL: self._conditional,
            OrchestrationPattern.QUORUM: self._quorum,
            OrchestrationPattern.DYNAMIC: self._dynamic,
        }
        return ExecutionPlan(
            pattern=job.pattern,
            steps=builders[job.pattern](job),
            reason=f"Шаблон оркестрации: {job.pattern.value}",
        )

    def replan(
        self,
        plan: ExecutionPlan,
        results: list[StepResult],
    ) -> tuple[ExecutionPlan, PlanRevision]:
        """Обновляет план с учётом неудачных шагов, переназначая роли для повышения устойчивости и обеспечивая версионирование изменений."""
        failed = {
            result.step_id: result
            for result in results
            if result.status in {StepStatus.FAILED, StepStatus.TIMED_OUT}
        }
        changed_roles: dict[str, str] = {}
        steps: list[PlanStep] = []
        for step in plan.steps:
            if step.id not in failed or step.kind != "agent":
                steps.append(step)
                continue
            fallback = next(
                (role for role in step.fallback_roles if role != step.assigned_role),
                None,
            )
            if fallback is None:
                steps.append(step)
                continue
            changed_roles[step.id] = fallback
            steps.append(
                step.model_copy(
                    update={
                        "assigned_role": fallback,
                        "fallback_roles": [
                            role for role in step.fallback_roles if role != fallback
                        ],
                    }
                )
            )
        reason = (
            "Роли изменены после ошибки: "
            + ", ".join(f"{key}→{value}" for key, value in changed_roles.items())
            if changed_roles
            else "Повтор плана после временной ошибки без смены роли"
        )
        updated = plan.model_copy(
            update={
                "version": plan.version + 1,
                "steps": steps,
                "reason": reason,
            }
        )
        return updated, PlanRevision(
            from_version=plan.version,
            to_version=updated.version,
            reason=reason,
            changed_roles=changed_roles,
        )

    def _base(self) -> list[PlanStep]:
        """Создаёт базовый шаг валидации, который гарантирует проверку входных данных перед выполнением основных этапов плана."""
        return [
            PlanStep(
                id="validate",
                title="Проверка входного задания",
                kind="validate",
                timeout_seconds=self.step_timeout_seconds,
            )
        ]

    def _sequential(self, job: OrchestrationJob) -> list[PlanStep]:
        """Формирует последовательный план шагов с чёткими зависимостями, обеспечивая упорядоченное выполнение и агрегирование результатов."""
        del job
        return [
            *self._base(),
            self._agent(
                "analysis",
                "Основной инженерный анализ",
                "Выполни инженерный анализ запроса и предложи проверяемое решение.",
                role="diagnostics_agent",
                depends_on=["validate"],
            ),
            PlanStep(
                id="aggregate",
                title="Формирование ответа",
                kind="aggregate",
                depends_on=["analysis"],
                timeout_seconds=self.step_timeout_seconds,
            ),
        ]

    def _parallel(self, job: OrchestrationJob) -> list[PlanStep]:
        """Создаёт параллельный план с независимыми агентскими шагами и последующим объединением результатов для ускорения обработки."""
        del job
        agent_ids = ["diagnostics", "knowledge", "risks"]
        return [
            *self._base(),
            self._agent(
                "diagnostics",
                "Диагностика",
                "Проведи диагностику и перечисли проверяемые шаги.",
                role="diagnostics_agent",
                depends_on=["validate"],
            ),
            self._agent(
                "knowledge",
                "Проверка базы знаний",
                "Найди подтверждённые сведения, runbook и ограничения.",
                role="knowledge_agent",
                depends_on=["validate"],
            ),
            self._agent(
                "risks",
                "Анализ рисков",
                "Проверь риски, ложные предположения и условия эскалации.",
                role="critic_agent",
                depends_on=["validate"],
            ),
            PlanStep(
                id="aggregate",
                title="Объединение параллельных результатов",
                kind="aggregate",
                depends_on=agent_ids,
                timeout_seconds=self.step_timeout_seconds,
            ),
        ]

    def _conditional(self, job: OrchestrationJob) -> list[PlanStep]:
        """Создаёт план с детерминированным ветвлением по уровню риска, гарантируя выбор безопасной и соответствующей стратегии обработки запроса."""
        del job
        return [
            *self._base(),
            PlanStep(
                id="risk-decision",
                title="Детерминированное ветвление по риску",
                kind="decision",
                depends_on=["validate"],
                timeout_seconds=self.step_timeout_seconds,
            ),
            self._agent(
                "standard-analysis",
                "Стандартная обработка",
                "Подготовь практическое решение стандартного инженерного запроса.",
                role="diagnostics_agent",
                depends_on=["risk-decision"],
                condition="low_or_medium_risk",
            ),
            self._agent(
                "high-risk-analysis",
                "Обработка высокого риска",
                "Проведи консервативный анализ высокого риска. Не предлагай опасных "
                "действий без проверки и явно укажи условия ручного согласования.",
                role="critic_agent",
                depends_on=["risk-decision"],
                condition="high_risk",
            ),
            PlanStep(
                id="aggregate",
                title="Формирование результата выбранной ветки",
                kind="aggregate",
                depends_on=["standard-analysis", "high-risk-analysis"],
                timeout_seconds=self.step_timeout_seconds,
            ),
        ]

    def _quorum(self, job: OrchestrationJob) -> list[PlanStep]:
        """Формирует план с голосованием нескольких агентов для коллективной оценки решения, обеспечивая согласованность и проверку качества результата."""
        del job
        voters = ["vote-diagnostics", "vote-knowledge", "vote-critic"]
        return [
            *self._base(),
            self._agent(
                "vote-diagnostics",
                "Голос диагностики",
                "Оцени решение как инженер диагностики и заверши ответ маркером "
                "[approve], [reject] или [abstain].",
                role="diagnostics_agent",
                depends_on=["validate"],
            ),
            self._agent(
                "vote-knowledge",
                "Голос базы знаний",
                "Проверь подтверждаемость решения и заверши ответ маркером "
                "[approve], [reject] или [abstain].",
                role="knowledge_agent",
                depends_on=["validate"],
            ),
            self._agent(
                "vote-critic",
                "Голос критика",
                "Проверь безопасность решения и заверши ответ маркером "
                "[approve], [reject] или [abstain].",
                role="critic_agent",
                depends_on=["validate"],
            ),
            PlanStep(
                id="aggregate",
                title="Кворум и итог",
                kind="aggregate",
                depends_on=voters,
                timeout_seconds=self.step_timeout_seconds,
            ),
        ]

    def _dynamic(self, job: OrchestrationJob) -> list[PlanStep]:
        """Строит адаптивный план, который учитывает доступные инструменты и данные, обеспечивая гибкость и надёжность обработки запроса."""
        del job
        return [
            *self._base(),
            self._agent(
                "adaptive-analysis",
                "Адаптивный анализ",
                "Реши запрос с учётом доступных tools и подтверждённых данных.",
                role="diagnostics_agent",
                fallback_roles=["knowledge_agent", "critic_agent"],
                depends_on=["validate"],
            ),
            PlanStep(
                id="aggregate",
                title="Формирование ответа адаптивного плана",
                kind="aggregate",
                depends_on=["adaptive-analysis"],
                timeout_seconds=self.step_timeout_seconds,
            ),
        ]

    def _agent(
        self,
        step_id: str,
        title: str,
        prompt: str,
        *,
        role: str,
        depends_on: list[str],
        fallback_roles: list[str] | None = None,
        condition: Literal["always", "low_or_medium_risk", "high_risk"] = "always",
    ) -> PlanStep:
        """Создаёт шаг плана, назначенный конкретной роли с условиями и зависимостями, гарантируя корректное распределение задач в оркестрации."""
        return PlanStep(
            id=step_id,
            title=title,
            kind="agent",
            prompt=prompt,
            assigned_role=role,
            fallback_roles=fallback_roles or [],
            depends_on=depends_on,
            condition=condition,
            timeout_seconds=self.step_timeout_seconds,
        )
