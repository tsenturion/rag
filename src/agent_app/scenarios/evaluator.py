"""Проверка критериев сценария для проверочных сценариев агента."""

from __future__ import annotations

from agent_app.models import AgentResponse, MemoryRecord
from agent_app.scenarios.models import (
    MemoryExpectation,
    ScenarioCheck,
    ScenarioCriteria,
)


class ScenarioEvaluator:
    """Инкапсулирует проверку соответствия поведения агента заданным критериям сценариев и формирует исчерпывающий отчёт о прохождении."""

    def evaluate(
        self,
        *,
        criteria: ScenarioCriteria,
        response: AgentResponse,
        memory_records: list[MemoryRecord],
    ) -> list[ScenarioCheck]:
        """Гарантирует воспроизводимую проверку ответа агента по критериям сценария и возвращает полный список результатов проверок."""
        checks: list[ScenarioCheck] = []
        answer = response.answer.lower()
        tool_calls = response.tool_calls
        trace = response.trace
        tool_error_count = (
            sum(1 for result in trace.tool_results if result.is_error)
            if trace is not None
            else 0
        )
        memory_created_count = len(trace.memory_created_ids) if trace is not None else 0
        memory_updated_count = len(trace.memory_updated_ids) if trace is not None else 0
        loop_guard_triggered = bool(trace and trace.loop_guard_triggered)
        retrieval_used = response.retrieval is not None
        retrieval_succeeded = bool(
            response.retrieval
            and response.retrieval.status == "ok"
            and response.citations
        )
        protocol_markers = (
            "recipient_name",
            "functions.",
            "<tool_call",
            '"tool_calls"',
        )
        checks.extend(
            [
                ScenarioCheck(
                    name="answer_present",
                    passed=bool(response.answer.strip()),
                    details=f"Длина ответа: {len(response.answer.strip())}",
                ),
                ScenarioCheck(
                    name="answer_protocol_clean",
                    passed=not any(marker in answer for marker in protocol_markers),
                    details="Финальный ответ не должен содержать разметку tool call.",
                ),
            ]
        )

        for expected in criteria.answer_contains:
            checks.append(
                ScenarioCheck(
                    name=f"answer_contains:{expected}",
                    passed=expected.lower() in answer,
                    details=f"Ожидался фрагмент ответа: {expected}",
                )
            )
        for forbidden in criteria.answer_not_contains:
            checks.append(
                ScenarioCheck(
                    name=f"answer_not_contains:{forbidden}",
                    passed=forbidden.lower() not in answer,
                    details=f"Запрещённый фрагмент ответа: {forbidden}",
                )
            )
        for tool in criteria.expected_tools:
            checks.append(
                ScenarioCheck(
                    name=f"expected_tool:{tool}",
                    passed=tool in tool_calls,
                    details=f"Вызванные tools: {tool_calls}",
                )
            )
        for tool in criteria.forbidden_tools:
            checks.append(
                ScenarioCheck(
                    name=f"forbidden_tool:{tool}",
                    passed=tool not in tool_calls,
                    details=f"Вызванные tools: {tool_calls}",
                )
            )
        checks.append(
            ScenarioCheck(
                name="min_tool_calls",
                passed=len(tool_calls) >= criteria.min_tool_calls,
                details=f"tool calls: {len(tool_calls)}, минимум: {criteria.min_tool_calls}",
            )
        )
        if criteria.max_tool_calls is not None:
            checks.append(
                ScenarioCheck(
                    name="max_tool_calls",
                    passed=len(tool_calls) <= criteria.max_tool_calls,
                    details=f"tool calls: {len(tool_calls)}, максимум: {criteria.max_tool_calls}",
                )
            )
        if not criteria.allow_tool_errors:
            checks.append(
                ScenarioCheck(
                    name="no_tool_errors",
                    passed=tool_error_count == 0,
                    details=f"Ошибок tools: {tool_error_count}",
                )
            )
        if criteria.require_loop_guard:
            checks.append(
                ScenarioCheck(
                    name="loop_guard_triggered",
                    passed=loop_guard_triggered,
                    details=f"loop_guard_triggered={loop_guard_triggered}",
                )
            )
        if criteria.forbid_loop_guard:
            checks.append(
                ScenarioCheck(
                    name="loop_guard_not_triggered",
                    passed=not loop_guard_triggered,
                    details=f"loop_guard_triggered={loop_guard_triggered}",
                )
            )
        if criteria.require_citations:
            checks.append(
                ScenarioCheck(
                    name="citations_present",
                    passed=bool(response.citations),
                    details=f"Количество citations: {len(response.citations)}",
                )
            )
        if criteria.forbid_citations:
            checks.append(
                ScenarioCheck(
                    name="citations_absent",
                    passed=not response.citations,
                    details=f"Количество citations: {len(response.citations)}",
                )
            )
        if criteria.require_retrieval:
            checks.append(
                ScenarioCheck(
                    name="retrieval_used",
                    passed=retrieval_used,
                    details=f"retrieval={response.retrieval}",
                )
            )
        if criteria.require_retrieval_success:
            checks.append(
                ScenarioCheck(
                    name="retrieval_succeeded",
                    passed=retrieval_succeeded,
                    details=f"retrieval={response.retrieval}",
                )
            )
        if criteria.expected_retrieval_status is not None:
            actual_status = response.retrieval.status if response.retrieval else None
            checks.append(
                ScenarioCheck(
                    name="retrieval_status",
                    passed=actual_status == criteria.expected_retrieval_status,
                    details=(
                        f"retrieval status: {actual_status}, ожидается: "
                        f"{criteria.expected_retrieval_status}"
                    ),
                )
            )
        if criteria.require_memory_created:
            checks.append(
                ScenarioCheck(
                    name="memory_created",
                    passed=memory_created_count > 0,
                    details=f"Создано записей памяти: {memory_created_count}",
                )
            )
        if criteria.require_memory_updated:
            checks.append(
                ScenarioCheck(
                    name="memory_updated",
                    passed=memory_updated_count > 0,
                    details=f"Обновлено записей памяти: {memory_updated_count}",
                )
            )
        for expectation in criteria.memory_contains:
            checks.append(
                ScenarioCheck(
                    name="memory_contains",
                    passed=self._memory_matches(memory_records, expectation),
                    details=expectation.model_dump_json(exclude_none=True),
                )
            )
        for expectation in criteria.memory_not_contains:
            checks.append(
                ScenarioCheck(
                    name="memory_not_contains",
                    passed=not self._memory_matches(memory_records, expectation),
                    details=expectation.model_dump_json(exclude_none=True),
                )
            )
        return checks

    @staticmethod
    def _memory_matches(
        memory_records: list[MemoryRecord],
        expectation: MemoryExpectation,
    ) -> bool:
        """Гарантирует, что хотя бы одна запись памяти соответствует всем условиям ожидания сценария, включая тип, ключ, значение и теги."""
        for record in memory_records:
            if (
                expectation.memory_type
                and record.memory_type != expectation.memory_type
            ):
                continue
            if (
                expectation.key_contains
                and expectation.key_contains.lower() not in record.key.lower()
            ):
                continue
            if (
                expectation.value_contains
                and expectation.value_contains.lower() not in record.value.lower()
            ):
                continue
            if expectation.tag and expectation.tag not in record.tags:
                continue
            return True
        return False
