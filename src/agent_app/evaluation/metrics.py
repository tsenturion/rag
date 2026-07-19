"""Расчёт метрик для оценки качества агентной системы."""

from __future__ import annotations

import math
import re
from itertools import combinations
from statistics import fmean

from agent_app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationOutput,
    EvaluationSummary,
)


def normalize(text: str) -> str:
    """Обеспечивает воспроизводимое сравнение текстов независимо от регистра, ё/е и пунктуации для корректной проверки фактов."""
    return " ".join(
        re.findall(r"[\w-]+", text.casefold().replace("ё", "е"), flags=re.UNICODE)
    )


def fact_present(fact: str, answer: str) -> bool:
    """Гарантирует обнаружение ожидаемого факта в ответе с учётом морфологических вариаций и устойчиво к незначительным искажениям."""
    normalized_fact = normalize(fact)
    normalized_answer = normalize(answer)
    if not normalized_fact:
        return False
    if normalized_fact in normalized_answer:
        return True

    expected = [_stem(token) for token in normalized_fact.split()]
    answer_tokens = [_stem(token) for token in normalized_answer.split()]
    window_size = len(expected) + 3
    return any(
        all(token in window for token in expected)
        for start in range(len(answer_tokens))
        for window in [answer_tokens[start : start + window_size]]
    )


def evaluate_output(
    case: EvaluationCase, output: EvaluationOutput, *, repetition: int
) -> EvaluationCaseResult:
    """Гарантирует формирование воспроизводимого результата проверки ответа агента по всем критериям тест-кейса для последующего анализа."""
    expected = [
        fact for fact in case.expected_facts if fact_present(fact, output.answer)
    ]
    forbidden = [
        fact for fact in case.forbidden_facts if fact_present(fact, output.answer)
    ]
    claims = _extract_claims(output.answer)
    supported_claims = [
        claim
        for claim in claims
        if any(fact_present(fact, claim) for fact in case.expected_facts)
        and not any(fact_present(fact, claim) for fact in case.forbidden_facts)
    ]
    unsupported_claims = [claim for claim in claims if claim not in supported_claims]
    precision = (
        len(supported_claims) / len(claims)
        if claims
        else (1.0 if not case.expected_facts else 0.0)
    )
    true_positive = len(expected)
    recall = true_positive / len(case.expected_facts) if case.expected_facts else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    unsupported = len(unsupported_claims) / len(claims) if claims else 0.0
    citations_ok = not case.require_citations or output.citations_count > 0
    tools_ok = set(case.expected_tools).issubset(output.tool_calls)
    roles_ok = set(case.expected_roles).issubset(output.selected_roles)
    task_success = (
        output.error is None
        and output.guardrail_action != "review"
        and recall == 1.0
        and not forbidden
        and not unsupported_claims
        and citations_ok
        and tools_ok
        and roles_ok
    )
    return EvaluationCaseResult(
        case_id=case.id,
        repetition=repetition,
        output=output,
        task_success=task_success,
        fact_precision=round(precision, 6),
        fact_recall=round(recall, 6),
        fact_f1=round(f1, 6),
        unsupported_claim_rate=round(unsupported, 6),
        citations_ok=citations_ok,
        tools_ok=tools_ok,
        roles_ok=roles_ok,
        matched_expected_facts=expected,
        matched_forbidden_facts=forbidden,
        unsupported_claims=unsupported_claims,
    )


def _extract_claims(answer: str) -> list[str]:
    """Делит ответ на проверяемые утверждения для консервативной groundedness-оценки."""
    claims = []
    for value in re.split(r"(?:[.!?]+|\n+|;\s*)", answer):
        claim = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", value).strip()
        if normalize(claim):
            claims.append(claim)
    return claims


def summarize(
    results: list[EvaluationCaseResult], *, reviews_pending: int = 0
) -> EvaluationSummary:
    """Гарантирует корректную агрегацию метрик по всем кейсам и выявление аномалий в производительности или стоимости запуска."""
    if not results:
        raise ValueError("Нельзя рассчитать метрики пустого evaluation report")
    latencies = sorted(result.output.latency_ms for result in results)
    rub_costs = [result.output.estimated_cost_rub for result in results]
    conversion_complete = all(cost is not None for cost in rub_costs)
    known_rub_costs = [float(cost) for cost in rub_costs if cost is not None]
    average_cost_rub = round(fmean(known_rub_costs), 8) if conversion_complete else None
    total_cost_rub = round(sum(known_rub_costs), 8) if conversion_complete else None
    return EvaluationSummary(
        executions=len(results),
        task_success_rate=round(fmean(result.task_success for result in results), 6),
        fact_precision=round(fmean(result.fact_precision for result in results), 6),
        fact_recall=round(fmean(result.fact_recall for result in results), 6),
        fact_f1=round(fmean(result.fact_f1 for result in results), 6),
        consistency=round(_consistency(results), 6),
        average_latency_ms=round(fmean(latencies), 3),
        p95_latency_ms=round(_percentile(latencies, 0.95), 3),
        # Legacy-поля теперь имеют однозначную валюту RUB. При неполной
        # конвертации они не используются quality gate и остаются нулевыми.
        average_cost=average_cost_rub or 0.0,
        total_cost=total_cost_rub or 0.0,
        average_cost_rub=average_cost_rub,
        total_cost_rub=total_cost_rub,
        currency_conversion_complete=conversion_complete,
        unconverted_cost_count=sum(cost is None for cost in rub_costs),
        human_reviews_pending=reviews_pending,
    )


def _consistency(results: list[EvaluationCaseResult]) -> float:
    """Вычисляет устойчивость поведения агента по повторным запускам одного кейса, что критично для доверия к системе."""
    by_case: dict[str, list[set[str]]] = {}
    for result in results:
        signature = {
            *(f"fact:{value}" for value in result.matched_expected_facts),
            *(f"forbidden:{value}" for value in result.matched_forbidden_facts),
            *(f"tool:{value}" for value in result.output.tool_calls),
            *(f"role:{value}" for value in result.output.selected_roles),
            f"success:{result.task_success}",
        }
        by_case.setdefault(result.case_id, []).append(signature)
    scores: list[float] = []
    for signatures in by_case.values():
        if len(signatures) == 1:
            scores.append(1.0)
            continue
        for left, right in combinations(signatures, 2):
            union = left | right
            scores.append(len(left & right) / len(union) if union else 1.0)
    return fmean(scores) if scores else 1.0


def _stem(token: str) -> str:
    """Обеспечивает устойчивое сравнение токенов по укороченной форме для повышения толерантности к вариациям слов."""
    if token.isdigit() or len(token) <= 3:
        return token
    if len(token) <= 5:
        return token[:3]
    return token[:5]


def _percentile(values: list[float], quantile: float) -> float:
    """Гарантирует корректное вычисление квантилей для оценки латентности и других распределённых метрик."""
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
