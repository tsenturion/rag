from __future__ import annotations

import re

from agent_app.multi_agent.models import (
    ComparisonScenario,
    MultiAgentResponse,
    QualityAssessment,
)


def assess_answer(
    answer: str,
    *,
    citations_count: int,
    expected_terms: list[str] | None = None,
    require_citations: bool = False,
    completed_roles: int = 0,
    expected_roles: int = 0,
) -> QualityAssessment:
    expected_terms = expected_terms or []
    normalized = answer.casefold()
    checks = {
        "answer_present": bool(answer.strip()),
        "no_internal_protocol_markup": not bool(
            re.search(r"(?i)<tool_call>|recipient_name|function_call", answer)
        ),
        "expected_terms": all(term.casefold() in normalized for term in expected_terms),
        "citations": not require_citations or citations_count > 0,
        "roles_completed": expected_roles == 0 or completed_roles == expected_roles,
    }
    notes = [name for name, passed in checks.items() if not passed]
    return QualityAssessment(
        score=round(sum(checks.values()) / len(checks), 4),
        checks=checks,
        notes=notes,
    )


def assess_multi_response(
    response: MultiAgentResponse,
    scenario: ComparisonScenario | None = None,
) -> QualityAssessment:
    completed = sum(result.state == "completed" for result in response.task_results)
    return assess_answer(
        response.answer,
        citations_count=len(response.citations),
        expected_terms=scenario.expected_terms if scenario else [],
        require_citations=scenario.require_citations if scenario else False,
        completed_roles=completed,
        expected_roles=len(response.tasks),
    )
