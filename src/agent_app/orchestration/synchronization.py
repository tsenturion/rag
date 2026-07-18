from __future__ import annotations

import re
from collections import Counter

from agent_app.orchestration.models import StepResult, StepStatus, SynchronizationResult

VOTE_RE = re.compile(r"\[(approve|reject|abstain)\]", re.IGNORECASE)


class QuorumCoordinator:
    """Сводит результаты barrier и проверяет достижение кворума."""

    def evaluate(
        self,
        results: list[StepResult],
        *,
        required: int,
        cancelled_steps: list[str] | None = None,
    ) -> SynchronizationResult:
        agent_results = [
            result
            for result in results
            if result.assigned_role is not None and result.status != StepStatus.SKIPPED
        ]
        successful = [result for result in agent_results if result.successful]
        votes = [self._vote(result) for result in successful]
        vote_counts = Counter(vote for vote in votes if vote != "abstain")
        consensus = "undetermined"
        if vote_counts:
            vote, count = vote_counts.most_common(1)[0]
            if count >= required:
                consensus = vote
        return SynchronizationResult(
            required=required,
            received=len(agent_results),
            successful=len(successful),
            quorum_reached=len(successful) >= required,
            consensus=consensus,
            cancelled_steps=cancelled_steps or [],
        )

    @staticmethod
    def _vote(result: StepResult) -> str:
        if result.vote is not None:
            return result.vote
        match = VOTE_RE.search(result.output)
        return match.group(1).lower() if match else "abstain"
