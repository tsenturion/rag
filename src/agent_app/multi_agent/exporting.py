from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from agent_app.multi_agent.models import (
    MultiAgentComparisonReport,
    MultiAgentRunResult,
)


class MultiAgentExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def export_run(self, result: MultiAgentRunResult) -> Path:
        run_id = result.response.run_id
        return self._transactional_export(
            run_id,
            files={
                "result.json": result.response.model_dump(mode="json"),
                "manifest.json": self._manifest(result),
            },
            jsonl={
                "messages.jsonl": [
                    item.model_dump(mode="json") for item in result.messages
                ],
                "trace.jsonl": [
                    item.model_dump(mode="json") for item in result.response.lifecycle
                ],
                "dead_letters.jsonl": [
                    item.model_dump(mode="json") for item in result.dead_letters
                ],
            },
        )

    def export_comparison(self, report: MultiAgentComparisonReport) -> Path:
        return self._transactional_export(
            f"comparison-{report.run_id}",
            files={"comparison.json": report.model_dump(mode="json")},
            jsonl={},
        )

    def load_result(self, run_id: str) -> dict[str, object] | None:
        path = self.output_dir / run_id / "result.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _transactional_export(
        self,
        name: str,
        *,
        files: dict[str, object],
        jsonl: dict[str, list[dict[str, object]]],
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / name
        if target.exists():
            raise FileExistsError(f"Артефакты запуска уже существуют: {target}")
        staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.output_dir))
        try:
            for filename, payload in files.items():
                (staging / filename).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            for filename, records in jsonl.items():
                content = "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                )
                (staging / filename).write_text(content, encoding="utf-8")
            staging.replace(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return target

    @staticmethod
    def _manifest(result: MultiAgentRunResult) -> dict[str, object]:
        response = result.response
        return {
            "run_id": response.run_id,
            "user_id": response.user_id,
            "session_id": response.session_id,
            "execution_mode": response.execution_mode,
            "selected_agents": response.selected_agents,
            "llm_routes": [
                route.model_dump(mode="json") for route in response.llm_routes
            ],
            "tasks_count": len(response.tasks),
            "messages_count": len(result.messages),
            "dead_letters_count": len(result.dead_letters),
            "history_messages_used": response.history_messages_used,
            "summary_used": response.summary_used,
            "usage": response.usage.model_dump(mode="json"),
            "quality": (
                response.quality.model_dump(mode="json")
                if response.quality is not None
                else None
            ),
        }
