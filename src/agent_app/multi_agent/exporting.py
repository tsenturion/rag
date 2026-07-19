"""Экспорт воспроизводимых артефактов для мультиагентной системы."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from uuid import UUID

from agent_app.multi_agent.models import (
    MultiAgentComparisonReport,
    MultiAgentRunResult,
)


class MultiAgentExporter:
    """Обеспечивает атомарную и воспроизводимую выгрузку результатов мультиагентных запусков для последующего анализа."""

    def __init__(self, output_dir: Path):
        """Гарантирует готовность экземпляра к сохранению и загрузке результатов в заданном каталоге."""
        self.output_dir = output_dir

    def export_run(self, result: MultiAgentRunResult) -> Path:
        """Гарантирует целостную сериализацию и сохранение всех артефактов запуска для последующего воспроизведения."""
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
        """Гарантирует сохранность и доступность результатов сравнения запусков для анализа и аудита."""
        return self._transactional_export(
            f"comparison-{report.run_id}",
            files={"comparison.json": report.model_dump(mode="json")},
            jsonl={},
        )

    def load_result(self, run_id: str) -> dict[str, object] | None:
        """Загружает результат только по каноническому UUID внутри output_dir."""
        try:
            canonical_run_id = str(UUID(run_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("run_id должен быть каноническим UUID") from exc
        if canonical_run_id != run_id.casefold():
            raise ValueError("run_id должен быть каноническим UUID")

        root = self.output_dir.resolve()
        path = (root / canonical_run_id / "result.json").resolve()
        if root not in path.parents:
            raise ValueError("Путь результата выходит за пределы output_dir")
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
        """Гарантирует атомарное сохранение артефактов запуска мультиагентной системы, предотвращая частичные записи и обеспечивая целостность данных при ошибках."""
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
        """Формирует стандартизированное описание результата запуска мультиагентной системы, обеспечивая единый интерфейс для анализа и отчётности."""
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
