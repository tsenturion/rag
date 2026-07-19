"""Командный интерфейс для оценки качества агентной системы."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    """Создаёт и настраивает parser аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Регрессионная оценка single- или multi-agent режима."
    )
    parser.add_argument("--config", required=True, help="Конфигурация support-агента")
    parser.add_argument("--suite", required=True, help="YAML с evaluation-кейсами")
    return parser


def main() -> int:
    """Запускает командный интерфейс и возвращает код завершения."""
    _configure_stdio()
    args = build_parser().parse_args()

    # Evaluation runtime включает MLflow, LLM и vector store, поэтому он нужен
    # только для реального запуска, но не для построения CLI-справки.
    from agent_app.config import load_agent_config
    from agent_app.evaluation.dataset import load_evaluation_suite
    from agent_app.evaluation.runner import (
        EvaluationRunner,
        RuntimeEvaluationExecutor,
    )
    from agent_app.observability import (
        configure_service_logging,
        configure_telemetry,
    )
    from agent_app.service.runtime import SupportApplicationRuntime

    config = load_agent_config(args.config)
    configure_service_logging(
        config.logging.level, json_format=config.logging.json_format
    )
    configure_telemetry(config.observability)
    suite = load_evaluation_suite(args.suite)
    runtime = SupportApplicationRuntime(config)
    try:
        runner = EvaluationRunner(
            config, RuntimeEvaluationExecutor(runtime, mode=suite.mode)
        )
        report = runner.run(suite)
    finally:
        runtime.close()
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if report.quality_gate.passed else 1


def _configure_stdio() -> None:
    """Сохраняет Unicode из ответов LLM при выводе отчёта в Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
