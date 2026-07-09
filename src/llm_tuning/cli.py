from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_tuning.config import load_fine_tuning_config
from llm_tuning.generation import LocalGenerationStage
from llm_tuning.pipeline import FineTuningPipeline
from rag_prep.utils import setup_logging


class RussianHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, *args, **kwargs) -> str:
        return super()._format_usage(*args, **kwargs).replace("usage:", "использование:", 1)


def _add_russian_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="показать это сообщение и выйти",
    )
    parser._positionals.title = "позиционные аргументы"
    parser._optionals.title = "параметры"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Локальный fine-tuning LLM через LoRA/QLoRA/PEFT.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(parser)
    parser.add_argument(
        "--config",
        default="config/fine_tuning.yaml",
        help="Путь к YAML-конфигу fine-tuning.",
    )
    subparsers = parser.add_subparsers(dest="command", title="команды")

    _add_simple_command(
        subparsers,
        "inspect-env",
        "Проверить torch, XPU/CUDA и выбранное устройство.",
    )
    _add_simple_command(
        subparsers,
        "validate-data",
        "Проверить train/eval датасеты без загрузки модели.",
    )
    _add_simple_command(
        subparsers,
        "baseline",
        "Снять baseline на базовой модели без обучения.",
    )
    _add_simple_command(
        subparsers,
        "train",
        "Запустить fine-tuning и сохранить LoRA adapter.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Оценить базовую модель с уже обученным LoRA adapter.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(evaluate)
    evaluate.add_argument(
        "--adapter-path",
        required=True,
        help="Путь к сохранённому LoRA adapter.",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Сравнить baseline и tuned JSON-отчёты.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(compare)
    compare.add_argument(
        "--baseline-report",
        default=None,
        help="Путь к baseline_report.json.",
    )
    compare.add_argument(
        "--tuned-report",
        default=None,
        help="Путь к tuned_report.json.",
    )

    generate = subparsers.add_parser(
        "generate",
        help="Вызвать локальную модель и получить один ответ без обучения.",
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(generate)
    generate.add_argument(
        "--prompt",
        required=True,
        help="Пользовательский запрос к локальной модели.",
    )
    generate.add_argument(
        "--system",
        default=None,
        help="Необязательная системная инструкция.",
    )
    generate.add_argument(
        "--adapter-path",
        default=None,
        help="Путь к LoRA adapter. Если не задан, используется базовая модель.",
    )
    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Ограничение длины ответа для этого запуска.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "validate-data"
    config = load_fine_tuning_config(Path(args.config))
    setup_logging(config.logging.level)
    pipeline = FineTuningPipeline(config)

    if command == "inspect-env":
        _, device = pipeline.validate()
        payload = device.model_dump(mode="json")
    elif command == "validate-data":
        dataset_validation, device = pipeline.validate()
        payload = {
            "dataset_validation": dataset_validation.model_dump(mode="json"),
            "device": device.model_dump(mode="json"),
        }
    elif command == "baseline":
        payload = pipeline.run_baseline().model_dump(mode="json")
    elif command == "train":
        payload = pipeline.run_training().model_dump(mode="json")
    elif command == "evaluate":
        payload = pipeline.run_evaluation(
            adapter_path=Path(args.adapter_path).expanduser().resolve()
        ).model_dump(mode="json")
    elif command == "compare":
        baseline_report = (
            Path(args.baseline_report).expanduser().resolve()
            if args.baseline_report
            else config.paths.reports_dir / config.paths.baseline_report_filename
        )
        tuned_report = (
            Path(args.tuned_report).expanduser().resolve()
            if args.tuned_report
            else config.paths.reports_dir / config.paths.tuned_report_filename
        )
        payload = pipeline.compare_reports(
            baseline_report_path=baseline_report,
            tuned_report_path=tuned_report,
        ).model_dump(mode="json")
    elif command == "generate":
        adapter_path = (
            Path(args.adapter_path).expanduser().resolve()
            if args.adapter_path
            else None
        )
        payload = LocalGenerationStage(config).run(
            prompt=args.prompt,
            system_prompt=args.system,
            adapter_path=adapter_path,
            max_new_tokens=args.max_new_tokens,
        ).model_dump(mode="json")
    else:
        raise ValueError(f"Неизвестная команда: {command}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_simple_command(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
) -> None:
    command = subparsers.add_parser(
        name,
        help=help_text,
        add_help=False,
        formatter_class=RussianHelpFormatter,
    )
    _add_russian_help(command)


if __name__ == "__main__":
    main()
