"""Реализация компонентов для вспомогательных сценариев проекта."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def main() -> None:
    """Запускает командный интерфейс и возвращает код завершения."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    args = build_parser().parse_args()

    if args.disable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if args.disable_symlink_warning:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(
            f"Не найден {args.token_env}. Добавьте токен в .env, например: "
            f"{args.token_env}=hf_..."
        )

    local_dir = args.local_dir.expanduser()
    if not local_dir.is_absolute():
        local_dir = project_root / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Модель: {args.model_id}")
    print(f"Папка: {local_dir}")
    print(f"Токен: {args.token_env}=<скрыто>")
    print(f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '')}")

    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=local_dir,
        token=token,
        max_workers=args.max_workers,
        force_download=args.force_download,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("Dry-run завершён. Файлы не скачивались.")
        for item in path:
            print(f"- {item.filename} ({item.file_size} байт)")
    else:
        print(f"Скачивание завершено: {path}")


def build_parser() -> argparse.ArgumentParser:
    """Создаёт и настраивает parser аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description=(
            "Скачивание произвольной Hugging Face модели в локальную папку проекта."
        ),
    )
    parser.add_argument(
        "--model-id",
        required=True,
        type=_non_empty,
        help="ID репозитория модели на Hugging Face, например Qwen/Qwen2.5-1.5B-Instruct.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        type=Path,
        help="Папка назначения относительно корня проекта или абсолютный путь.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Имя переменной окружения с Hugging Face токеном.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Количество потоков скачивания. 1 обычно стабильнее на Windows.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Перекачать файлы заново вместо продолжения/использования кеша.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверить список файлов без скачивания.",
    )
    parser.add_argument(
        "--disable-xet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Отключить Xet-загрузчик Hugging Face.",
    )
    parser.add_argument(
        "--disable-symlink-warning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Отключить предупреждение Hugging Face о symlink-кеше на Windows.",
    )
    return parser


def _non_empty(value: str) -> str:
    """Гарантирует, что аргумент командной строки не пустой и пригоден для дальнейшей обработки."""
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("значение не может быть пустым")
    return normalized


if __name__ == "__main__":
    main()
