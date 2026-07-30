"""Парсинг исходных документов для подготовки документов."""

from __future__ import annotations

import csv
import logging
from typing import Any

from unstructured.partition.auto import partition

from rag_prep.config import ParserConfig
from rag_prep.models import ParseFailure, ParseResult, RawElement, SourceFile
from rag_prep.utils import stable_id

LOGGER = logging.getLogger(__name__)
SECTION_TYPES = {"Title", "Header"}
TITLE_TYPES = {"Title"}


class UnstructuredParsingStage:
    """Парсит файлы в semantic elements через Unstructured и структурированные CSV-строки."""

    def __init__(self, config: ParserConfig, default_section: str):
        """Подготавливает экземпляр с конфигурацией и базовым разделом, обеспечивая корректную настройку парсинга неструктурированных данных."""
        self.config = config
        self.default_section = default_section

    def run(self, sources: list[SourceFile]) -> ParseResult:
        """Обрабатывает список исходных файлов, возвращая распарсенные элементы и аккумулируя ошибки, при этом может прерывать выполнение при критических ошибках."""
        elements: list[RawElement] = []
        failures: list[ParseFailure] = []
        for source in sources:
            try:
                elements.extend(self._parse_source(source))
            except Exception as exc:
                if self.config.fail_on_error:
                    raise
                LOGGER.exception("Не удалось распарсить %s", source.source)
                failures.append(
                    ParseFailure(
                        source=source.source,
                        file_name=source.file_name,
                        file_type=source.file_type,
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                )

        LOGGER.info(
            "Распарсено raw elements: %d из %d файлов; файлов с ошибкой: %d",
            len(elements),
            len(sources),
            len(failures),
        )
        return ParseResult(elements=elements, failures=failures)

    def _parse_source(self, source: SourceFile) -> list[RawElement]:
        """Разбирает содержимое файла в структурированные элементы с учётом типа файла и иерархии разделов для унифицированного представления данных."""
        if source.file_type == "csv":
            return self._parse_csv_source(source)

        parsed = partition(
            filename=str(source.path),
            strategy=self.config.strategy,
            encoding=self.config.encoding,
            languages=self.config.languages,
            pdf_infer_table_structure=self.config.pdf_infer_table_structure,
            skip_infer_table_types=self.config.skip_infer_table_types,
            metadata_filename=str(source.path),
        )

        section_path = [self.default_section]
        raw_elements: list[RawElement] = []
        for index, element in enumerate(parsed):
            text = str(element).strip()
            if not text:
                continue

            element_type = getattr(element, "category", element.__class__.__name__)
            if element_type in SECTION_TYPES:
                section_path = self._next_section_path(section_path, text, element_type)
            section = section_path[-1] if section_path else self.default_section

            raw_elements.append(
                RawElement(
                    source_file=source,
                    element_id=self._element_id(source, index),
                    element_index=index,
                    text=text,
                    element_type=element_type,
                    section=section,
                    section_path=section_path,
                    metadata=self._metadata_to_dict(getattr(element, "metadata", None)),
                )
            )
        return raw_elements

    def _parse_csv_source(self, source: SourceFile) -> list[RawElement]:
        """Обеспечивает корректное чтение CSV-файла и преобразование строк в элементы с метаданными, сохраняя структуру и контекст данных."""
        with source.path.open("r", encoding=self.config.encoding, newline="") as file:
            sample = file.read(65_536)
            file.seek(0)
            try:
                # csv.Sniffer учитывает кавычки и экранирование, поэтому надёжнее
                # ручного split для русских CSV с ';', tab или '|'.
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                LOGGER.warning(
                    "Не удалось определить CSV delimiter для %s; используется запятая",
                    source.source,
                )
                dialect = csv.excel
            reader = csv.DictReader(file, dialect=dialect)
            columns = reader.fieldnames or []
            raw_elements: list[RawElement] = []
            for row_number, row in enumerate(reader, start=1):
                normalized_row = self._normalize_csv_row(row)
                text = self._csv_row_to_text(normalized_row)
                if not text:
                    continue
                raw_elements.append(
                    RawElement(
                        source_file=source,
                        element_id=self._element_id(source, row_number - 1),
                        element_index=row_number - 1,
                        text=text,
                        element_type="CSVRow",
                        section=self._csv_section(normalized_row),
                        section_path=self._csv_section_path(normalized_row),
                        metadata={
                            "csv_row_number": row_number,
                            "csv_columns": columns,
                            "csv": normalized_row,
                            "languages": self.config.languages,
                        },
                    )
                )
        return raw_elements

    def _csv_section(self, row: dict[str, str]) -> str:
        """Определяет раздел документа на основе приоритетных полей CSV-строки, обеспечивая консистентную категоризацию элементов."""
        section = self._csv_value(row, "section", "раздел", "секция", "category")
        title = self._csv_value(row, "title", "заголовок", "тема", "name", "название")
        if section and title:
            return f"{section} / {title}"
        return title or section or self.default_section

    def _csv_section_path(self, row: dict[str, str]) -> list[str]:
        """Формирует иерархический путь раздела из CSV-данных, гарантируя наличие валидного пути для структурирования контента."""
        section = self._csv_value(row, "section", "раздел", "секция", "category")
        title = self._csv_value(row, "title", "заголовок", "тема", "name", "название")
        return [value for value in [section, title] if value] or [self.default_section]

    def _next_section_path(
        self, current_path: list[str], title: str, element_type: str
    ) -> list[str]:
        """Обновляет путь раздела в зависимости от типа элемента, поддерживая корректную иерархию документа."""
        if element_type in TITLE_TYPES:
            return [title]
        if current_path:
            return [current_path[0], title]
        return [title]

    @staticmethod
    def _normalize_csv_row(row: dict[str, Any]) -> dict[str, str]:
        """Обеспечивает стандартизированное представление CSV-строки с очищенными ключами и значениями для надёжного парсинга и обработки."""
        normalized: dict[str, str] = {}
        for key, value in row.items():
            if key is None:
                continue
            normalized[str(key).strip()] = "" if value is None else str(value).strip()
        return normalized

    @staticmethod
    def _csv_row_to_text(row: dict[str, str]) -> str:
        """Гарантирует преобразование строки CSV в человекочитаемый текст для унифицированной подготовки документов."""
        lines = [f"{key}: {value}" for key, value in row.items() if value]
        return "\n".join(lines).strip()

    @staticmethod
    def _csv_value(row: dict[str, str], *keys: str) -> str:
        """Гарантирует извлечение первого непустого значения по списку ключей без учёта регистра, обеспечивая устойчивость к разным вариантам заголовков."""
        by_lower_key = {key.lower(): value for key, value in row.items()}
        for key in keys:
            value = by_lower_key.get(key.lower())
            if value:
                return value
        return ""

    @staticmethod
    def _element_id(source: SourceFile, element_index: int) -> str:
        """Гарантирует уникальность идентификатора элемента в рамках исходного файла и позиции для отслеживания происхождения данных."""
        return stable_id(source.source_hash, element_index)

    @staticmethod
    def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
        """Гарантирует получение словаря метаданных вне зависимости от их исходного формата или отсутствия."""
        if metadata is None:
            return {}
        if hasattr(metadata, "to_dict"):
            return metadata.to_dict()
        if isinstance(metadata, dict):
            return metadata
        return {}
