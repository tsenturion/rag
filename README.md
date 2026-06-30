# Пайплайн подготовки данных для RAG

Подготовка данных для RAG без чанкинга, embeddings и vector DB.

Пайплайн делает только подготовительные этапы:

1. загрузка источников через LlamaIndex `SimpleDirectoryReader`;
2. парсинг `pdf`, `txt`, `html` через Unstructured и строковый парсинг `csv` через структурированный CSV parser;
3. очистка текста regex-правилами;
4. нормализация Unicode/регистра и sentence statistics через spaCy;
5. дедупликация exact hash + near-duplicate MinHash LSH через datasketch;
6. структурирование в `text + metadata` и LlamaIndex `Document`;
7. экспорт в JSON и JSONL;
8. оркестрация через Prefect и логирование артефактов/метрик в MLflow.

Выходной формат подготовлен для следующих этапов RAG: чанкинга, embeddings и vector DB. Эти этапы здесь намеренно не реализованы, но metadata уже содержит lineage, hierarchy и quality signals, чтобы следующие процессы могли сохранять связь чанков с исходниками и элементами.

## Быстрый запуск

Зависимости и локальный пакет уже установлены в глобальный Python. Запуск из корня проекта:

```powershell
rag-prep --config config/default.yaml
```

Прямой запуск без Prefect, но с теми же классами этапов:

```powershell
rag-prep --config config/default.yaml --no-prefect
```

Результаты пишутся в `data/prepared/`:

- `documents.json`
- `documents.jsonl`
- `manifest.json`

MLflow tracking по умолчанию пишется в `mlruns/`.
В `manifest.json` сохраняются параметры запуска, числовые счётчики и диагностический блок `parse_failures` для файлов, которые не удалось разобрать при `parser.fail_on_error: false`.
Файлы результата пишутся через временный файл и атомарную замену, поэтому частично записанный JSON/JSONL не остаётся на месте целевого файла.

## Установка

Проект рассчитан на установку в текущий глобальный Python, без создания отдельного окружения. Команды нужно выполнять из корня проекта.

Обновить зависимости:

```powershell
python -m pip install --upgrade -r requirements.txt
```

Установить локальный пакет в editable-режиме, чтобы команда `rag-prep` была доступна после изменения исходников:

```powershell
python -m pip install -e . --no-deps
```

Флаг `--no-deps` уместен, если зависимости уже установлены через `requirements.txt`. Если проект переносится на новую машину, сначала ставится `requirements.txt`, затем локальный пакет.

Проверить целостность глобального окружения:

```powershell
python -m pip check
```

На Windows для Unstructured дополнительно установлен `python-magic-bin`, чтобы `unstructured.partition.auto` корректно определял типы файлов. Для OCR-режимов PDF могут понадобиться системные Tesseract и Poppler, но для текстовых PDF достаточно `parser.strategy: fast`.
Команда `rag-prep` перед запуском Prefect добавляет `localhost`, `127.0.0.1` и `::1` в `NO_PROXY`, чтобы локальный временный сервер Prefect не ломался из-за системных proxy-настроек. Для локальных CLI-запусков также отключается Prefect EventsWorker: это не мешает оркестрации, но не даёт процессу зависать на websocket-событиях временного сервера. Внешние API, включая OpenAI, при этом не отключаются от системного proxy. Импорт `rag_prep.flow` сам по себе переменные окружения не меняет.

## Примеры входных данных

В `data/raw/` лежат русскоязычные примеры, похожие на реальные документы организации:

- `sample.txt` - регламент обработки заявок в ИТ-службе;
- `sample.html` - инструкция по передаче документов в электронный архив;
- `sample.csv` - табличный реестр этапов service desk, где каждая строка становится отдельным элементом, а в конце есть exact и near-дубли для проверки дедупликации;
- `sample.pdf` - памятка по подготовке комплекта документов к архивированию.

Эти файлы нужны для smoke-проверки пайплайна. Для своих данных можно заменить содержимое `data/raw/` или указать другой `paths.input_dir` в `config/default.yaml`.

## Формат результата

Каждая запись:

```json
{
  "text": "...",
  "metadata": {
    "id": "stable-document-id",
    "source": "absolute/path/to/file",
    "section": "full_document",
    "file_name": "sample.txt",
    "file_type": "txt",
    "source_hash": "...",
    "text_hash": "...",
    "parent_ids": ["source-id"],
    "origin_element_ids": ["element-id-1", "element-id-2"],
    "lineage": {
      "source_id": "source-id",
      "source_hash": "...",
      "origin_element_ids": ["element-id-1", "element-id-2"],
      "element_range": [0, 3],
      "pipeline_stage": "prepared_document"
    },
    "hierarchy": {
      "section_path": ["Раздел", "Подраздел"],
      "section_depth": 2,
      "document_order": 0
    },
    "element_start": 0,
    "element_end": 3,
    "element_types": ["NarrativeText", "Title"],
    "page_number": null,
    "char_count": 100,
    "word_count": 16,
    "sentence_count": 2,
    "pipeline_run_id": "...",
    "parsed_at": "...",
    "extra": {}
  }
}
```

## Готовность к следующим этапам

Поля, которые нужны следующим этапам без реализации самих этапов:

- `metadata.id` - стабильный идентификатор подготовленного документа;
- `metadata.parent_ids` - родительские сущности, сейчас это идентификатор source;
- `metadata.origin_element_ids` - элементы парсинга, из которых собран документ;
- `metadata.lineage` - цепочка `source -> parsed elements -> prepared document`;
- `metadata.hierarchy.section_path` - путь секции для будущего document tree;
- `metadata.hierarchy.document_order` - порядок документа внутри исходника;
- `metadata.extra.quality` - диагностические scores для boilerplate/OCR garbage/menu leftovers.

Quality scoring не удаляет документы сам по себе. Он даёт сигнал будущим этапам чанкинга/retrieval, чтобы можно было фильтровать мусор, расследовать retrieval misses и объяснять происхождение ответа.

## Конфигурация

Основные параметры находятся в `config/default.yaml`:

- `paths.input_dir` и `paths.output_dir`;
- `loader.allowed_extensions`;
- `parser.strategy`: `fast`, `auto`, `hi_res`, `ocr_only`;
- `parser.fail_on_error`: `false` пропускает проблемный файл и пишет failure в manifest, `true` останавливает пайплайн;
- `cleaning.drop_patterns` для удаления boilerplate;
- `normalization.spacy_language`;
- `deduplication.threshold`, `num_perm`, `shingle_size`;
- `structuring.group_by_section`;
- `logging.mlflow_enabled`.

Относительные пути в `paths.*` считаются относительно корня проекта, если конфиг лежит в папке `config/`. Если конфиг расположен в другой папке, относительные пути считаются относительно папки этого YAML-файла.

Для OCR PDF на Windows могут дополнительно понадобиться системные Tesseract/Poppler. Для текстовых PDF используется `strategy: fast`.

# Пайплайн чанкинга для RAG

Второй пайплайн делает только чанкинг подготовленных документов. Он не считает embeddings, не пишет vector DB и не запускает retrieval.

Вход берётся из результата первого пайплайна:

- `data/prepared/documents.jsonl` - структурированные документы `text + metadata`;
- lineage, hierarchy, source metadata и quality signals переносятся в каждый чанк.

Выход пишется в `data/chunks/`:

- `chunks.json`
- `chunks.jsonl`
- `manifest.json`

## Стек чанкинга

- LlamaIndex `SentenceSplitter` - основной семантически ориентированный splitter, который старается не резать текст внутри предложений;
- LlamaIndex `TokenTextSplitter` - альтернативная стратегия для строгого token-based режима;
- `tiktoken` - подсчёт токенов под `text-embedding-3-small`;
- Pydantic - схема чанка и metadata;
- Prefect - воспроизводимая оркестрация;
- MLflow - параметры запуска, метрики размера чанков и артефакты экспорта.

## Логика чанкинга

Пайплайн не режет весь документ одной плоской строкой. Он работает в таком порядке:

1. читает `PreparedDocument` из первого пайплайна;
2. восстанавливает semantic blocks внутри секции по границам абзацев/элементов, которые были сохранены как `\n\n`;
3. упаковывает блоки в chunk до `chunk_size`;
4. применяет `chunk_overlap` только внутри текущей секции;
5. режет LlamaIndex splitter'ом только oversized-блок, который сам больше token budget;
6. считает chunk-level quality signals и сохраняет span metadata.

Так чанки не пересекают границы prepared document/section, а table/list/code-like блоки и повторяющиеся абзацы получают стабильные `semantic_block_ids`. Для обычных block-aware чанков offsets берутся из заранее посчитанных block spans, без глобального `text.find()` по всему документу. Если fallback splitter не сможет точно найти подстроку для oversized-блока, чанк получит `offset_strategy` с `estimated_*`, warning в логах и отдельный `estimated_offsets_count` в validation.

## Запуск чанкинга

Перед запуском чанкинга должен существовать файл `data/prepared/documents.jsonl`.

Через Prefect:

```powershell
rag-prep chunk --config config/chunking.yaml
```

Прямой запуск без Prefect:

```powershell
rag-prep chunk --config config/chunking.yaml --no-prefect
```

Старые команды подготовки данных остаются рабочими:

```powershell
rag-prep prepare --config config/default.yaml
rag-prep prepare --config config/default.yaml --no-prefect
```

Legacy-форма тоже сохранена:

```powershell
rag-prep --config config/default.yaml
```

## Конфигурация чанкинга

Параметры находятся в `config/chunking.yaml`:

- `paths.input_jsonl` - JSONL из первого пайплайна;
- `paths.output_dir` - директория экспорта чанков;
- `chunking.strategy` - `sentence` или `token`;
- `chunking.chunk_size` - целевой размер чанка в токенах;
- `chunking.chunk_overlap` - overlap в токенах, должен быть меньше `chunk_size`;
- `chunking.tokenizer_model` - модель токенизатора, по умолчанию `text-embedding-3-small`;
- `chunking.embedding_model` - модель будущих embeddings, записывается в metadata, сами embeddings не считаются;
- `chunking.preserve_section_boundaries` - не смешивать разные prepared documents/sections;
- `chunking.preserve_block_boundaries` - сначала упаковывать semantic blocks и резать только oversized-блоки;
- `chunking.max_chunk_tokens` - validation guardrail перед embeddings;
- `chunking.min_quality_score` - порог для диагностического счётчика low-quality chunks;
- `chunking.fail_on_validation_error` - останавливать пайплайн при пустых, слишком маленьких, слишком больших, low-quality чанках, estimated offsets или проблемах lineage.

## Формат Чанка

Каждая запись в `chunks.jsonl` готова к следующему embeddings-пайплайну:

```json
{
  "text": "...",
  "metadata": {
    "id": "stable-chunk-id",
    "document_id": "prepared-document-id",
    "source": "absolute/path/to/file",
    "section": "Раздел",
    "position": 0,
    "chunk_start_char": 0,
    "chunk_end_char": 430,
    "chunk_token_count": 128,
    "chunk_size": 220,
    "chunk_overlap": 40,
    "chunking_strategy": "sentence",
    "tokenizer_model": "text-embedding-3-small",
    "embedding_model": "text-embedding-3-small",
    "semantic_block_ids": ["semantic-block-id-1"],
    "semantic_block_start": 0,
    "semantic_block_end": 0,
    "offset_strategy": "semantic_block_span",
    "parent_ids": ["prepared-document-id"],
    "origin_element_ids": ["element-id-1"],
    "lineage": {
      "document_id": "prepared-document-id",
      "chunk_id": "stable-chunk-id",
      "chunk_position": 0,
      "semantic_block_ids": ["semantic-block-id-1"],
      "semantic_block_range": [0, 0],
      "pipeline_stage": "chunk"
    },
    "hierarchy": {
      "section_path": ["Раздел"],
      "document_order": 0,
      "chunk_position": 0,
      "semantic_block_count": 1,
      "semantic_block_range": [0, 0]
    },
    "source_hash": "...",
    "document_text_hash": "...",
    "text_hash": "...",
    "file_name": "sample.txt",
    "file_type": "txt",
    "quality": {
      "token_density": 0.37,
      "language_confidence": 0.99,
      "ocr_noise_score": 0.0,
      "structure_score": 0.82,
      "unique_token_ratio": 0.74,
      "semantic_block_count": 1,
      "is_low_quality_chunk": false
    },
    "chunked_at": "..."
  }
}
```

`position`, `chunk_start_char`, `chunk_end_char`, `semantic_block_ids`, `parent_ids`, `origin_element_ids` и `lineage` нужны для отладки retrieval misses, hallucinations и обратной трассировки ответа к исходному документу. `embedding_model` фиксируется заранее, чтобы следующий этап мог проверить совместимость чанков с выбранной моделью embeddings.

# Пайплайн embeddings для RAG

Третий пайплайн считает embeddings для готовых чанков. Он не создаёт vector DB, не индексирует данные и не запускает retrieval.

Вход:

- `data/chunks/chunks.jsonl` - чанки из пайплайна чанкинга.

Выход:

- `data/embeddings/embeddings.json`
- `data/embeddings/embeddings.jsonl`
- `data/embeddings/manifest.json`

## Стек embeddings

- OpenAI Python SDK - вызов `client.embeddings.create`;
- модель по умолчанию `text-embedding-3-small`;
- `tiktoken` - контроль token limits и token budget для batch;
- `tenacity` - retry/backoff при временных ошибках OpenAI API;
- `numpy` - расчёт нормы и опциональная L2-нормализация vectors;
- Pydantic - строгие схемы `embedding + metadata`;
- Prefect - оркестрация;
- MLflow - параметры, метрики и артефакты запуска.

## Запуск embeddings

В `.env` должен быть OpenAI API-ключ:

```powershell
OPENAI_API_KEY=sk-...
```

Через Prefect:

```powershell
rag-prep embed --config config/embeddings.yaml
```

Прямой запуск без Prefect:

```powershell
rag-prep embed --config config/embeddings.yaml --no-prefect
```

Перед запуском должен существовать `data/chunks/chunks.jsonl`.

## Конфигурация embeddings

Параметры находятся в `config/embeddings.yaml`:

- `paths.input_jsonl` - входной JSONL с чанками;
- `paths.output_dir` - директория экспорта embeddings;
- `embedding.provider` - сейчас поддерживается `openai`;
- `embedding.model` - модель embeddings, по умолчанию `text-embedding-3-small`;
- `embedding.dimensions` - ожидаемая размерность, для `text-embedding-3-small` используется `1536`;
- `embedding.batch_size` - количество чанков в одном API batch;
- `embedding.max_batch_tokens` - общий token budget batch;
- `embedding.max_input_tokens` - лимит одного текста;
- `embedding.max_retries` и `timeout_seconds` - сетевые guardrails;
- `embedding.normalize` - опциональная L2-нормализация vectors;
- `embedding.clear_no_proxy_for_openai` - создаёт OpenAI HTTP-клиент при очищенном `NO_PROXY/no_proxy`, но не меняет эти переменные во время самого API-запроса; это оставляет Prefect доступ к localhost и не ломает маршрут до OpenAI API;
- `embedding.fail_on_validation_error` - останавливать пайплайн при ошибках validation.

## Формат записи embedding

Каждая строка в `embeddings.jsonl` готова к загрузке в vector store:

```json
{
  "text": "...",
  "embedding": [0.0123, -0.0456],
  "metadata": {
    "id": "chunk-id",
    "document_id": "prepared-document-id",
    "source": "absolute/path/to/file",
    "section": "Раздел",
    "position": 0,
    "chunk_token_count": 128,
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536,
    "embedding_vector_hash": "...",
    "embedding_norm": 1.02,
    "embedding_run_id": "...",
    "lineage": {},
    "hierarchy": {}
  }
}
```

Validation проверяет соответствие количества chunks и embeddings, размерность vectors, `NaN/Infinity`, дубликаты chunk ids, наличие базовой metadata, соответствие модели и превышение token limit. `manifest.json` сохраняет config, counts и diagnostics, чтобы результат был воспроизводимым и пригодным для следующего этапа загрузки в Chroma, Qdrant, FAISS, pgvector или другую vector DB.

# Пайплайн vector store для RAG

Четвёртый пайплайн создаёт локальное векторное хранилище, загружает уже готовые embeddings и выполняет smoke-проверку similarity search. Embeddings здесь повторно не считаются, OpenAI API не вызывается.

Выбран **Qdrant**, потому что он даёт полноценную коллекцию с vectors + payload metadata, умеет filtering/search и при этом может работать локально без Docker через embedded storage.

Вход:

- `data/embeddings/embeddings.jsonl` - результат embeddings pipeline.

Выход:

- `data/vector_store/manifest.json`
- `data/vector_store/validation.json`
- `data/vector_store/search_results.json`
- `data/qdrant_storage/` - локальное embedded-хранилище Qdrant.

Embedded local mode удобен для локальной разработки, но это режим одного процесса. Пайплайн берёт OS-lock на `.rag_prep.lock` внутри `data/qdrant_storage/` и остановит второй локальный запуск с понятной ошибкой. Сам файл lock может остаться после аварийного завершения, но блокировка держится операционной системой и освобождается при завершении процесса. Для конкурентных запусков лучше поднять Qdrant server и переключить `vector_store.mode: http`.

## Стек vector store

- `qdrant-client` - создание коллекции, upsert vectors, similarity search;
- Qdrant local mode - локальное хранилище без отдельного сервера;
- Pydantic - схемы результатов indexing/validation/search;
- Prefect - оркестрация;
- MLflow - параметры, метрики и артефакты проверки.

## Запуск vector store

Перед запуском должен существовать `data/embeddings/embeddings.jsonl`.

Через Prefect:

```powershell
rag-prep vector-store --config config/vector_store.yaml
```

Прямой запуск без Prefect:

```powershell
rag-prep vector-store --config config/vector_store.yaml --no-prefect
```

## Конфигурация vector store

Параметры находятся в `config/vector_store.yaml`:

- `paths.input_jsonl` - готовые embeddings;
- `paths.output_dir` - директория отчётов;
- `vector_store.provider` - сейчас `qdrant`;
- `vector_store.mode` - `local` для embedded Qdrant или `http` для внешнего Qdrant server;
- `vector_store.collection_name` - имя коллекции;
- `vector_store.vector_size` - размерность vectors, для `text-embedding-3-small` это `1536`;
- `vector_store.distance` - метрика similarity, по умолчанию `Cosine`;
- `vector_store.recreate_collection` - пересоздавать коллекцию для воспроизводимого запуска; в кодовом дефолте это `false`, а в `config/vector_store.yaml` явно стоит `true`;
- `vector_store.batch_size` - размер upsert batch;
- `vector_store.local_storage_path` - путь к embedded-хранилищу;
- `vector_store.search_limit` - количество результатов на тестовый запрос;
- `vector_store.test_queries_count` - сколько готовых embeddings использовать как тестовые запросы;
- `vector_store.validation_sample_size` - сколько точек проверить через scroll;
- `vector_store.fail_on_validation_error` - останавливать пайплайн при некорректном индексе.

## Что загружается в Qdrant

Vector сохраняется как Qdrant vector, а payload содержит:

```json
{
  "text": "...",
  "chunk_id": "stable-chunk-id",
  "document_id": "prepared-document-id",
  "source": "absolute/path/to/file",
  "section": "Раздел",
  "position": 0,
  "file_name": "sample.txt",
  "file_type": "txt",
  "embedding_model": "text-embedding-3-small",
  "embedding_provider": "openai",
  "embedding_dimensions": 1536,
  "metadata": {}
}
```

Qdrant point id строится детерминированно как UUID5 от `collection_name + chunk_id`, потому что исходный `metadata.id` не обязан быть UUID. Оригинальный chunk id сохраняется в `payload.chunk_id` и `payload.metadata.id`. Перед upsert pipeline проверяет дубликаты chunk ids и сгенерированных point ids, чтобы случайный повтор не перезаписал точку молча.

## Проверки

Пайплайн проверяет:

- количество points в коллекции совпадает с количеством embeddings;
- направление расхождения counts сохраняется в `count_delta`, `extra_points_count` и `missing_points_count`;
- размерность коллекции и vectors соответствует `vector_size`;
- точки без возвращённого vector считаются отдельно в `missing_vector_count`, а не смешиваются с неправильной размерностью;
- несовпадение размерности коллекции и отдельных point vectors логируется раздельно через `collection_vector_size_mismatch_count` и `point_vector_size_mismatch_count`;
- distance metric соответствует конфигу;
- payload содержит `text`;
- payload содержит полную `metadata`;
- обязательные поля metadata присутствуют;
- similarity search возвращает результаты;
- для smoke-запросов ближайший результат обычно совпадает с исходным chunk.

`search_results.json` сохраняет тестовые запросы и найденные hits. Это не production retrieval, а базовая проверка корректности записи/чтения индекса перед следующим этапом RAG.

# Модуль агента с tools и памятью

Это модуль LangGraph-агента с tools и полноценной памятью.

## Что реализовано

- LangGraph workflow `agent -> tools -> agent`;
- OpenAI chat-модель, по умолчанию `gpt-4.1-nano`;
- tools:
  - `calculator` для точных вычислений;
  - `current_datetime` для текущей даты и времени;
  - `get_weather` для OpenWeatherMap через `httpx2`;
  - `save_memory`;
  - `search_memory`;
  - `get_memory`;
  - `update_memory`;
  - `delete_memory`;
  - `list_memories`;
  - `clear_session_memory`;
- short-term buffer memory для текущей сессии;
- summary memory для сжатия длинного диалога;
- long-term memory в SQLite;
- CLI-команда `rag-agent`.

## Конфигурация агента

Параметры находятся в `config/agent.yaml`:

- `agent.model` - модель агента;
- `agent.temperature` - температура генерации;
- `agent.max_history_messages` - размер short-term buffer;
- `agent.max_summary_chars` - ограничение summary memory;
- `memory.sqlite_path` - SQLite-файл долговременной памяти;
- `memory.default_user_id` и `default_session_id`;
- `weather.api_key_env` - имя переменной окружения с OpenWeatherMap API-ключом;
- `weather.default_city`;
- `weather.default_units`.

В `.env` должны быть:

```powershell
OPENAI_API_KEY=...
OPENWEATHER_API_KEY=...
```

## Запуск агента

Одноразовый запрос:

```powershell
rag-agent --config config/agent.yaml --message "Сколько будет 128 * 47?"
```

Интерактивный режим:

```powershell
rag-agent --config config/agent.yaml
```

Выбрать область пользователя и сессии:

```powershell
rag-agent --user-id default --session-id lesson-1 --message "Запомни, что мой проект называется RAG Engineer Assistant"
```

Посмотреть память:

```powershell
rag-agent --list-memory
```

Очистить память текущей сессии:

```powershell
rag-agent --clear-session-memory
```

## Проверочные запросы

```text
Сколько будет 128 * 47?
```

```text
Какая сегодня дата?
```

```text
Какая погода в Екатеринбурге?
```

```text
Запомни, что мой проект называется RAG Engineer Assistant.
```

```text
Как называется мой проект?
```

```text
Что ты обо мне помнишь?
```

```text
Обнови название моего проекта на Engineer Support Agent.
```

```text
Забудь название моего проекта.
```
