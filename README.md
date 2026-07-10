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

MLflow tracking по умолчанию пишется в `mlruns/` относительно корня проекта, определённого по расположению YAML-конфига. Запуск CLI из другого рабочего каталога не создаёт отдельное хранилище экспериментов.
В `manifest.json` сохраняются параметры запуска, числовые счётчики и диагностический блок `parse_failures` для файлов, которые не удалось разобрать при `parser.fail_on_error: false`.
JSON, JSONL и manifest сначала полностью формируются во временной директории, а затем заменяются как согласованный набор. При ошибке записи или замены предыдущая версия всех артефактов восстанавливается, поэтому новый JSON не смешивается со старым JSONL или manifest.

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
    "source_key": "relative/path/to/file",
    "section": "full_document",
    "file_name": "sample.txt",
    "file_type": "txt",
    "source_hash": "...",
    "text_hash": "...",
    "parent_ids": ["source-id"],
    "origin_element_ids": ["element-id-1", "element-id-2"],
    "lineage": {
      "source_id": "source-id",
      "source_key": "relative/path/to/file",
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

`source` хранит фактический абсолютный путь для диагностики текущего запуска, а `source_key` - путь относительно входной директории. Stable IDs источника, элементов и документа строятся по SHA-256 содержимого и позиции элемента, поэтому перенос проекта, переименование или размещение того же файла в другом каталоге не меняет идентификаторы.

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
4. применяет `chunk_overlap` только внутри текущей секции и не превышает заданный предел;
5. режет LlamaIndex splitter'ом только oversized-блок, который сам больше token budget;
6. считает chunk-level quality signals и сохраняет span metadata.

Так чанки не пересекают границы prepared document/section, а table/list/code-like блоки и повторяющиеся абзацы получают стабильные `semantic_block_ids`. Overlap переносит только целые semantic blocks: если ближайший блок больше лимита, фактическое перекрытие будет меньше заданного значения или нулевым, но никогда не превысит `chunk_overlap`. Для обычных block-aware чанков offsets берутся из заранее посчитанных block spans, без глобального `text.find()` по всему документу. Если fallback splitter не сможет точно найти подстроку для oversized-блока, чанк получит `offset_strategy` с `estimated_*`, warning в логах и отдельный `estimated_offsets_count` в validation.

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
- `chunking.chunk_overlap` - максимальный overlap в токенах, должен быть меньше `chunk_size`; при сохранении целых semantic blocks фактическое перекрытие может быть меньше;
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

Для локального конфига `config/embeddings_local.yaml` результат пишется отдельно в `data/embeddings_local/`, чтобы не смешивать 1536-мерные OpenAI vectors и 384-мерные локальные vectors. GigaChat как chat-модель может работать поверх любого уже построенного vector store; пересчитывать embeddings только из-за смены LLM не нужно.

## Стек embeddings

- OpenAI Python SDK - вызов `client.embeddings.create` для `text-embedding-3-small`;
- `langchain-gigachat` - опциональный provider для GigaChat embeddings;
- `transformers` + `torch` - локальный расчёт embeddings для локального сценария;
- локальная embedding-модель `intfloat/multilingual-e5-small`, скачанная в `data/models/hf/multilingual-e5-small`;
- mean pooling по encoder hidden states для локальной модели;
- L2-нормализация vectors для локального cosine search;
- `tiktoken` - контроль token limits и token budget для OpenAI batch;
- `tenacity` - retry/backoff при временных ошибках OpenAI API;
- `numpy` - расчёт нормы и опциональная L2-нормализация vectors;
- Pydantic - строгие схемы `embedding + metadata`;
- Prefect - оркестрация;
- MLflow - параметры, метрики и артефакты запуска.

## Запуск embeddings

Способ расчёта embeddings выбирается конфигом и должен соответствовать модели, которую вы используете в сценарии:

- `config/embeddings.yaml` - OpenAI-вариант: `text-embedding-3-small`, размерность `1536`, нужен `OPENAI_API_KEY`;
- `config/embeddings_local.yaml` - локальный вариант: `multilingual-e5-small`, размерность `384`, OpenAI API не вызывается;
- `config/embeddings_gigachat.yaml` - опциональный GigaChat-вариант: по умолчанию `Embeddings-2`, размерность `1024`, нужен `GIGACHAT_AUTH_KEY`.

OpenAI-вариант:

```powershell
rag-prep embed --config config/embeddings.yaml --no-prefect
```

Локальный вариант перед первым запуском требует скачать embedding-модель:

```powershell
python scripts/download_hf_model.py --model-id intfloat/multilingual-e5-small --local-dir data/models/hf/multilingual-e5-small
```

Если модель скачана вручную, положите файлы Hugging Face repo в эту же папку:

```text
data/models/hf/multilingual-e5-small
```

Внутри должны быть `config.json`, веса модели (`model.safetensors` или `pytorch_model.bin`) и tokenizer-файлы. После этого повторно скачивать модель не нужно: `config/embeddings_local.yaml` читает её локально через `local_files_only: true`.

Затем локальный расчёт:

```powershell
rag-prep embed --config config/embeddings_local.yaml --no-prefect
```

Перед запуском должен существовать `data/chunks/chunks.jsonl`.

Один и тот же файл чанков можно использовать для всех вариантов embeddings. При расчёте фактическая модель и provider записываются в metadata embedding-записи: OpenAI-вариант сохранит `text-embedding-3-small`, локальный вариант сохранит путь к `multilingual-e5-small`, GigaChat-вариант сохранит имя модели GigaChat.

Через Prefect используются те же конфиги без `--no-prefect`:

```powershell
rag-prep embed --config config/embeddings.yaml
rag-prep embed --config config/embeddings_local.yaml
```

## Конфигурация embeddings

Параметры находятся в `config/embeddings.yaml`, `config/embeddings_local.yaml` или `config/embeddings_gigachat.yaml`:

- `paths.input_jsonl` - входной JSONL с чанками;
- `paths.output_dir` - директория экспорта embeddings;
- `embedding.provider` - `openai`, `local` или `gigachat`;
- `embedding.model` - `text-embedding-3-small` для OpenAI, локальный путь к encoder-модели или модель GigaChat embeddings;
- `embedding.dimensions` - ожидаемая размерность: `1536` для `text-embedding-3-small`, `384` для `multilingual-e5-small`, `1024` / `2560` / `2048` для разных GigaChat embeddings-моделей;
- `embedding.batch_size` - количество чанков в одном batch;
- `embedding.max_batch_tokens` - общий token budget batch;
- `embedding.max_input_tokens` - лимит одного текста;
- `embedding.local_device` - `auto`, `xpu`, `cuda` или `cpu`;
- `embedding.local_dtype` - `auto`, `bf16`, `fp16` или `fp32`;
- `embedding.local_files_only` - не скачивать модель во время расчёта embeddings;
- `embedding.pooling` - стратегия получения одного vector из encoder output;
- `embedding.passage_prefix` и `query_prefix` - префиксы E5-семейства для документов и запросов;
- `embedding.max_retries` и `timeout_seconds` - сетевые guardrails для OpenAI-режима;
- `embedding.normalize` - L2-нормализация vectors;
- `embedding.clear_no_proxy_for_openai` - создаёт OpenAI HTTP-клиент при очищенном `NO_PROXY/no_proxy`, но не меняет эти переменные во время самого API-запроса; это оставляет Prefect доступ к localhost и не ломает маршрут до OpenAI API;
- `embedding.gigachat_scope` - OAuth scope GigaChat, по умолчанию `GIGACHAT_API_PERS`;
- `embedding.gigachat_verify_ssl_certs` - проверка SSL-сертификатов для GigaChat SDK;
- `embedding.gigachat_chars_per_token` - оценка token budget для GigaChat batch без отдельного tokenizer API;
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

Validation проверяет соответствие количества chunks и embeddings, множества chunk ids, текста и identity metadata, размерность vectors, `NaN/Infinity`, дубликаты chunk ids, наличие базовой metadata, соответствие модели и превышение token limit. `manifest.json` сохраняет config, counts и diagnostics, чтобы результат был воспроизводимым и пригодным для следующего этапа загрузки в Chroma, Qdrant, FAISS, pgvector или другую vector DB.

# Пайплайн vector store для RAG

Четвёртый пайплайн создаёт локальное векторное хранилище, загружает уже готовые embeddings и выполняет smoke-проверку similarity search. Embeddings здесь повторно не считаются.

Выбран **Qdrant**, потому что он даёт полноценную коллекцию с vectors + payload metadata, умеет filtering/search и при этом может работать локально без Docker через embedded storage.

Вход:

- `data/embeddings/embeddings.jsonl` - результат embeddings pipeline.

Выход:

- `data/vector_store/manifest.json`
- `data/vector_store/validation.json`
- `data/vector_store/search_results.json`
- `data/qdrant_storage/` - локальное embedded-хранилище Qdrant.

Для локальных embeddings используется отдельный набор путей: `data/vector_store_local/` и `data/qdrant_storage_local/`. Для опциональных GigaChat embeddings используется `data/vector_store_gigachat/` и `data/qdrant_storage_gigachat/`.

Embedded local mode удобен для локальной разработки, но это режим одного процесса. Пайплайн берёт OS-lock на `.rag_prep.lock` внутри `data/qdrant_storage/` и остановит второй локальный запуск с понятной ошибкой. Сам файл lock может остаться после аварийного завершения, но блокировка держится операционной системой и освобождается при завершении процесса. Для конкурентных запусков лучше поднять Qdrant server и переключить `vector_store.mode: http`.

## Стек vector store

- `qdrant-client` - создание коллекции, upsert vectors, similarity search;
- Qdrant local mode - локальное хранилище без отдельного сервера;
- Pydantic - схемы результатов indexing/validation/search;
- Prefect - оркестрация;
- MLflow - параметры, метрики и артефакты проверки.

## Запуск vector store

Перед запуском должен существовать `data/embeddings/embeddings.jsonl`.

Для OpenAI embeddings используется основной конфиг:

```powershell
rag-prep vector-store --config config/vector_store.yaml --no-prefect
```

Для локальных embeddings используется отдельный конфиг:

```powershell
rag-prep vector-store --config config/vector_store_local.yaml --no-prefect
```

Для опциональных GigaChat embeddings используется отдельный конфиг:

```powershell
rag-prep vector-store --config config/vector_store_gigachat.yaml --no-prefect
```

Через Prefect используются те же конфиги без `--no-prefect`:

```powershell
rag-prep vector-store --config config/vector_store.yaml
rag-prep vector-store --config config/vector_store_local.yaml
rag-prep vector-store --config config/vector_store_gigachat.yaml
```

## Конфигурация vector store

Параметры находятся в `config/vector_store.yaml`:

- `paths.input_jsonl` - готовые embeddings;
- `paths.output_dir` - директория отчётов;
- `vector_store.provider` - сейчас `qdrant`;
- `vector_store.mode` - `local` для embedded Qdrant или `http` для внешнего Qdrant server;
- `vector_store.collection_name` - имя коллекции;
- `vector_store.vector_size` - размерность vectors: `1536` для `text-embedding-3-small`, `384` для `multilingual-e5-small`, `1024` / `2560` / `2048` для разных GigaChat embeddings-моделей;
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

# Интеграция GigaChat

Проект поддерживает GigaChat как отдельный provider для chat-модели агента. Это не меняет уже построенный RAG-индекс: LLM, которая отвечает пользователю, и embedding-модель, которая строит vector store, являются разными компонентами.

Практический сценарий:

1. посчитать embeddings через `config/embeddings.yaml` или `config/embeddings_local.yaml`;
2. загрузить их в Qdrant через `config/vector_store.yaml` или `config/vector_store_local.yaml`;
3. использовать `config/agent_gigachat.yaml` как генеративную модель агента с tools и памятью.

Для работы нужен Authorization key в `.env`:

```powershell
GIGACHAT_AUTH_KEY=...
```

Ключ передаётся в GigaChat SDK как `credentials`, а SDK сам получает и обновляет access token через OAuth. В коде не нужно вручную вызывать `/api/v2/oauth`.

## GigaChat для агента

Конфиг:

```text
config/agent_gigachat.yaml
```

Запуск одноразового запроса:

```powershell
rag-agent --config config/agent_gigachat.yaml --message "Сколько будет 128 * 47? Используй калькулятор."
```

Запуск сценариев MVP-агента:

```powershell
rag-agent --config config/agent_gigachat.yaml --run-scenarios --scenario-report data/agent/scenario_report_gigachat.json
```

В `config/agent_gigachat.yaml` основные параметры:

- `agent.provider: gigachat`;
- `agent.model` - по умолчанию `GigaChat-2`; для более сложных задач можно указать `GigaChat-2-Pro` или `GigaChat-2-Max`;
- `agent.gigachat_auth_key_env` - имя переменной с Authorization key;
- `agent.gigachat_scope` - scope для OAuth, по умолчанию `GIGACHAT_API_PERS`;
- `agent.gigachat_verify_ssl_certs` - проверка SSL-сертификатов SDK;
- `agent.max_new_tokens` - передаётся в GigaChat как `max_tokens`.

LangGraph остаётся тем же: `agent -> tools -> agent`. `langchain-gigachat` поддерживает `bind_tools`, поэтому backend выполняет tools так же, как в OpenAI-варианте: модель выбирает tool call, `ToolNode` исполняет Python-функцию, затем результат возвращается модели.

## GigaChat и готовый vector store

GigaChat chat-модель не требует, чтобы vectors были посчитаны GigaChat embeddings-моделью. Для неё подходят уже подготовленные индексы:

- OpenAI embeddings: `1536` координат, `config/vector_store.yaml`;
- локальные embeddings: `384` координаты, `config/vector_store_local.yaml`.

Размерность vector store важна только для этапа индексации и поиска. Chat-модель получает уже найденный текстовый контекст, поэтому она не зависит от того, были vectors размерности `1536`, `384` или другой.

## Опциональные GigaChat embeddings

Конфиг:

```text
config/embeddings_gigachat.yaml
```

Запуск:

```powershell
rag-prep embed --config config/embeddings_gigachat.yaml --no-prefect
```

Результат пишется отдельно:

```text
data/embeddings_gigachat/
```

Поддерживаемые размерности GigaChat embeddings нужно согласовывать с `embedding.dimensions` и `vector_store.vector_size`:

- `Embeddings` - `1024`;
- `Embeddings-2` - `1024`;
- `EmbeddingsGigaR` - `2560`;
- `Embeddings-3B-2025-09` - `2048`.

Если в `config/embeddings_gigachat.yaml` указана известная модель и неверная размерность, конфиг не загрузится. Это защищает от ситуации, когда embeddings уже посчитаны в одной размерности, а Qdrant-коллекция создана под другую.

После расчёта GigaChat embeddings используется отдельный Qdrant-конфиг:

```powershell
rag-prep vector-store --config config/vector_store_gigachat.yaml --no-prefect
```

Для модели `Embeddings-2` в `config/vector_store_gigachat.yaml` стоит `vector_store.vector_size: 1024`. При смене embedding-модели GigaChat нужно поменять оба значения: `embedding.dimensions` и `vector_store.vector_size`.

## Файлы GigaChat-интеграции

- `config/agent_gigachat.yaml` - агент на GigaChat;
- `config/embeddings_gigachat.yaml` - расчёт GigaChat embeddings;
- `config/vector_store_gigachat.yaml` - Qdrant-индекс под GigaChat vectors;
- `src/agent_app/llm.py` - сборка `langchain_gigachat.chat_models.GigaChat`;
- `src/rag_prep/embedding_stages/embedding.py` - `GigaChatEmbeddingStage`;
- `.env.example` - пример переменной `GIGACHAT_AUTH_KEY`.

# Модуль агента с tools и памятью

Это модуль LangGraph-агента с tools и полноценной памятью.

## Что реализовано

- LangGraph workflow `agent -> tools -> agent` для OpenAI, GigaChat и локального Qwen backend;
- OpenAI chat-модель, по умолчанию `gpt-4.1-nano`;
- GigaChat chat-модель через `config/agent_gigachat.yaml`;
- локальный backend `transformers` через `config/agent_local.yaml` для Qwen без OpenAI API;
- Qwen-style tool calling через chat template, XML-теги `<tool_call>` и parser в `AIMessage.tool_calls`;
- tools:
  - `calculator` для точных вычислений;
  - `current_datetime` для текущей даты и времени;
  - `get_weather` для OpenWeatherMap через `httpx2`;
  - `calculate_travel_budget` для расчёта бюджета поездки;
  - `advise_packing` для подбора вещей в поездку;
  - `create_project`, `create_task`, `update_task_status`, `list_project_tasks`, `summarize_project_state` для проектных сценариев;
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
- сценарии MVP-агента в `config/agent_scenarios.yaml`;
- JSON-отчёт прохождения сценариев;
- CLI-команда `rag-agent`.

## Конфигурация агента

Параметры находятся в `config/agent.yaml`:

- `agent.provider` - `openai`, `gigachat` или `local`;
- `agent.model` - модель агента или путь к локальной модели;
- `agent.adapter_path` - необязательный путь к LoRA adapter для локального backend;
- `agent.temperature` - температура генерации;
- `agent.max_new_tokens` - лимит ответа локальной модели;
- `agent.gigachat_auth_key_env`, `gigachat_scope`, `gigachat_verify_ssl_certs` - параметры GigaChat provider;
- `agent.max_history_messages` - максимальный размер short-term buffer; при переполнении история делится только на границе полного хода `user -> assistant`;
- `agent.max_summary_chars` - ограничение summary memory;
- `agent.tool_error_retries` - количество повторов identical tool call после ошибочного результата, по умолчанию `1`;
- `memory.sqlite_path` - SQLite-файл долговременной памяти;
- `memory.default_user_id` и `default_session_id`; global-записи с `session_id=null` и записи конкретных сессий хранятся в независимых scope;
- `weather.api_key_env` - имя переменной окружения с OpenWeatherMap API-ключом;
- `weather.default_city`;
- `weather.default_units`.

В `.env` должны быть:

```powershell
OPENAI_API_KEY=...
OPENWEATHER_API_KEY=...
GIGACHAT_AUTH_KEY=...
HF_TOKEN=hf_...
```

Пример без секретов лежит в `.env.example`.

## Запуск агента

Одноразовый запрос:

```powershell
rag-agent --config config/agent.yaml --message "Сколько будет 128 * 47?"
```

Локальный вызов Qwen без OpenAI API:

```powershell
rag-agent --config config/agent_local.yaml --message "Кратко объясни, зачем нужен LoRA adapter."
```

Локальный вызов Qwen с tool:

```powershell
rag-agent --config config/agent_local.yaml --message "Сколько будет 128 * 47? Используй калькулятор."
```

Вызов GigaChat с tool:

```powershell
rag-agent --config config/agent_gigachat.yaml --message "Сколько будет 128 * 47? Используй калькулятор."
```

В `config/agent_local.yaml` можно указать уже обученный adapter:

```yaml
agent:
  adapter_path: data/models/lora/<run_id>
```

Локальный backend использует `transformers.apply_chat_template(..., tools=...)`, парсит `<tool_call>{...}</tool_call>`, передаёт вызов в LangGraph `ToolNode`, а затем возвращает результат модели через `<tool_response>`. Для Qwen2.5-1.5B-Instruct это рабочий учебный режим, но он менее надёжен, чем OpenAI function calling или более крупные Qwen-модели: маленькая модель может ошибаться в выборе tool, аргументах или финальной формулировке.

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

## MVP-агент: сценарии и проверка поведения

MVP-слой показывает не просто одиночный ответ LLM, а воспроизводимое поведение агента:

1. сценарий задаёт цель, пользовательский запрос и ожидаемый результат;
2. `test_case_id` связывает сценарий и каждый шаг с конкретным тест-кейсом;
3. шаги сценария фиксируют цепочку действий;
4. поля `llm_role`, `tools_role` и `memory_role` объясняют роли компонентов;
5. `action_chain`, `decision_points` и `transition_rules` описывают сценарную логику;
6. agent trace показывает стартовое состояние, tool calls, tool results, изменения памяти и финальное состояние;
7. evaluator проверяет критерии прохождения;
8. результат сохраняется в JSON-отчёт.

Основные файлы:

- `config/agent_scenarios.yaml` - сценарии, шаги и критерии прохождения;
- `src/agent_app/scenarios/models.py` - Pydantic-схемы сценариев и отчётов;
- `src/agent_app/scenarios/loading.py` - загрузка YAML;
- `src/agent_app/scenarios/runner.py` - запуск сценариев через реального `AgentRunner`;
- `src/agent_app/scenarios/evaluator.py` - проверка ответов, tools и памяти;
- `src/agent_app/graph.py` - LangGraph-агент и trace состояний;
- `src/agent_app/tools/` - tools агента;
- `src/agent_app/memory/` - short-term, summary и long-term memory.

Типы сценариев:

- `main` - основной рабочий сценарий;
- `alternative` - сценарий с неполными данными и допущениями;
- `error` - ошибочный сценарий, например попытка сохранить секрет;
- `recovery` - восстановление контекста из долговременной памяти;
- `tool_failure` - корректная обработка ошибки внешнего tool;
- `loop_guard` - проверка защиты от повторяющихся tool-вызовов.

Запустить все сценарии:

```powershell
rag-agent --config config/agent.yaml --run-scenarios
```

Запустить один сценарий:

```powershell
rag-agent --config config/agent.yaml --run-scenario project_manager
```

Сохранить отчёт в явный путь:

```powershell
rag-agent --config config/agent.yaml --run-scenarios --scenario-report data/agent/scenario_report.json
```

Получить полный JSON в консоли:

```powershell
rag-agent --config config/agent.yaml --run-scenario travel_planning --json
```

Проверить сценарий защиты от циклов:

```powershell
rag-agent --config config/agent.yaml --run-scenario tool_loop_guard --scenario-report data/agent/scenario_report_loop_guard.json
```

В отчёте для каждого шага сохраняются:

- `test_case_id`;
- финальный ответ агента;
- вызванные tools;
- результаты tools;
- стартовое, промежуточные и финальное состояния;
- флаг `loop_guard_triggered`, если агент остановил повторяющийся tool loop;
- правила переходов и точки принятия решений;
- созданные, обновлённые и удалённые записи памяти;
- список checks с `passed=true/false`.

Сценарий считается пройденным, если прошли все step-level checks и scenario-level checks: ожидаемые tools были вызваны, запрещённые tools не использовались, нужные записи памяти появились, секреты не сохранились, а агент завершил ответ без зацикливания.

Команда возвращает exit code `0`, только если весь выбранный набор сценариев прошёл. При `failed` возвращается `1`, поэтому результат можно напрямую использовать в CI.

Связь сценариев с тест-кейсами:

- `TC-MVP-001` - основной сценарий планирования поездки;
- `TC-MVP-002` - альтернативный сценарий с неполными данными;
- `TC-MVP-003` - ошибочный сценарий с попыткой сохранить секрет;
- `TC-MVP-004` - основной сценарий project manager;
- `TC-MVP-005` - восстановление контекста из long-term memory;
- `TC-MVP-006` - fallback при ошибке внешнего weather tool;
- `TC-MVP-007` - защита от повторяющегося tool loop.

Отладка сценариев:

- если сценарий `failed`, сначала открыть `data/agent/scenario_report.json`;
- проверить failed checks внутри `step_results[].checks`;
- сверить `response.tool_calls` с `expected_tools` и `forbidden_tools`;
- посмотреть `response.trace.tool_results`, чтобы увидеть ошибку tool;
- посмотреть `response.trace.memory_created_ids`, `memory_updated_ids` и `memory_deleted_ids`;
- скорректировать либо критерии в `config/agent_scenarios.yaml`, либо поведение tools/graph в `src/agent_app/`.

Типовые сбои, которые уже покрыты кодом:

- попытка сохранить API-ключ останавливается secret guardrail до LLM/tools;
- ошибка weather API попадает в tool result и допускается только в сценарии `tool_failure`;
- `agent.recursion_limit` ограничивает переходы `agent -> tools -> agent`;
- после ошибочного tool result допускается не более `agent.tool_error_retries` повторов с теми же аргументами;
- после успешного результата или исчерпания retry loop guard останавливает identical tool call и закрывает отменённый вызов результатом `status=cancelled`;
- ошибка LLM-вызова summary memory не отменяет уже готовый ответ; short-term history обрезается до последних полных ходов;
- один LLM backend переиспользуется во всём наборе сценариев, поэтому локальная модель не загружается в XPU повторно для каждого кейса;
- `ScenarioRunner` ловит исключение шага и записывает failed check вместо аварийного обрыва всего запуска.

Deterministic проверка loop guard без реальных API-вызовов:

```powershell
python -m unittest discover -s tests -v
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

# LoRA/QLoRA/PEFT fine-tuning локальной LLM

Этот модуль готовит воспроизводимый fine-tuning локальной LLM через PEFT.

Основная модель для основного запуска:

- Hugging Face ID: `Qwen/Qwen2.5-1.5B-Instruct`;
- локальный путь после скачивания: `data/models/hf/Qwen2.5-1.5B-Instruct`;
- fallback для быстрых проверок: `Qwen/Qwen2.5-0.5B-Instruct`.

Для Intel(R) Arc(TM) 140T GPU 16GB базовый режим - `LoRA` через `torch.xpu`. QLoRA оставлен как опциональный режим, но не включён по умолчанию: 4-bit режимы через `bitsandbytes` исторически сильнее завязаны на CUDA, поэтому для Intel Arc надёжнее начинать с обычного LoRA, маленького batch size и gradient accumulation.

## Выбор метода

Критерии выбора:

- размер модели: чем больше base model, тем сильнее давление на память GPU/CPU;
- доступная память: на Intel Arc 140T 16GB безопаснее начинать с 0.5B-1.5B модели;
- требования к качеству: большая модель обычно даёт лучший baseline, но медленнее обучается;
- скорость итераций: для учебных запусков важнее быстрый цикл `данные -> обучение -> отчёт`, чем максимальный размер модели.

LoRA:

- базовая модель загружается в обычной точности `bf16`/`fp16`/`fp32`;
- обучаются только adapter-веса;
- режим проще и надёжнее для Intel Arc/XPU;
- выбран по умолчанию.

QLoRA:

- базовая модель загружается в 4-bit quantized режиме;
- экономит память, но добавляет зависимость от 4-bit backend;
- полезен для более крупных моделей при ограниченной памяти;
- в этом проекте включается только явно через `peft.method: qlora` и заранее проверяет поддержку.

## Что реализовано

- отдельный пакет `src/llm_tuning/`;
- конфиг `config/fine_tuning.yaml`;
- русскоязычные train/eval JSONL-датасеты в `data/fine_tuning/`;
- проверка датасета и пересечения train/eval id;
- auto-device выбор `xpu` / `cuda` / `cpu`;
- загрузка tokenizer и causal LM через `transformers`;
- PEFT LoRA adapter через `peft`;
- QLoRA-ветка с явной проверкой доступности;
- supervised chat tokenization с masking prompt-токенов через `-100`;
- запуск fine-tuning через `Trainer`;
- baseline evaluation до обучения;
- evaluation после обучения;
- сравнение baseline/tuned отчётов;
- сохранение adapter-а, manifest и JSON-отчётов;
- MLflow-логирование параметров, метрик и артефактов;
- CLI-команда `llm-tune`.

## Основные файлы

- `config/fine_tuning.yaml` - локальная Qwen-модель, LoRA/QLoRA, параметры обучения, пути и MLflow;
- `config/fine_tuning_smoke.yaml` - быстрый smoke-конфиг для проверки кода на маленькой модели;
- `data/fine_tuning/train.jsonl` - обучающие примеры;
- `data/fine_tuning/eval.jsonl` - проверочные примеры;
- `src/llm_tuning/config.py` - Pydantic-конфиг;
- `src/llm_tuning/dataset.py` - загрузка и валидация JSONL;
- `src/llm_tuning/device.py` - выбор `torch.xpu`, CUDA или CPU;
- `src/llm_tuning/modeling.py` - загрузка модели и подключение PEFT;
- `src/llm_tuning/tokenization.py` - токенизация chat-примеров;
- `src/llm_tuning/training.py` - fine-tuning через `Trainer`;
- `src/llm_tuning/evaluation.py` - генерация ответов и eval loss;
- `src/llm_tuning/comparison.py` - сравнение поведения до/после;
- `src/llm_tuning/pipeline.py` - ООП-фасад над этапами;
- `src/llm_tuning/cli.py` - CLI.

## Установка зависимостей

Проект рассчитан на глобальное окружение Python без отдельного venv:

```powershell
python -m pip install --upgrade -r requirements.txt
python -m pip install -e . --no-deps
```

Проверить окружение:

```powershell
python -m pip check
```

Если `torch.xpu.is_available()` возвращает `False`, нужно проверить установленную сборку PyTorch для Intel GPU и драйвер Intel Arc. На CPU модуль тоже запускается, но обучение будет заметно медленнее.

Для Intel Arc нужна XPU-сборка PyTorch. По официальной документации PyTorch для Intel GPU, после установки драйвера Intel GPU stable wheel ставится так:

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Проверка XPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

Если вывод `False`, обучение пойдёт на CPU. Это не ошибка кода fine-tuning, а признак того, что в текущем глобальном Python установлена CPU-сборка PyTorch или не готов драйвер Intel GPU.

Если Hugging Face зависает на Xet/CAS-загрузке больших файлов, в конфиге можно оставить:

```yaml
model:
  hub_disable_xet: true
  hub_disable_symlink_warning: true
```

Это переводит загрузку в более предсказуемый режим для локального учебного запуска.

## Команды без запуска обучения

Проверить датасет и выбранное устройство:

```powershell
llm-tune --config config/fine_tuning.yaml validate-data
```

Проверить только окружение:

```powershell
llm-tune --config config/fine_tuning.yaml inspect-env
```

Эти команды не загружают модель для обучения и не запускают fine-tuning.

Для быстрой проверки всего кода без скачивания большой instruct-модели есть smoke-конфиг:

```powershell
llm-tune --config config/fine_tuning_smoke.yaml train
```

Он использует `sshleifer/tiny-gpt2`, делает 2 training steps, сохраняет LoRA adapter, baseline/tuned reports, comparison и MLflow run. Этот режим проверяет работоспособность pipeline, но не предназначен для оценки качества русскоязычной модели.

## Скачивание локальной модели

Перед обучением основной модели можно заранее скачать `Qwen/Qwen2.5-1.5B-Instruct`. Команды выполняются из корня проекта:

```powershell
cd C:\Users\user\Desktop\rag
```

Проверить доступ к Hugging Face без скачивания весов:

```powershell
python scripts/download_hf_model.py --dry-run
```

Скачать модель:

```powershell
python scripts/download_hf_model.py
```

Скрипт читает `HF_TOKEN` из `.env`, отключает Xet-загрузчик Hugging Face и сохраняет модель в:

```text
data/models/hf/Qwen2.5-1.5B-Instruct
```

Если `Qwen/Qwen2.5-1.5B-Instruct` уже скачана вручную, структура такая же: файлы repo должны лежать в `data/models/hf/Qwen2.5-1.5B-Instruct`. Скрипт скачивания в этом случае запускать не нужно.

Основной `config/fine_tuning.yaml` уже настроен на локальную папку:

```yaml
model:
  model_id: data/models/hf/Qwen2.5-1.5B-Instruct
```

Локальные `model_id`, `tokenizer_id` и `fallback_model_id` разрешаются относительно корня проекта, определённого по расположению YAML-конфига. Поэтому `llm-tune` можно запускать из другого рабочего каталога, передав абсолютный путь к `config/fine_tuning.yaml`.

Если нужно брать модель напрямую с Hugging Face Hub, можно временно заменить это значение на:

```yaml
model:
  model_id: Qwen/Qwen2.5-1.5B-Instruct
```

## Локальный вызов модели

Проверить базовую локальную модель без OpenAI API:

```powershell
llm-tune --config config/fine_tuning.yaml generate --prompt "Кратко объясни, зачем нужен LoRA adapter." --max-new-tokens 80
```

Проверить ту же модель с обученным LoRA adapter:

```powershell
llm-tune --config config/fine_tuning.yaml generate --prompt "Кратко объясни, зачем нужен LoRA adapter." --adapter-path data/models/lora/<run_id> --max-new-tokens 80
```

Эта команда не запускает обучение. Она загружает локальную модель из `data/models/hf/Qwen2.5-1.5B-Instruct`, при необходимости подключает adapter и возвращает один JSON-ответ.

## Baseline, обучение и оценка

Снять baseline на базовой модели без обучения:

```powershell
llm-tune --config config/fine_tuning.yaml baseline
```

Запустить LoRA fine-tuning:

```powershell
llm-tune --config config/fine_tuning.yaml train
```

Команда `train` при текущем конфиге:

1. валидирует train/eval датасеты;
2. снимает baseline до обучения;
3. запускает LoRA fine-tuning;
4. сохраняет adapter в `data/models/lora/<run_id>/`;
5. оценивает adapter на том же eval-наборе;
6. сравнивает baseline и tuned-ответы;
7. пишет отчёты в `data/fine_tuning/reports/<run_id>/`, не перезаписывая результаты других запусков;
8. логирует параметры и метрики в MLflow.

Оценить уже обученный adapter отдельно:

```powershell
llm-tune --config config/fine_tuning.yaml evaluate --adapter-path data/models/lora/<run_id>
```

Сравнить два готовых отчёта:

```powershell
llm-tune --config config/fine_tuning.yaml compare --baseline-report data/fine_tuning/reports/<run_id>/baseline_report.json --tuned-report data/fine_tuning/reports/<run_id>/tuned_report.json
```

Оба пути для `compare` задаются явно. Метрики сравнения вычисляются только по общим `example_id`; отчёты без общих примеров отклоняются как несопоставимые.

## Ключевые параметры

- `model.model_id` - базовая локальная модель;
- `model.fallback_model_id` - резервная локальная или Hub-модель, которая используется, если основную модель невозможно загрузить; в отчётах сохраняется фактически загруженный `model_id`;
- `model.device` - `auto`, `xpu`, `cuda` или `cpu`;
- `model.dtype` - `auto`, `bf16`, `fp16` или `fp32`;
- `model.max_seq_length` - максимальная длина обучающего примера;
- `model.hub_disable_xet` - отключение Xet-загрузчика Hugging Face для более стабильного локального скачивания;
- `peft.method` - `lora` или `qlora`;
- `peft.r`, `lora_alpha`, `lora_dropout` - параметры LoRA;
- `peft.target_modules` - слои, куда вставляется adapter;
- `training.learning_rate` - learning rate;
- `training.per_device_train_batch_size` - batch size на устройство;
- `training.gradient_accumulation_steps` - накопление градиента;
- `training.num_train_epochs` - число эпох;
- `training.eval_strategy`, `save_strategy` - частота оценки и чекпоинтов.

Практический подход к настройке:

- если модель переобучается, уменьшить `learning_rate`, число эпох или увеличить dropout;
- если модель недообучается, увеличить число эпох, `r` или аккуратно поднять `learning_rate`;
- если не хватает памяти, уменьшить `per_device_train_batch_size`, `max_seq_length` или перейти на fallback-модель;
- если обучение нестабильно, снизить `learning_rate` и оставить `gradient_accumulation_steps` больше 1;
- каждую итерацию фиксировать в YAML-конфиге и MLflow, чтобы можно было сравнить runs.

## Метрики и выводы

В отчётах фиксируются:

- `train_loss`;
- `eval_loss`;
- `perplexity`;
- `pass_rate` по проверочным критериям;
- `log_history` с динамикой train/eval метрик;
- ответы baseline и tuned-модели;
- примеры, где качество улучшилось;
- примеры с регрессией;
- итоговый текстовый вывод.

Это позволяет сравнивать поведение модели до/после на одинаковых запросах, а не оценивать fine-tuning только по субъективному впечатлению.
