FROM python:3.14.7-slim@sha256:656d12e70054d5fda18a045e2494c96701e9792dd1445f95b3d038df954f57e9 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /home/app app

COPY requirements-service.txt requirements-service-local.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements-service.txt \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0+cpu \
    && python -m pip install --requirement requirements-service-local.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY bpmn ./bpmn
RUN python -m pip install --no-deps . \
    && mkdir -p /app/data/agent /app/data/embeddings_openai \
        /app/data/embeddings_local /app/data/models \
        /app/data/multi_agent /app/data/orchestration \
        /app/data/vector_store_docker_openai \
        /app/data/vector_store_docker_local /app/mlruns \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=5)"

CMD ["rag-support"]
