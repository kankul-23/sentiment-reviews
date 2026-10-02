# Сервис оценки отзывов: rubert-tiny2 (seed 42) + температурная калибровка, CPU.
# Сборка (из корня репозитория; веса models/rubert-tiny2/seed_42 должны лежать локально):
#   docker build -t sentiment-reviews .
#   docker run --rm -p 8000:8000 sentiment-reviews
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-сборка torch заметно меньше стандартной (без CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

RUN useradd --create-home --uid 1000 app

COPY src/__init__.py src/__init__.py
COPY src/serving/ src/serving/
COPY configs/transformer.yaml configs/transformer.yaml
COPY reports/calibration.json reports/calibration.json
COPY models/rubert-tiny2/seed_42/ models/rubert-tiny2/seed_42/

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
