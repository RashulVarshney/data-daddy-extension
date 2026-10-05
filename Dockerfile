FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
# data/, models/ and eval/listings are mounted at runtime (see docker-compose.yml); none are baked into the image
COPY eval/listings ./eval/listings
EXPOSE 8000
CMD ["uvicorn", "datadaddy_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
