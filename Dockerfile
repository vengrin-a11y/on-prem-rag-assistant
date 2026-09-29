FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
COPY sample_data/ ./sample_data/
CMD ["uvicorn", "on_prem_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
