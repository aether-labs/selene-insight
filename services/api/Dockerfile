FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY services/ services/

RUN pip install --no-cache-dir . && \
    playwright install --with-deps chromium

EXPOSE 8000

CMD ["python", "-m", "services.api.runner"]
