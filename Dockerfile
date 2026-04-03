# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
EXPOSE 7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
