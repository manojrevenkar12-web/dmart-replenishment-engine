# One image for the nightly batch (`dre run`) and the order API (`dre serve`).
FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
ARG EXTRAS=""
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".${EXTRAS}"
COPY config ./config
USER app
ENV PYTHONUNBUFFERED=1 DRE_REGISTRY=/registry
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=2)" || exit 1
CMD ["dre", "serve", "--registry", "/registry", "--port", "8080"]
