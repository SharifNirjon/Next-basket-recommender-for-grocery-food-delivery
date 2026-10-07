# ---- build stage: resolve wheels into an isolated virtualenv ----------------
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-serve.txt /tmp/
# Optional CA bundle for builds behind a TLS-intercepting proxy:
#   docker build --secret id=pip_ca,src=/path/to/ca.crt .   (never stored in the image)
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r /tmp/requirements-serve.txt
RUN find /opt/venv -depth \( -name "__pycache__" -o -name "tests" -o -name "*.pyc" \) -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.11/site-packages/pip* /opt/venv/lib/python3.11/site-packages/setuptools*

# ---- runtime stage: slim image, non-root, artifacts mounted at runtime -------
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
COPY src/recsys /app/src/recsys
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts \
    OMP_NUM_THREADS=1
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "recsys.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
