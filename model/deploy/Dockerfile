# PMRF prediction service: CPU only, 2 vCPU / 4 GB per container.
# Build from the project root:  docker build -f deploy/Dockerfile -t pmrf-api:v4-20261010 .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_OFFLINE=1 \
    PMRF_MODEL_DIR=/models/current PMRF_DATA_DIR=/data PMRF_THREADS=2 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu \
 && pip install -r requirements.txt

COPY pmrf/ pmrf/
COPY service/ service/
# The model version is baked into the image (read-only, no download at start-up). To load a version from S3 instead,
# set PMRF_MODEL_URI=s3://<bucket>/models/<version>/ and PMRF_MODEL_DIR=/tmp/model (see DEPLOY.md).
ARG MODEL_VERSION=v7-20261010
COPY deploy/model_store/${MODEL_VERSION}/ /models/current/

RUN useradd --uid 10001 --create-home pmrf && mkdir -p /data && chown pmrf /data
USER pmrf
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "75"]
