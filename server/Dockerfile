# AWS image: the model team's image (model/deploy/Dockerfile) plus server/, which keeps person data on DynamoDB + S3.
# Build from the repo root:  docker build -f server/Dockerfile --build-arg BASE_IMAGE=pmrf-api:<version> -t pmrf-aws:<tag> .
ARG BASE_IMAGE=pmrf-api:v7-20261010
FROM ${BASE_IMAGE}

COPY server/requirements.txt /tmp/server-requirements.txt
RUN pip install --no-cache-dir -r /tmp/server-requirements.txt
COPY server/ /app/server/
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "75"]
