"""Download a model store version from S3 (needed before `docker build`; the model files are not in git).
usage: AWS_PROFILE=... python -I fetch_model_store.py [s3 uri] [local dir]
default: s3://nutrix-models/models/v7-20261010/ -> deploy/model_store/v7-20261010
"""
import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from service.storage import fetch_model
from pmrf.store import load_store

URI = sys.argv[1] if len(sys.argv) > 1 else "s3://nutrix-models/models/v7-20261010/"
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT_DIR, "deploy", "model_store", URI.rstrip("/").split("/")[-1])
fetch_model(URI, DST, os.environ.get("AWS_REGION", "us-west-2"))
print("checksums OK:", load_store(DST, verify=True)["manifest"]["version"], "->", DST)
