"""Upload a model store version to S3 (s3://<bucket>/models/<version>/). Files first, manifest.json last, so a partial
upload is never loadable; refuses to overwrite a version that already has a manifest.
usage: AWS_PROFILE=... python -I upload_model_store.py <store version dir> <bucket> [prefix=models]
"""
import sys, os, json, time
import boto3
from boto3.s3.transfer import TransferConfig

SRC, BUCKET = sys.argv[1].rstrip("/"), sys.argv[2]
PREFIX = (sys.argv[3] if len(sys.argv) > 3 else "models").strip("/")
man = json.load(open(os.path.join(SRC, "manifest.json")))
ver = man["version"]
s3 = boto3.client("s3")
base = f"{PREFIX}/{ver}/"
existing = s3.list_objects_v2(Bucket=BUCKET, Prefix=base + "manifest.json").get("KeyCount", 0)
if existing:
    sys.exit(f"s3://{BUCKET}/{base} already has a manifest; upload a new version instead of overwriting")
cfg = TransferConfig(multipart_threshold=64 * 1024 ** 2, multipart_chunksize=64 * 1024 ** 2, max_concurrency=8)
t, total = time.time(), 0
for rel in man["files"]:
    p = os.path.join(SRC, rel)
    s3.upload_file(p, BUCKET, base + rel, Config=cfg, ExtraArgs=dict(Metadata={"sha256": man["files"][rel]}))
    total += os.path.getsize(p)
s3.upload_file(os.path.join(SRC, "manifest.json"), BUCKET, base + "manifest.json", ExtraArgs=dict(ContentType="application/json"))
print(f"uploaded {len(man['files']) + 1} files, {total / 1e6:.1f} MB to s3://{BUCKET}/{base} in {time.time() - t:.0f}s")
