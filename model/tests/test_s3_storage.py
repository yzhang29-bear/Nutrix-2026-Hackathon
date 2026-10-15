"""S3 code paths (photo store, presigned upload URL, model download) against a mocked S3 (moto); no AWS account needed.
usage: python -I tests/test_s3_storage.py <model store version dir>"""
import os, sys, json, hashlib, tempfile
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.environ.update(AWS_ACCESS_KEY_ID="test", AWS_SECRET_ACCESS_KEY="test", AWS_DEFAULT_REGION="us-east-1")
import boto3
from moto import mock_aws
from service.storage import PhotoStore, fetch_model

STORE = sys.argv[1]
with mock_aws():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="pmrf-test")
    ps = PhotoStore(bucket="pmrf-test", region="us-east-1")
    key = ps.new_key("u1", "image/jpeg")
    assert key.startswith("photos/u1/") and key.endswith(".jpg")
    ps.put(key, b"\xff\xd8jpeg-bytes", "image/jpeg")
    assert ps.get(key) == b"\xff\xd8jpeg-bytes"
    url = ps.upload_url(key, "image/jpeg")
    assert url.startswith("https://") and "X-Amz-Signature" in url
    try:
        ps.get("photos/u1/missing.jpg"); raise AssertionError("missing key must raise")
    except FileNotFoundError:
        pass
    for bad in ["../etc/passwd", "/abs", "a/../../b"]:
        try:
            ps.get(bad); raise AssertionError(f"bad key accepted: {bad}")
        except ValueError:
            pass
    man = json.load(open(os.path.join(STORE, "manifest.json")))
    ver = man["version"]
    for rel in ["manifest.json"] + list(man["files"]):
        s3.upload_file(os.path.join(STORE, rel), "pmrf-test", f"models/{ver}/{rel}")
    with tempfile.TemporaryDirectory() as d:
        fetch_model(f"s3://pmrf-test/models/{ver}/", d, "us-east-1")
        for rel, h in man["files"].items():
            assert hashlib.sha256(open(os.path.join(d, rel), "rb").read()).hexdigest() == h, rel
        from pmrf.store import load_store
        b = load_store(d)
        assert b["manifest"]["version"] == ver
print("S3 storage tests passed: photo put/get, presigned PUT URL, key validation, model download + checksum + load")
