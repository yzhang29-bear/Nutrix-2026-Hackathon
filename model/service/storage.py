"""Photo storage (local directory or S3) and model download from S3.

Photos: the app uploads to S3 with a presigned URL (or to POST /v1/photos) and sends only the key; the service reads the
photo by key. Locally (development, tests) the same keys are files under a directory.
"""
import os
import re
import uuid

KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./ -]{0,511}$")
CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/heic": ".heic", "image/webp": ".webp"}


def check_key(key):
    if not key or not KEY_RE.match(key) or ".." in key.split("/") or key.startswith("/"):
        raise ValueError(f"invalid photo key: {key!r}")
    return key


class PhotoStore:
    def __init__(self, root=None, bucket=None, prefix="photos/", region=None):
        self.root, self.bucket, self.prefix = root, bucket, prefix
        self._s3 = None
        if bucket:
            import boto3
            from botocore.config import Config
            # SigV4 presigned URLs (required in most regions; the boto3 default can produce older signatures)
            self._s3 = boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))

    def describe(self):
        return f"s3://{self.bucket}/{self.prefix}" if self.bucket else f"dir:{self.root}"

    def new_key(self, pid, content_type):
        return f"{self.prefix if self.bucket else ''}{pid}/{uuid.uuid4().hex}{CONTENT_TYPES.get(content_type, '.jpg')}"

    def put(self, key, data, content_type="image/jpeg"):
        check_key(key)
        if self.bucket:
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        else:
            path = os.path.join(self.root, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return key

    def get(self, key):
        check_key(key)
        if self.bucket:
            try:
                return self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            except self._s3.exceptions.NoSuchKey:
                raise FileNotFoundError(key)
        path = os.path.realpath(os.path.join(self.root, key))
        if not path.startswith(os.path.realpath(self.root) + os.sep):
            raise ValueError(f"invalid photo key: {key!r}")
        with open(path, "rb") as f:
            return f.read()

    def upload_url(self, key, content_type, expires=900):
        if not self.bucket:
            raise NotImplementedError("presigned uploads need PMRF_PHOTO_BUCKET (S3)")
        return self._s3.generate_presigned_url("put_object", Params=dict(Bucket=self.bucket, Key=key, ContentType=content_type),
                                               ExpiresIn=expires)


def fetch_model(uri, dest, region=None):
    """s3://bucket/prefix/<version>/ -> local copy (downloads every file listed in the manifest; checksums are verified when
    the store is loaded). Returns the local directory."""
    import json
    import boto3
    assert uri.startswith("s3://")
    bucket, _, prefix = uri[5:].partition("/")
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3", region_name=region)
    os.makedirs(dest, exist_ok=True)
    s3.download_file(bucket, prefix + "manifest.json", os.path.join(dest, "manifest.json"))
    for rel in json.load(open(os.path.join(dest, "manifest.json")))["files"]:
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        s3.download_file(bucket, prefix + rel, out)
    return dest
