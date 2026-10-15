"""Delete everything stored for the given person ids: DynamoDB items, S3 first-week state and S3 photos.
Used by scripts/test_integration.sh to start its test users (flowtest-user, demo-hospital-001) from scratch.
usage: python -I ops/data/reset_person.py <pid> [<pid> ...]
Env: AWS_PROFILE (nutrix), AWS_REGION (us-west-2), PMRF_TABLE (nutrix-people), PMRF_STATE_BUCKET (nutrix-person-state),
     PMRF_PHOTO_BUCKET (nutrix-input-images)
"""
import os
import sys

import boto3
from boto3.dynamodb.conditions import Key

session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "nutrix"), region_name=os.environ.get("AWS_REGION", "us-west-2"))
table = session.resource("dynamodb").Table(os.environ.get("PMRF_TABLE", "nutrix-people"))
s3 = session.client("s3")
buckets = [(os.environ.get("PMRF_STATE_BUCKET", "nutrix-person-state"), "{pid}/"),
           (os.environ.get("PMRF_PHOTO_BUCKET", "nutrix-input-images"), "photos/{pid}/")]

for pid in sys.argv[1:]:
    n = 0
    kw = dict(KeyConditionExpression=Key("pid").eq(pid), ProjectionExpression="pid, sk")
    with table.batch_writer() as bw:
        while True:
            r = table.query(**kw)
            for it in r["Items"]:
                bw.delete_item(Key=dict(pid=it["pid"], sk=it["sk"]))
                n += 1
            if "LastEvaluatedKey" not in r:
                break
            kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    objs = 0
    for bucket, prefix in buckets:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.format(pid=pid)):
            keys = [dict(Key=o["Key"]) for o in page.get("Contents", [])]
            if keys:
                s3.delete_objects(Bucket=bucket, Delete=dict(Objects=keys, Quiet=True))
                objs += len(keys)
    print(f"{pid}: {n} table items, {objs} S3 objects deleted")
