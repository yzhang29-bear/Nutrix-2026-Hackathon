"""Person store on DynamoDB + S3, a drop-in replacement for pmrf.state.PersonStore (files on EFS).

The model team's code stays untouched: server/app.py swaps this class in at start-up. Same methods, same return types.

  DynamoDB table (pid, sk):
    sk = "meta"                         meta document (JSON) + cgm_updated (epoch s, last CGM write)
    sk = "sig#<kind>#<YYYY-MM-DD>"      one day of one signal kind as parquet bytes + row count n
  S3 <state bucket>/<pid>/:
    hist.parquet, arrays.npz, profile.json   first-week state (1-3 MB; above DynamoDB's 400 KB item limit).
                                             profile.json is written last and marks the state as complete.

Concurrency: writes for one person are serialised by a lock inside one container (as in PersonStore). Two containers
writing the same person's signals at the same moment can still race; fine for one app client per person.
"""
import io
import json
import os
import time

import boto3
import numpy as np
import pandas as pd
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from pmrf.state import (ARRAY_COLS, SIGNAL_COLUMNS, SIGNAL_KEYS, STATE_FORMAT, PersonStore, _naive, _now,
                        check_pid)

STATE_FILES = ("hist.parquet", "arrays.npz", "profile.json")


def _parquet(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class DynamoPersonStore(PersonStore):
    def __init__(self, table, state_bucket, region=None, stamp_root="/tmp/pmrf-stamps"):
        super().__init__(stamp_root)          # root only holds local cache-stamp files (see _dir)
        self.table = boto3.resource("dynamodb", region_name=region).Table(table)
        self.s3 = boto3.client("s3", region_name=region)
        self.bucket = state_bucket

    def describe(self):
        return f"dynamodb:{self.table.name} + s3://{self.bucket}"

    # ---------------------------------------------------------------- meta
    def _meta_item(self, pid):
        return self.table.get_item(Key=dict(pid=check_pid(pid), sk="meta")).get("Item")

    def exists(self, pid):
        return self._meta_item(pid) is not None

    def meta(self, pid):
        item = self._meta_item(pid)
        if item is None:
            raise FileNotFoundError(f"no meta for {pid}")
        return json.loads(item["doc"])

    def write_meta(self, pid, **kw):
        item = self._meta_item(pid)
        m = json.loads(item["doc"]) if item else dict(pid=pid, created=_now())
        m.update(kw, updated=_now())
        self.table.update_item(Key=dict(pid=check_pid(pid), sk="meta"), UpdateExpression="SET doc = :d",
                               ExpressionAttributeValues={":d": json.dumps(m, default=str)})
        return m

    # ---------------------------------------------------------------- signals
    def _chunks(self, pid, kind, attrs=None):
        kw = dict(KeyConditionExpression=Key("pid").eq(check_pid(pid)) & Key("sk").begins_with(f"sig#{kind}#"))
        if attrs:
            kw.update(ProjectionExpression=", ".join(f"#{a}" for a in attrs), ExpressionAttributeNames={f"#{a}": a for a in attrs})
        while True:
            r = self.table.query(**kw)
            yield from r["Items"]
            if "LastEvaluatedKey" not in r:
                return
            kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]

    @staticmethod
    def _frame(item):
        return pd.read_parquet(io.BytesIO(item["data"].value))

    def read_signal(self, pid, kind):
        parts = [self._frame(it) for it in self._chunks(pid, kind)]
        parts = [p for p in parts if len(p)]
        if not parts:
            return pd.DataFrame(columns=SIGNAL_COLUMNS[kind])
        return pd.concat(parts, ignore_index=True).sort_values(SIGNAL_KEYS[kind]).reset_index(drop=True)

    def signal_day(self, pid, kind, date):
        """one calendar day (YYYY-MM-DD, the person's wall-clock time) of one signal kind: a single DynamoDB item"""
        item = self.table.get_item(Key=dict(pid=check_pid(pid), sk=f"sig#{kind}#{date}")).get("Item")
        df = self._frame(item) if item else pd.DataFrame(columns=SIGNAL_COLUMNS[kind])
        return df.reindex(columns=SIGNAL_COLUMNS[kind])

    def signal_dates(self, pid, kind):
        """{date: row count} of the stored day chunks of one signal kind (keys and counts only, no data read)"""
        return {it["sk"].rsplit("#", 1)[1]: int(it["n"]) for it in self._chunks(pid, kind, attrs=["sk", "n"])}

    def append_signal(self, pid, kind, df, replace=False):
        """append rows (deduplicated on the timestamp; a new row replaces an old one at the same time)"""
        df = df.reindex(columns=SIGNAL_COLUMNS[kind])
        key = SIGNAL_KEYS[kind]
        df[key] = _naive(df[key])
        with self._lock:
            if replace:
                with self.table.batch_writer() as bw:
                    for it in self._chunks(pid, kind, attrs=["pid", "sk"]):
                        bw.delete_item(Key=dict(pid=it["pid"], sk=it["sk"]))
            day = df[key].dt.strftime("%Y-%m-%d").fillna("none") if len(df) else pd.Series(dtype=str)
            for d, rows in df.groupby(day, sort=True):
                sk = f"sig#{kind}#{d}"
                old = None if replace else self.table.get_item(Key=dict(pid=pid, sk=sk)).get("Item")
                parts = [x for x in ((self._frame(old) if old else None), rows) if x is not None and len(x)]
                new = pd.concat(parts, ignore_index=True)
                if kind != "meds":
                    new = new.drop_duplicates(key, keep="last")
                new = new.sort_values(key).reset_index(drop=True)
                self.table.put_item(Item=dict(pid=pid, sk=sk, data=_parquet(new), n=len(new)))
            if kind == "cgm":
                self.table.update_item(Key=dict(pid=pid, sk="meta"), UpdateExpression="SET cgm_updated = :t",
                                       ExpressionAttributeValues={":t": int(time.time() * 1000)})
        return int(sum(int(it["n"]) for it in self._chunks(pid, kind, attrs=["n"])))

    def _dir(self, pid, *parts):
        """The service's /profile cache keys on the mtime of signals/cgm.parquet. Keep a local marker file whose mtime is
        the last CGM write recorded in DynamoDB, so the cache stays right across containers."""
        p = super()._dir(pid, *parts)
        if parts == ("signals", "cgm.parquet"):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "a").close()
            item = self._meta_item(pid) or {}
            t = int(item.get("cgm_updated", 0)) / 1000
            os.utime(p, (t, t))
        return p

    # ---------------------------------------------------------------- state
    def _state_key(self, pid, name):
        return f"{check_pid(pid)}/{name}"

    def has_state(self, pid):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._state_key(pid, "profile.json"))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def save_state(self, pid, state):
        H = state["hist"].copy()
        arrays = {c: np.stack(H[c].to_numpy()) if len(H) else np.zeros((0, 0)) for c in ARRAY_COLS}
        E, Y = state["own_lib"]
        arrays.update(own_emb=np.asarray(E, np.float32), own_y=np.asarray(Y, float))
        H = H.drop(columns=ARRAY_COLS)
        for c in H.columns:
            if H[c].dtype == object and c not in ("pid", "meal_type", "photo", "photo_key", "group", "source"):
                H[c] = pd.to_numeric(H[c], errors="coerce")
        hist = io.BytesIO()
        H.to_parquet(hist)
        npz = io.BytesIO()
        np.savez(npz, **arrays)
        prof = {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in state["profile"].items()}
        extra = {k: state[k] for k in ("summary", "in_library") if k in state}
        profile = json.dumps(dict(profile=prof, state_format=STATE_FORMAT, **extra), indent=1, default=str).encode()
        for name, body in zip(STATE_FILES, (hist.getvalue(), npz.getvalue(), profile)):     # profile.json last
            self.s3.put_object(Bucket=self.bucket, Key=self._state_key(pid, name), Body=body)
        with self._lock:
            self._cache.pop(pid, None)

    def load_state(self, pid):
        stamp = self.s3.head_object(Bucket=self.bucket, Key=self._state_key(pid, "profile.json"))["ETag"]
        with self._lock:
            hit = self._cache.get(pid)
            if hit and hit[0] == stamp:
                return hit[1]
        get = lambda name: self.s3.get_object(Bucket=self.bucket, Key=self._state_key(pid, name))["Body"].read()
        H = pd.read_parquet(io.BytesIO(get("hist.parquet")))
        a = dict(np.load(io.BytesIO(get("arrays.npz"))))
        for c in ARRAY_COLS:
            H[c] = list(a[c]) if len(H) else []
        pj = json.loads(get("profile.json"))
        prof = {k: (np.nan if v is None else v) for k, v in pj["profile"].items()}
        st = dict(hist=H, profile=prof, own_lib=(a["own_emb"], a["own_y"]), summary=pj.get("summary"),
                  in_library=bool(pj.get("in_library")), state_format=int(pj.get("state_format", 1)))
        with self._lock:
            self._cache[pid] = (stamp, st)
            if len(self._cache) > 512:
                self._cache.pop(next(iter(self._cache)))
        return st
