"""Qwen3-VL-235B (AWS Bedrock) on CGMacros meal photos. Resumable (one JSON line per photo; finished photos are skipped).

mode zeroshot      : one call per photo -> dish, items with grams, nutrition and glucose-relevant food attributes
mode fewshot <cut> : nutrition calibrated on the SAME person's own earlier meals: the K most similar photos (MobileCLIP
                     cosine) among that person's meals on days 1..<cut>, each shown with its logged nutrition; the query
                     meal itself is never an example. <cut> = history cut-off of the split the estimate is used in
                     (4/5/6 = rolling inner validation, 7 = final model); queries = photos on days 1..7 (or all days for 7).
usage: AWS_PROFILE=hackathon python -I qwen_bedrock.py <stage1.pkl> <embeddings.npz> <CGMacros dir> <out dir> zeroshot|fewshot [cut] [limit]
"""
import sys, os, io, re, json, time, threading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd, pickle
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps

S1, EMB, DATA, OUT, MODE = sys.argv[1:6]
CUT = int(sys.argv[6]) if MODE == "fewshot" else None
LIMIT = int(sys.argv[7] if MODE == "fewshot" else (sys.argv[6] if len(sys.argv) > 6 else 0)) if len(sys.argv) > (7 if MODE == "fewshot" else 6) else 0
MODEL = os.environ.get("VLM_MODEL", "qwen.qwen3-vl-235b-a22b")      # any Bedrock vision model (e.g. us.anthropic.claude-sonnet-5-5)
K, WORKERS = 4, int(os.environ.get("QWEN_WORKERS", "8"))
DAYS = [int(d) for d in os.environ.get("VLM_DAYS", "").split(",") if d]  # optional: only photos from these study days
os.makedirs(OUT, exist_ok=True)
outp = os.path.join(OUT, "zeroshot.jsonl" if MODE == "zeroshot" else f"fewshot_c{CUT}.jsonl")
rt = boto3.Session().client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-west-2"),
                            config=Config(retries=dict(max_attempts=10, mode="adaptive"), read_timeout=120, max_pool_connections=32))

st = pickle.load(open(S1, "rb"))
M = st["outer"]["M"]
M = M[(M.source == "cgmacros") & M.photo.notna()].copy()
z = np.load(EMB, allow_pickle=True)
key = {p: i for i, p in enumerate(z["path"])}
M["rel"] = M.photo.map(lambda p: os.path.relpath(p, DATA))
M = M[M.rel.isin(key)]
E = z["emb"][[key[r] for r in M.rel]]
M["eidx"] = np.arange(len(M))


def jpeg(path, side):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((side, side))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return b.getvalue()


def num(v):
    return "unknown" if v is None or not np.isfinite(v) else f"{v:.0f}"


ZS_PROMPT = """You are a dietitian. Estimate what is in this meal photo as it will be eaten (the serving shown).
Reply with JSON only, no other text:
{"dish": short name, "items": [{"name": str, "grams": number}], "carbs_g": number, "sugar_g": number, "protein_g": number,
 "fat_g": number, "fiber_g": number, "kcal": number,
 "form": "liquid" | "mixed" | "solid", "sugary_drink": true|false, "glycemic_index": "low" | "medium" | "high",
 "refined_carb_share": number between 0 and 1 (share of carbs from refined grains/sugar),
 "fried": true|false, "whole_grain": true|false, "vegetable_share": number between 0 and 1, "portion": "small" | "medium" | "large",
 "cuisine": str, "confidence": number between 0 and 1}"""

FS_INTRO = ("You are a dietitian. Below are earlier meals of the SAME person, each with the nutrition they logged. The person "
            "eats similar portions and often the same foods or restaurants, so use these examples to calibrate portion size; if the "
            "new meal is the same food as an example, reuse that example's numbers.")
FS_ASK = ('New meal ({mt}, {hh}). Estimate its nutrition as it will be eaten. Reply with JSON only: {{"carbs_g": number, "protein_g": number, '
          '"fat_g": number, "fiber_g": number, "same_as_example": example number or null, "confidence": number between 0 and 1}}')


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = m.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", s))
        except json.JSONDecodeError:
            return None


def call(content, max_tokens):
    cfg = {"maxTokens": max_tokens} if "anthropic" in MODEL else {"maxTokens": max_tokens, "temperature": 0}   # newer Claude models reject temperature
    r = rt.converse(modelId=MODEL, messages=[{"role": "user", "content": content}], inferenceConfig=cfg)
    return r["output"]["message"]["content"][0]["text"], r["usage"]


def job(i):
    r = M.loc[i]
    t = time.time()
    if MODE == "zeroshot":
        content = [{"image": {"format": "jpeg", "source": {"bytes": jpeg(r.photo, 768)}}}, {"text": ZS_PROMPT}]
        examples = []
        text, usage = call(content, 700)
    else:
        pool = M[(M.pid == r.pid) & (M.day <= CUT) & (M.index != i) & M.carbs.notna()]
        examples = []
        content = [{"text": FS_INTRO}]
        if len(pool):
            s = E[pool.eidx.to_numpy()] @ E[r.eidx]
            for k, j in enumerate(pool.index[np.argsort(-s)[:K]], 1):
                e = M.loc[j]
                examples.append(int(j))
                content += [{"image": {"format": "jpeg", "source": {"bytes": jpeg(e.photo, 448)}}},
                            {"text": f"Example {k} ({e.meal_type}): logged carbs {num(e.carbs)} g, protein {num(e.protein)} g, "
                                     f"fat {num(e.fat)} g, fiber {num(e.fiber)} g."}]
        content += [{"image": {"format": "jpeg", "source": {"bytes": jpeg(r.photo, 768)}}},
                    {"text": FS_ASK.format(mt=r.meal_type, hh=pd.Timestamp(r.time).strftime("%H:%M"))}]
        text, usage = call(content, 200)
    return dict(meal=int(i), pid=r.pid, rel=r.rel, day=int(r.day), examples=examples, raw=text, parsed=parse(text),
                in_tok=usage["inputTokens"], out_tok=usage["outputTokens"], s=round(time.time() - t, 2))


done = set()
if os.path.exists(outp):
    for line in open(outp):
        try:
            done.add(json.loads(line)["meal"])
        except Exception:
            pass
q = M if MODE == "zeroshot" or CUT >= 7 else M[M.day <= 7]
if DAYS:
    q = q[q.day.isin(DAYS)]
todo = [i for i in q.index if int(i) not in done]
if LIMIT:
    todo = todo[:LIMIT]
print(f"{MODE} cut={CUT}: {len(q)} photos, {len(done)} done, {len(todo)} to do, {WORKERS} workers", flush=True)
lock, t0, n, tok, fails = threading.Lock(), time.time(), 0, [0, 0], 0
with open(outp, "a") as f, ThreadPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(job, i): i for i in todo}
    for fu in as_completed(futs):
        try:
            res = fu.result()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            fails += 1
            print("ERROR", futs[fu], code, str(e)[:200], flush=True)
            if code in ("ExpiredTokenException", "UnrecognizedClientException", "AccessDeniedException"):
                print("credentials expired or access denied: stopping (rerun to resume)", flush=True)
                ex.shutdown(cancel_futures=True)
                break
            continue
        with lock:
            f.write(json.dumps(res) + "\n"); f.flush()
            n += 1; tok[0] += res["in_tok"]; tok[1] += res["out_tok"]
            bad = bad + (res["parsed"] is None) if "bad" in dir() else int(res["parsed"] is None)
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s tokens in {tok[0]:,} out {tok[1]:,} unparsed {bad}", flush=True)
print(f"finished {n} calls, {fails} errors, tokens in {tok[0]:,} out {tok[1]:,}, {time.time() - t0:.0f}s", flush=True)
