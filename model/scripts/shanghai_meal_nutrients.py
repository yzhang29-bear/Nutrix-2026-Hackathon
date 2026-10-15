"""Shanghai T1DM/T2DM meal records (food items with weights, English + Chinese) -> carbs/protein/fat/fiber (g) with the
Qwen3-235B text model on AWS Bedrock. One call per distinct meal text (identical records get identical values); resumable.
usage: AWS_PROFILE=hackathon python -I shanghai_meal_nutrients.py <shanghai root> <out.jsonl> [limit]
"""
import sys, os, re, json, glob, time, hashlib, threading, warnings
import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT, OUT = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODEL = "qwen.qwen3-235b-a22b-2507-v1:0"
rt = boto3.Session().client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-west-2"),
                            config=Config(retries=dict(max_attempts=10, mode="adaptive"), read_timeout=120, max_pool_connections=32))
warnings.simplefilter("ignore")
NA = {"", "nan", "data not available", "数据不可用", "未记录", "无", "不详"}


def meal_texts(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "Shanghai_T*DM", "*.xls*"))):
        if os.path.basename(f).startswith("~$"):
            continue
        d = pd.read_excel(f)
        d.columns = [str(c).strip() for c in d.columns]
        if "Dietary intake" not in d:
            continue
        for _, r in d[d["Dietary intake"].notna()].iterrows():
            en = str(r["Dietary intake"]).strip()
            zh = str(r.get("饮食", "")).strip()
            if en.lower() in NA or zh in NA:
                continue
            k = hashlib.sha1(f"{en}||{zh}".encode()).hexdigest()[:16]
            out[k] = (en, zh)
    return out


PROMPT = """You are a clinical dietitian in Shanghai. Below is one meal record (the same foods in English and in Chinese) with the
amount of each food as eaten. Compute the meal's total carbohydrate, protein, fat and dietary fiber in grams, using standard
Chinese food composition values. Amounts refer to the food as served (米饭 = cooked rice, 大米/生米 = raw rice, 馒头 = steamed
bun; ml of milk or soy milk ~ g). If an item has no amount, assume one typical serving. Reply with JSON only:
{{"items": [{{"name": str, "grams": number, "carbs_g": number}}], "carbs_g": number, "protein_g": number, "fat_g": number, "fiber_g": number}}

English: {en}
中文: {zh}"""


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def job(k, en, zh):
    r = rt.converse(modelId=MODEL, messages=[{"role": "user", "content": [{"text": PROMPT.format(en=en, zh=zh)}]}],
                    inferenceConfig={"maxTokens": 600, "temperature": 0})
    text = r["output"]["message"]["content"][0]["text"]
    return dict(key=k, en=en, zh=zh, raw=text, parsed=parse(text), in_tok=r["usage"]["inputTokens"], out_tok=r["usage"]["outputTokens"])


texts = meal_texts(ROOT)
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["key"] for l in open(OUT)}
todo = [(k, *v) for k, v in texts.items() if k not in done]
if LIMIT:
    todo = todo[:LIMIT]
print(f"{len(texts)} distinct meal texts, {len(done)} done, {len(todo)} to do", flush=True)
t0, n, bad, lock = time.time(), 0, 0, threading.Lock()
with open(OUT, "a") as f, ThreadPoolExecutor(int(os.environ.get("QWEN_WORKERS", "8"))) as ex:
    futs = {ex.submit(job, *a): a[0] for a in todo}
    for fu in as_completed(futs):
        try:
            res = fu.result()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            print("ERROR", futs[fu], code, str(e)[:160], flush=True)
            if code in ("ExpiredTokenException", "UnrecognizedClientException", "AccessDeniedException"):
                ex.shutdown(cancel_futures=True); break
            continue
        with lock:
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            n += 1; bad += res["parsed"] is None
            if n % 100 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s unparsed {bad}", flush=True)
print(f"finished {n}, unparsed {bad}, {time.time() - t0:.0f}s", flush=True)
