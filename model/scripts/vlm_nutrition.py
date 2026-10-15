"""Zero-shot nutrition estimates from a vision-language model (Qwen2.5-VL-3B-Instruct, CPU) for day-8+ test photos.
Resumable: appends one CSV row per photo and skips photos already done.
usage: python -I vlm_nutrition.py <stage1 dir> <out.csv> [threads] [all]
"""
import sys, os, re, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, pandas as pd
from PIL import Image, ImageOps
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

S1, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 32)
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
Mo = st["outer"]["M"]
ALL = len(sys.argv) > 4 and sys.argv[4] == "all"     # also history photos (test photos first), for model-integration tests
todo = Mo[(Mo.emb_idx >= 0) & (True if ALL else ~Mo.in_hist)].assign(_o=lambda d: d.in_hist.astype(int)).sort_values(["_o", "pid", "time"])
todo = todo[["pid", "time", "meal_type", "photo"]]
done = set(pd.read_csv(OUT).photo) if os.path.exists(OUT) else set()
todo = todo[~todo.photo.isin(done)]
print(f"{len(todo)} photos to do ({len(done)} already done)", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu").eval()
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=448 * 448)
PROMPT = ("Estimate the nutrition of everything shown in this {meal} photo, as one person would eat it. "
          "Reply with JSON only, no other text: {{\"dish\": \"short description\", \"carbs_g\": number, \"protein_g\": number, "
          "\"fat_g\": number, \"fiber_g\": number, \"kcal\": number}}")
first = not os.path.exists(OUT)
for k, r in enumerate(todo.itertuples()):
    t = time.perf_counter()
    img = ImageOps.exif_transpose(Image.open(r.photo)).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(meal=r.meal_type)}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=96, do_sample=False)
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", ans, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    row = dict(pid=r.pid, time=str(r.time), meal_type=r.meal_type, photo=r.photo, dish=j.get("dish"),
               carbs_g=j.get("carbs_g"), protein_g=j.get("protein_g"), fat_g=j.get("fat_g"), fiber_g=j.get("fiber_g"), kcal=j.get("kcal"),
               raw=ans.replace("\n", " ")[:400], seconds=round(time.perf_counter() - t, 1))
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=first, index=False); first = False
    if k % 10 == 0:
        print(f"{k + 1}/{len(todo)}  {row['seconds']} s  {row['dish']}  carbs {row['carbs_g']}", flush=True)
print("done")
