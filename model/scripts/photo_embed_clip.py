"""Embed before-meal photos with an OpenCLIP model (CPU).
usage: python -I photo_embed_clip.py <CGMacros dir> <out.npz> [model] [pretrained]"""
import sys, os, glob
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image, ImageOps
ROOT, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(16)
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "CGMacros-*", "CGMacros-*.csv"))):
    sid = os.path.basename(f)[9:12]
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
    for _, r in d[d["Meal Type"].notna() & d["Image path"].notna()].iterrows():
        p = os.path.join(ROOT, f"CGMacros-{sid}", r["Image path"])
        if os.path.exists(p): rows.append((sid, r["Timestamp"], p))
print("photos:", len(rows), flush=True)
MODEL = sys.argv[3] if len(sys.argv) > 3 else "ViT-L-14"; PRE = sys.argv[4] if len(sys.argv) > 4 else "openai"
model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRE)
model.eval()
embs = []
with torch.inference_mode():
    for i in range(0, len(rows), 32):
        x = torch.stack([pre(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for _, _, p in rows[i:i + 32]])
        e = model.encode_image(x); e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.numpy().astype(np.float32)); print(i + len(x), flush=True)
np.savez_compressed(OUT, emb=np.concatenate(embs), sid=np.array([r[0] for r in rows]), ts=np.array([r[1] for r in rows]),
                    path=np.array([os.path.relpath(r[2], ROOT) for r in rows]))
print("saved", OUT)
