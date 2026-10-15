"""Embed every before-meal photo with an ImageNet ResNet50 (CPU) for the photo->macro baseline.
usage: python -I photo_embed.py <CGMacros dir> <out.npz>
"""
import sys, os, glob
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image, ImageOps

ROOT, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(32)
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "CGMacros-*", "CGMacros-*.csv"))):
    sid = os.path.basename(f)[9:12]
    d = pd.read_csv(f)
    d.columns = [c.strip() for c in d.columns]
    m = d[d["Meal Type"].notna() & d["Image path"].notna()]
    for _, r in m.iterrows():
        p = os.path.join(ROOT, f"CGMacros-{sid}", r["Image path"])
        if os.path.exists(p):
            rows.append((sid, r["Timestamp"], p))
print("photos to embed:", len(rows), flush=True)
w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=w).eval()
model.fc = torch.nn.Identity()
tf = w.transforms()
embs = []
with torch.inference_mode():
    for i in range(0, len(rows), 64):
        batch = torch.stack([tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for _, _, p in rows[i:i + 64]])
        embs.append(model(batch).numpy().astype(np.float32))
        print(i + len(batch), flush=True)
np.savez_compressed(OUT, emb=np.concatenate(embs), sid=np.array([r[0] for r in rows]), ts=np.array([r[1] for r in rows]),
                    path=np.array([os.path.relpath(r[2], ROOT) for r in rows]))
print("saved", OUT)
