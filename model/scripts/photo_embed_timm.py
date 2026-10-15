"""Embed before-meal photos with a timm vision backbone (CPU), e.g. DINOv2. Output layout matches photo_embed_clip.py.
Feature = [CLS token, mean of patch tokens] (the DINOv2 linear-probe recipe), L2-normalised so cosine retrieval works.
usage: python -I photo_embed_timm.py <CGMacros dir> <out.npz> <timm model> [image size]
"""
import sys, os, glob, time
import numpy as np, pandas as pd, torch, timm
from PIL import Image, ImageOps
ROOT, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 224
torch.set_num_threads(16)
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "CGMacros-*", "CGMacros-*.csv"))):
    sid = os.path.basename(f)[9:12]
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
    for _, r in d[d["Meal Type"].notna() & d["Image path"].notna()].iterrows():
        p = os.path.join(ROOT, f"CGMacros-{sid}", r["Image path"])
        if os.path.exists(p): rows.append((sid, r["Timestamp"], p))
print("photos:", len(rows), flush=True)
model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=SIZE).eval()
cfg = timm.data.resolve_model_data_config(model)
cfg.update(input_size=(3, SIZE, SIZE), crop_pct=0.875)
pre = timm.data.create_transform(**cfg, is_training=False)
npt = getattr(model, "num_prefix_tokens", 1)
embs, t0 = [], time.time()
with torch.inference_mode():
    for i in range(0, len(rows), 32):
        x = torch.stack([pre(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for _, _, p in rows[i:i + 32]])
        tok = model.forward_features(x)
        e = torch.cat([tok[:, 0], tok[:, npt:].mean(1)], 1)
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.numpy().astype(np.float32)); print(i + len(x), f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(OUT, emb=np.concatenate(embs), sid=np.array([r[0] for r in rows]), ts=np.array([r[1] for r in rows]),
                    path=np.array([os.path.relpath(r[2], ROOT) for r in rows]))
print("saved", OUT, np.concatenate(embs).shape)
