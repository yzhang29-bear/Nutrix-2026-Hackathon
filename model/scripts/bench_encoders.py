"""Single-photo CPU latency of the three image encoders compared in the project (8 threads, 20 timed runs after warm-up).
usage: python -I bench_encoders.py <photo.jpg> <out.json>
"""
import sys, json, time
import numpy as np, torch, torchvision, open_clip
from PIL import Image, ImageOps

PHOTO, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(8)
img = ImageOps.exif_transpose(Image.open(PHOTO)).convert("RGB")
res = {}

def bench(name, pre, fwd, params):
    with torch.inference_mode():
        for _ in range(3):
            fwd(pre(img).unsqueeze(0))
        ts = []
        for _ in range(20):
            t = time.perf_counter(); fwd(pre(img).unsqueeze(0)); ts.append((time.perf_counter() - t) * 1000)
    res[name] = dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)), params_m=params / 1e6)
    print(name, res[name], flush=True)

w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
m = torchvision.models.resnet50(weights=w).eval(); m.fc = torch.nn.Identity()
bench("ResNet50 (ImageNet)", w.transforms(), m, sum(p.numel() for p in m.parameters()))
for name, arch, tag in [("MobileCLIP-S2", "MobileCLIP-S2", "datacompdr"), ("CLIP ViT-L/14", "ViT-L-14", "openai")]:
    m, _, pre = open_clip.create_model_and_transforms(arch, pretrained=tag); m.eval()
    bench(name, pre, m.encode_image, sum(p.numel() for p in m.visual.parameters()))
json.dump(res, open(OUT, "w"), indent=1)
