"""Training bundle (pickle, our side only) -> versioned model store in stable formats for the service (pmrf/store.py).
usage: python -I export_model_store.py <results dir> <out store dir> <version>
"""
import sys, os, json, pickle, glob
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np
from pmrf.store import export_store
from pmrf.vision import PhotoEncoder, FOOD_PROMPTS

RES, OUT, VERSION = sys.argv[1], sys.argv[2], sys.argv[3]
DEMO_DIR = sys.argv[4] if len(sys.argv) > 4 else "demo"           # e.g. demo_v7 for the v7 build
bundle = pickle.load(open(os.path.join(RES, DEMO_DIR, "bundle.pkl"), "rb"))
risk = pickle.load(open(os.path.join(RES, "risk", "models.pkl"), "rb"))
prof = json.load(open(os.path.join(RES, "risk", "cgmacros_profiles.json")))
risk["t2d_tier_reference"] = sorted(float(v["t2d_risk"]["probability"]) for v in prof.values() if v.get("t2d_risk", {}).get("available"))
effects = pickle.load(open(os.path.join(RES, "guide", "shake_effects.pkl"), "rb"))
enc = PhotoEncoder(*bundle["encoder"])
text_emb = enc.encode_texts(FOOD_PROMPTS)
w = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--apple--MobileCLIP-S2-OpenCLIP/snapshots/*/open_clip_model.safetensors"))
assert len(w) == 1, w
man = export_store(bundle, os.path.join(OUT, VERSION), VERSION, w[0], text_emb, risk, effects,
                   notes=os.environ.get("STORE_NOTES", "v4: CGMacros + BIG IDEAs, MobileCLIP-S2, LightGBM x3 seeds + in-context transformer x3 seeds"))
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(os.path.join(OUT, VERSION)) for f in fs)
print(f"exported {VERSION}: {len(man['files'])} files, {size / 1e6:.1f} MB")
