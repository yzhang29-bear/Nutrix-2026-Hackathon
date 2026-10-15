"""Plot the predicted 0-120 min glucose curve of a /predict response, with its 80% band, and the observed curve when
the response has `observed` (the API adds it when the CGM after the meal is already stored), with a table of
predicted vs observed outcomes.
usage: python scripts/plot_prediction.py [prediction.json] [out.png]
Defaults: the newest tests/integration/out/predict_*.json (written by scripts/predict.sh), PNG next to it.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path(__file__).resolve().parents[1] / "tests" / "integration" / "out"
src = Path(sys.argv[1]) if len(sys.argv) > 1 else max(out_dir.glob("predict_*.json"), key=lambda f: f.stat().st_mtime)
png = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")

d = json.loads(src.read_text(encoding="utf8"))
p, c = d["prediction"], d["prediction"]["curve"]
m = c["minutes"]

obs = d.get("observed")
if obs is None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
else:
    fig, (ax, tax) = plt.subplots(1, 2, figsize=(12.5, 4.5), width_ratios=[2.2, 1])
ax.fill_between(m, c["lower80"], c["upper80"], color="tab:blue", alpha=0.18, label="80% band")
ax.plot(m, c["glucose_mgdl"], color="tab:blue", lw=2.5, label="predicted glucose")
for v, col in [(140, "tab:orange"), (180, "tab:red")]:
    ax.axhline(v, color=col, ls="--", lw=1)
    ax.text(m[-1], v, f" {v}  P={p['prob_peak_above_mgdl'][str(v)]:.0%}", color=col, va="center", fontsize=9)
# The peak is predicted on its own, not read off the curve: the mean curve is smoother, so its top is lower
ax.plot(p["time_to_peak_min"], p["peak_glucose_mgdl"], "D", color="tab:purple",
        label=f"predicted peak {p['peak_glucose_mgdl']:.0f} (+{p['peak_rise_mgdl']:.0f}) at {p['time_to_peak_min']:.0f} min")

if obs is not None:
    ax.plot(obs["curve"]["minutes"], obs["curve"]["glucose_mgdl"], color="black", lw=2,
            label=f"observed glucose ({obs['share_of_curve_inside_band80']:.0%} inside the band)")
    ax.plot(obs["time_to_peak_min"], obs["peak_glucose_mgdl"], "o", color="black", label=f"observed peak {obs['peak_glucose_mgdl']:.0f} "
            f"(+{obs['peak_rise_mgdl']:.0f}) at {obs['time_to_peak_min']:.0f} min")
    rows = [(f"glucose at {k} min", p["glucose_at_min"][k], obs["glucose_at_min"][k]) for k in ("30", "60", "90", "120")]
    rows += [("peak glucose", p["peak_glucose_mgdl"], obs["peak_glucose_mgdl"]), ("peak rise", p["peak_rise_mgdl"], obs["peak_rise_mgdl"]),
             ("time to peak (min)", p["time_to_peak_min"], obs["time_to_peak_min"]),
             ("iAUC 0-120 min", p["iauc_0_120_mgdl_min"], obs["iauc_0_120_mgdl_min"])]
    tax.axis("off")
    tab = tax.table(cellText=[[name, f"{a:.0f}", f"{b:.0f}", f"{a - b:+.0f}"] for name, a, b in rows],
                    colLabels=["", "predicted", "observed", "error"], cellLoc="right", colWidths=[0.46, 0.18, 0.18, 0.18], bbox=[0, 0.05, 1, 0.9])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("0.85")
        if j == 0:
            cell.set_text_props(ha="left")

n = d["meal"]["estimated_nutrition"]
ax.set_title(f"{d['person_id']}  {d['meal']['time']} ({d['meal']['meal_type']})  "
             f"carbs {n['carbs_g']:.0f} g  confidence {d['confidence']['level']}", fontsize=10)
ax.set_xlabel("minutes after the meal")
ax.set_ylabel("glucose (mg/dL)")
ax.set_xticks(range(0, 121, 15))
ax.set_xlim(0, 120)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(png, dpi=130)
print(png)
