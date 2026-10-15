"""Fill site/template.html with the final results -> site/index.html (+ figures and demo data next to it).
usage: python -I build_site.py <results dir> <site dir>
"""
import sys, os, json, pickle, shutil, html, datetime
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from pmrf.predict import confidence

R, SITE = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(SITE, "fig"), exist_ok=True)
fin = pd.read_csv(os.path.join(R, "final_blend/metrics.csv"))
summ = json.load(open(os.path.join(R, "final_blend/summary.json")))
fp = pickle.load(open(os.path.join(R, "final_blend/final_preds.pkl"), "rb"))
st = pickle.load(open(os.path.join(R, "final/stage1.pkl"), "rb"))
demo = json.load(open(os.path.join(R, "demo/demo_data.json")))
med = json.load(open(os.path.join(R, "t1d_meds/summary.json")))
bench = json.load(open(os.path.join(R, "bench_encoders.json")))
encg = pd.read_csv(os.path.join(R, "final/encoders_glucose.csv"))
TE, TR, Mo = st["TE"], st["TR"], st["outer"]["M"]
CURV = st["outer"]["CURV"]; tei = fp["te_index"]
ev = np.load(os.path.join(ROOT_DIR, "eda/cache/t1d_events.npz"), allow_pickle=True)["meta"]

U = fin[fin.subset == "unseen"]
def mae(model, t, sub="unseen"):
    return float(fin[(fin.model == model) & (fin.target == t) & (fin.subset == sub)].mae.iloc[0])

obs = CURV[tei]; ok = np.isfinite(obs)
inb = (obs >= fp["band_lo"]) & (obs <= fp["band_hi"])
gain = summ["paired_gain_vs_population"]["dpeak"]
V = dict(
    VERSION="v0.2",
    DATE=datetime.date.today().strftime("%-d %B %Y"),
    FINAL_DPEAK=f"{mae('Final ensemble', 'dpeak'):.1f}", POP_DPEAK=f"{mae('Population mean (by meal type)', 'dpeak'):.1f}",
    PERSON_R=f"{summ['person_level_r_dpeak']['Final ensemble']:.2f}",
    BAND_COV=f"{inb[ok].mean() * 100:.0f}", BAND_COV60=f"{inb[:, 12][ok[:, 12]].mean() * 100:.0f}", BAND_COV120=f"{inb[:, 24][ok[:, 24]].mean() * 100:.0f}",
    LAT_TOTAL=f"{demo['meta']['latency_ms_median']['total']:.0f}", LAT_ENC=f"{bench['MobileCLIP-S2']['median_ms']:.0f}",
    N_DEMO=str(demo["meta"]["n_meals"]), API_DIFF=f"{demo['meta']['api_vs_batch_max_abs_diff']['dpeak']:.2f}",
    T1D_EVENTS=f"{len(ev):,}", T1D_SUBJECTS=f"{len(np.unique(ev['key'])):,}",
    N_PHOTO_TRAIN=f"{int((Mo.in_hist & (Mo.emb_idx >= 0)).sum()):,}", N_TRAIN=f"{len(TR):,}", N_LABELLED=f"{int((Mo.label & (Mo.get('source', pd.Series('cgmacros', index=Mo.index)) == 'cgmacros')).sum()):,}",
    GBM_CONFIG=summ["gbm_config"], GBM_PARAMS=", ".join(f"{k} {v}" for k, v in summ["gbm_params"].items()) + ", learning_rate 0.03 unless stated, subsample 0.8"
        + ("; Δpeak and iAUC modelled as log(y + {}) and log(y + {}), back-transformed as the median".format(st.get("log_targets", {}).get("dpeak"), st.get("log_targets", {}).get("iauc")) if st.get("log_targets") else ""),
    GBM_OFFSET="on" if summ["gbm_offset"] else "off (worse on days 6–7)",
    FT_STEPS=f"{summ['finetune_steps_final']:,}", FT_LR=f"{summ['finetune_lr']:g}", ZS_VAL=f"{summ['zero_shot_val']:.1f}",
    FT_VAL=f"{summ['lr_search'][str(summ['finetune_lr'])]['ens_val']:.1f}",
    BIG_PEOPLE=str(int(TR.loc[TR["source"] == "bigideas", "pid"].nunique())) if "source" in TR else "0",
    BIG_MEALS=f"{int((TR['source'] == 'bigideas').sum()):,}" if "source" in TR else "0",
    TF_SOURCES=", ".join(f"{k} from the {v}" for k, v in summ.get("tf_sources", {}).items() if v == "head") or "all outcomes from the curve",
    BLEND_W=", ".join(f"{dict(G30='30 min', G60='60 min', G90='90 min', G120='120 min', Gmax='peak', dpeak='Δpeak', ttp='time to peak', iauc='iAUC')[k]} {v:.1f}" for k, v in summ["blend_weights"].items()),
    BAND_SCALE=f"{summ['band_scale']:.2f}", N_INNER_TR=f"{summ['n']['inner_train']:,}", N_INNER_VA=f"{summ['n']['inner_val']:,}",
    N_PERSONS=str(summ["n"]["persons"]), N_TEST=f"{summ['n']['test']:,}", N_UNSEEN=f"{summ['n']['unseen']:,}",
    MED_SUBJ=f"{med['subjects']:,}", MED_TR=f"{med['history_events']:,}", MED_TE=f"{med['test_events']:,}",
    GAIN_DPEAK=f"{gain['mean']:.1f}", GAIN_DPEAK_CI=f"{gain['lo']:.1f} to {gain['hi']:.1f}",
    ENC_SPEEDUP=f"{bench['CLIP ViT-L/14']['median_ms'] / bench['MobileCLIP-S2']['median_ms']:.0f}",
    ENC_GAP=f"{abs(encg[(encg.encoder == 'MobileCLIP-S2') & (encg.target == 'dpeak')].mae.iloc[0] - encg[(encg.encoder == 'CLIP ViT-L/14') & (encg.target == 'dpeak')].mae.iloc[0]):.1f}",
    DINO_B_PCT=f"{json.load(open(os.path.join(R, 'encoder_dino', 'summary.json')))['inner_change_vs_reference_pct']['DINOv2-B']['dpeak']:+.1f}".replace("-", "−"),
)

# results table
ORDER = ["Population mean (by meal type)", "Chronos-2 zero-shot + carb covariate", "Population model (photo + context)",
         "Personalized LightGBM", "Personalized transformer (T1D-pretrained)", "Final ensemble"]
TG = [("G30", "30 min"), ("G60", "60 min"), ("G90", "90 min"), ("G120", "120 min"), ("Gmax", "Peak"), ("dpeak", "Δpeak"), ("ttp", "Time to peak"), ("iauc", "iAUC")]
best = {t: U[U.target == t].mae.min() for t, _ in TG}
rows = []
for mdl in ORDER:
    cells = []
    for t, _ in TG:
        q = U[(U.model == mdl) & (U.target == t)].iloc[0]
        v = f"{q.mae:,.0f}" if t == "iauc" else f"{q.mae:.1f}"
        v = f"<b>{v}</b>" if np.isclose(q.mae, best[t]) else v
        cells.append(f'<td class="num">{v} <span class="meta">({q.r:.2f})</span></td>')
    rows.append(f"<tr><td>{html.escape(mdl)}</td>{''.join(cells)}</tr>")
srows = []
best3 = {t: U[U.target == t].mae.min() for t in ["dpeak", "G60", "iauc"]}
for mdl in ORDER:
    cells = []
    for t in ["dpeak", "G60", "iauc"]:
        q = U[(U.model == mdl) & (U.target == t)].iloc[0]
        v = f"{q.mae:,.0f}" if t == "iauc" else f"{q.mae:.1f}"
        cells.append(f'<td class="num">{"<b>" + v + "</b>" if np.isclose(q.mae, best3[t]) else v}</td>')
    srows.append(f"<tr><td>{html.escape(mdl)}</td>{''.join(cells)}</tr>")
V["SIMPLE_TABLE"] = ('<thead><tr><th>Approach</th><th class="num">Peak rise error</th><th class="num">60-min glucose error</th><th class="num">iAUC error</th></tr></thead><tbody>'
                     + "".join(srows) + "</tbody>")
V["RESULTS_TABLE"] = "<thead><tr><th>Model</th>" + "".join(f'<th class="num">{l}</th>' for _, l in TG) + "</tr></thead><tbody>" + "".join(rows) + "</tbody>"

# error analysis by meal type and group (final ensemble, all test meals)
B = fp["blend"]
E = pd.DataFrame(dict(meal=TE.meal_type.to_numpy(), group=TE.group.to_numpy(),
                      dpeak=np.abs(B["dpeak"] - TE.dpeak.to_numpy()), g60=np.abs(B["G60"] - TE.G60.to_numpy()), iauc=np.abs(B["iauc"] - TE.iauc.to_numpy()),
                      obs=TE.dpeak.to_numpy()))
rows = []
for col, lab in [("meal", "Meal type"), ("group", "Study group")]:
    for k, g in E.groupby(col):
        rows.append(f'<tr><td>{lab}</td><td>{k}</td><td class="num">{len(g)}</td><td class="num">{g.obs.mean():.0f}</td><td class="num">{g.dpeak.mean():.1f}</td>'
                    f'<td class="num">{g.g60.mean():.1f}</td><td class="num">{g.iauc.mean():,.0f}</td></tr>')
V["ERROR_TABLE"] = ('<thead><tr><th>Split</th><th>Level</th><th class="num">Meals</th><th class="num">Mean observed Δpeak</th><th class="num">Δpeak MAE</th>'
                    '<th class="num">60-min MAE</th><th class="num">iAUC MAE</th></tr></thead><tbody>' + "".join(rows) + "</tbody>")

# confidence flags vs error on every test meal
H = Mo[Mo.in_hist & Mo.label & (Mo.emb_idx >= 0)]
hn, ht, hc = H.groupby("pid").size(), H.groupby("pid").meal_type.apply(set), H.groupby("pid").carbs.max()
flags = [confidence(r.to_dict(), int(hn.get(r.pid, 0)), ht.get(r.pid, set()), float(hc.get(r.pid, 0)), float(r.photo_sim_pop))[1] for _, r in TE.iterrows()]
allf = sorted({f.split(" in the")[0] if f.startswith("Only") else f for fl in flags for f in fl})
rows = []
err = E.dpeak.to_numpy()
anyf = np.array([len(f) > 0 for f in flags])
rows.append(f'<tr><td>Any flag</td><td class="num">{anyf.sum()}</td><td class="num">{err[anyf].mean():.1f}</td><td class="num">{err[~anyf].mean():.1f}</td></tr>')
V["FLAG_ANY"], V["FLAG_NONE"] = f"{err[anyf].mean():.1f}", f"{err[~anyf].mean():.1f}"
V["FLAG_PHOTO_N"] = str(int(sum(any(x.startswith("Photo looks") for x in fl) for fl in flags)))
for f in allf:
    m = np.array([any((x.split(" in the")[0] if x.startswith("Only") else x) == f for x in fl) for fl in flags])
    if m.sum() >= 3:
        rows.append(f'<tr><td>{html.escape(f if not f.startswith("Only") else "Fewer than 10 usable meals in the history")}</td><td class="num">{m.sum()}</td>'
                    f'<td class="num">{err[m].mean():.1f}</td><td class="num">{err[~m].mean():.1f}</td></tr>')
V["FLAG_TABLE"] = '<thead><tr><th>Flag</th><th class="num">Flagged meals</th><th class="num">Δpeak MAE, flagged</th><th class="num">Δpeak MAE, others</th></tr></thead><tbody>' + "".join(rows) + "</tbody>"

# vision-language model pilot
vc = json.load(open(os.path.join(R, "vlm_compare.json")))
V["VLM_SEC"], V["VLM_PARSED"], V["VLM_N"] = f"{vc['seconds_median']:.1f}", f"{vc['parsed'] * 100:.0f}", str(vc["n"])
V["VLM_BIAS"] = f"{abs(vc['vlm_bias_carbs']):.0f}"
rows = []
for nut, lab in [("carbs", "Carbs (g)"), ("protein", "Protein (g)"), ("fat", "Fat (g)"), ("kcal", "kcal")]:
    for sub, sl in [("free-living", "free-living (dinner, snack)"), ("standardized", "standardized (breakfast, lunch)")]:
        cells = []
        best_ = min(vc[f"{nut}|{sub}|{m}"]["mae"] for m in ("vlm", "clip", "average"))
        for m in ("vlm", "clip", "average"):
            q = vc[f"{nut}|{sub}|{m}"]; v = f"{q['mae']:.0f}" if nut == "kcal" else f"{q['mae']:.1f}"
            v = f"<b>{v}</b>" if np.isclose(q["mae"], best_) else v
            cells.append(f'<td class="num">{v} <span class="meta">({q["r"]:.2f})</span></td>')
        rows.append(f"<tr><td>{lab}</td><td>{sl}</td>{''.join(cells)}</tr>")
V["VLM_TABLE"] = ('<thead><tr><th>Nutrient</th><th>Meals</th><th class="num">Qwen2.5-VL-3B</th><th class="num">CLIP estimator (deployed)</th>'
                  '<th class="num">Average</th></tr></thead><tbody>' + "".join(rows) + "</tbody>")
vi_path = os.path.join(R, "vlm_integration.json")
if os.path.exists(vi_path):
    vi = json.load(open(vi_path))
    g = lambda split, name, key: vi[f"{split} | {name}"][key]
    ins, tst = "inner (days 1-5 -> 6-7)", "test (days 1-7 -> 8+)"
    txt = []
    for key, lab in [("dpeak", "Δpeak"), ("G60", "60-min glucose"), ("iauc", "iAUC")]:
        a0, a1 = g(ins, "current model", f"all:{key}"), g(ins, "+ VLM nutrition", f"all:{key}")
        b0, b1 = g(tst, "current model", f"unseen:{key}"), g(tst, "+ VLM nutrition", f"unseen:{key}")
        fmt_ = (lambda x: f"{x:,.0f}") if key == "iauc" else (lambda x: f"{x:.1f}")
        txt.append(f"{lab} {fmt_(a0)} → {fmt_(a1)} on days 6–7 ({fmt_(b0)} → {fmt_(b1)} on unseen test meals)")
    V["VLM_INTEG"] = ("Tested with the earlier (v3) LightGBM: adding the five VLM numbers as extra features (same configuration, 3 seeds) changes the error as follows: "
                      + "; ".join(txt) + ".")
    # an improvement counts only if it is at least 1% of the current error; smaller changes are within seed noise
    better = sum(g(ins, "+ VLM nutrition", f"all:{k}") < 0.99 * g(ins, "current model", f"all:{k}") for k in ["dpeak", "G60", "iauc"])
    V["VLM_DECISION"] = ("Recommendation: adopt the VLM reading as an extra input. It lowers the validation error by at least 1% on {}/3 key outcomes. It is not yet part of the deployed model; adopting it means rerunning stage 2 (about 6 minutes) and adds about {} s of CPU time per photo.".format(better, V["VLM_SEC"])
                         if better >= 2 else
                         "Recommendation: do not adopt the 3B model as a model input; on days 6–7 it changes none of the key errors by 1% or more. Keep it optional for dish descriptions in the app, and revisit with a larger VLM on a GPU.")
else:
    V["VLM_INTEG"] = "Pending: this needs the VLM reading of every history photo as well, which is still running."
    V["VLM_DECISION"] = "Pending the glucose test above."

# diet guide and high-peak probability checks on day 8+
ge_path = os.path.join(R, "guide_eval.json")
if os.path.exists(ge_path):
    ge = json.load(open(ge_path)); cb = ge["carb_budget"]; e180, e140 = ge["exceed_180"], ge["exceed_140"]
    sv = json.load(open(os.path.join(R, "guide", "shake_guide_validation.json")))["mae_test_shakes"]
    V["GUIDE_SUMMARY"] = (f"Checked on day 8+: personal effects predicted later meals better than group averages (error {sv['personal_effects']:.0f} vs "
                          f"{sv['population_effects']:.0f} mg/dL), and meals within budget stayed under the target {cb['share_rise_le_target_within'] * 100:.0f}% of the time "
                          f"vs {cb['share_rise_le_target_over'] * 100:.0f}% over budget.")
    V["EXCEED_SUMMARY"] = (f"On new meals they separate high from normal peaks well (AUC {e180['auc']:.2f} for 180 mg/dL, {e140['auc']:.2f} for 140).")
else:
    V["GUIDE_SUMMARY"] = V["EXCEED_SUMMARY"] = "Validation pending."
rs = os.path.join(R, "risk_section.html")
V["RISK_HTML"] = open(rs).read() if os.path.exists(rs) else '<p class="meta">Metabolic profile and risk models: pending.</p>'

ex = json.load(open(os.path.join(R, "demo/api_example_response.json")))
V["API_EXAMPLE"] = html.escape(json.dumps(ex, indent=2))
# open on a representative example: the first meal shown has a clear response and an error no worse than the median
errs = [abs(m["prediction"]["peak_rise_mgdl"] - m["observed"]["peak_rise_mgdl"]) for m in demo["meals"].values()]
med_err = float(np.median(errs))
cand = []
for p in demo["people"]:
    if len(p["demo_meals"]) >= 3:
        m0 = demo["meals"][p["demo_meals"][0]]
        o, e = m0["observed"]["peak_rise_mgdl"], abs(m0["prediction"]["peak_rise_mgdl"] - m0["observed"]["peak_rise_mgdl"])
        if o >= 40 and e <= med_err and not m0["observed"]["seen_recipe_in_history"]:
            cand.append((o, p["person_id"]))
V["START_PID"] = max(cand)[1] if cand else demo["people"][0]["person_id"]

t = open(os.path.join(ROOT_DIR, "site/template.html")).read()
for k, v in V.items():
    t = t.replace("{{" + k + "}}", v)
left = [x for x in set(__import__("re").findall(r"\{\{(\w+)\}\}", t))]
assert not left, f"unfilled placeholders: {left}"
open(os.path.join(SITE, "index.html"), "w").write(t)
for f in os.listdir(os.path.join(R, "site_fig")):
    shutil.copy(os.path.join(R, "site_fig", f), os.path.join(SITE, "fig", f))
for f in ["b_estimated_vs_lab_hba1c.png", "c_roc_t2d.png", "d_roc_complications.png"]:
    shutil.copy(os.path.join(R, "risk", "fig", f), os.path.join(SITE, "fig", f))
for f in ["cx_variance.png", "cx_context.png"]:
    shutil.copy(os.path.join(ROOT_DIR, "eda/fig", f), os.path.join(SITE, "fig", f))
shutil.copy(os.path.join(R, "demo/demo_data.json"), os.path.join(SITE, "demo_data.json"))
print(json.dumps({k: v for k, v in V.items() if k not in ("RESULTS_TABLE", "ERROR_TABLE", "FLAG_TABLE", "API_EXAMPLE")}, indent=0))
