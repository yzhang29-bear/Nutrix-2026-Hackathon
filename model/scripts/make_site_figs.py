"""Static result figures for the project site.  usage: python -I make_site_figs.py <results dir> <out fig dir>"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eda_common import save_fig, GROUPS
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

R, F = sys.argv[1], sys.argv[2]
fin = pd.read_csv(os.path.join(R, "final_blend/metrics.csv"))
summ = json.load(open(os.path.join(R, "final_blend/summary.json")))
abl = pd.read_csv(os.path.join(R, "final/ablations.csv"))
encg = pd.read_csv(os.path.join(R, "final/encoders_glucose.csv"))
encp = pd.read_csv(os.path.join(R, "final/encoders_photo.csv"))
bench = json.load(open(os.path.join(R, "bench_encoders.json")))
med = pd.read_csv(os.path.join(R, "t1d_meds/metrics.csv"))
tfs = json.load(open(os.path.join(R, "tf_v1/summary.json")))
tfm = pd.read_csv(os.path.join(R, "tf_v1/metrics.csv"))
fp = pickle.load(open(os.path.join(R, "final_blend/final_preds.pkl"), "rb"))
st = pickle.load(open(os.path.join(R, "final/stage1.pkl"), "rb"))
TE = st["TE"]; CURV = st["outer"]["CURV"]; PRE = st["outer"]["PRE"]

ORDER = ["Population mean (by meal type)", "Chronos-2 zero-shot + carb covariate", "Population model (photo + context)",
         "Personalized LightGBM", "Personalized transformer (T1D-pretrained)", "Final ensemble"]

def f_models(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    d = fin[fin.subset == "unseen"]
    for ax, t, lab in [(axes[0], "dpeak", "Peak rise (mg/dL)"), (axes[1], "G60", "Glucose at 60 min (mg/dL)"), (axes[2], "iauc", "iAUC 0–120 min (mg/dL·min)")]:
        q = d[d.target == t].set_index("model").reindex(ORDER)
        cols = [T["grid"], T["ref"], T["ref"], T["s"][1], T["s"][2], T["s"][0]]
        y = np.arange(len(ORDER))
        ax.barh(y, q.mae, color=cols, height=0.66)
        ax.errorbar(q.mae, y, xerr=[q.mae - q.lo, q.hi - q.mae], fmt="none", ecolor=T["ink2"], lw=1.2, capsize=3)
        for yi, v, h in zip(y, q.mae, q.hi):
            ax.text(h + q.hi.max() * 0.02, yi, f"{v:,.0f}" if t == "iauc" else f"{v:.1f}", va="center", fontsize=9, color=T["ink"])
        ax.set_title(lab); ax.set_xlabel("mean absolute error, 95% CI (bootstrap by person)")
        ax.grid(axis="y", visible=False); ax.set_xlim(0, q.hi.max() * 1.22)
    axes[0].set_yticks(np.arange(len(ORDER)), ORDER); axes[0].invert_yaxis()
    return fig
save_fig(f_models, "r_models", F)

def f_ablation(T):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    base = abl[abl.model == "All modalities"].set_index("target").mae
    names = [m for m in abl.model.unique() if m != "All modalities"]
    for ax, t, lab in [(axes[0], "dpeak", "Peak rise"), (axes[1], "iauc", "iAUC 0–120 min")]:
        inc = [(abl[(abl.model == n) & (abl.target == t)].mae.iloc[0] / base[t] - 1) * 100 for n in names]
        o = np.argsort(inc)[::-1]
        ax.barh(np.arange(len(names)), np.array(inc)[o], color=[T["s"][0] if v > 1 else T["ref"] for v in np.array(inc)[o]], height=0.62)
        for yi, v in enumerate(np.array(inc)[o]):
            ax.text(max(v, 0) + 0.3, yi, f"{v:+.1f}%", va="center", fontsize=9, color=T["ink"])
        ax.axvline(0, color=T["ink2"], lw=1); ax.set_title(lab); ax.set_xlabel("error increase when the modality is removed")
        ax.grid(axis="y", visible=False)
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(names)), [names[i] for i in o]); ax.invert_yaxis()
    return fig
save_fig(f_ablation, "r_ablation", F)

def f_encoders(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9), gridspec_kw=dict(wspace=0.35))
    encs = ["ResNet50 (ImageNet)", "MobileCLIP-S2", "CLIP ViT-L/14"]
    cols = [T["ref"], T["s"][0], T["s"][2]]
    ax = axes[0]
    q = encp[(encp.nutrient == "carbs")]
    for k, sub in enumerate(["free-living (dinner, snack)", "standardized (breakfast, lunch)"]):
        v = [q[(q.encoder == e) & (q.subset == sub)].r.iloc[0] for e in encs]
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, v, width=0.34, color=cols, alpha=1 if k == 0 else 0.45)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=T["ink2"], label="free-living (dinner, snack)"), Patch(color=T["ink2"], alpha=0.45, label="standardized (breakfast, lunch)")],
              fontsize=8, loc="upper left")
    ax.set_xticks(range(3), ["ResNet50", "MobileCLIP-S2", "ViT-L/14"]); ax.set_ylabel("correlation, estimated vs logged carbs")
    ax.set_title("Carbs from the photo"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 0.62)
    ax = axes[1]
    for j, (t, mk, lab) in enumerate([("dpeak", "o", "Δpeak"), ("G60", "s", "glucose at 60 min")]):
        v = [encg[(encg.encoder == e) & (encg.target == t)].mae.iloc[0] for e in encs]
        ax.plot(range(3), v, color=T["grid"], lw=1, zorder=1)
        ax.scatter(range(3), v, c=cols, s=70, marker=mk, edgecolor=T["panel"], lw=1.5, zorder=2, label=lab)
        for i, x in enumerate(v): ax.text(i + 0.08, x, f"{x:.1f}", va="center", fontsize=8.5, color=T["ink"])
    ax.set_xlim(-0.4, 2.6); ax.set_xticks(range(3), ["ResNet50", "MobileCLIP-S2", "ViT-L/14"])
    ax.set_ylabel("MAE on unseen meals (mg/dL)"); ax.set_title("Downstream glucose error (dot plot)"); ax.legend(fontsize=8, loc="upper right")
    ax = axes[2]
    lat = [bench[e]["median_ms"] for e in encs]; prm = [bench[e]["params_m"] for e in encs]
    ax.bar(range(3), lat, color=cols, width=0.6)
    for i, (x, pm) in enumerate(zip(lat, prm)): ax.text(i, x + max(lat) * 0.03, f"{x:.0f} ms\n{pm:.0f} M params", ha="center", va="bottom", fontsize=8.5, color=T["ink"])
    ax.set_xticks(range(3), ["ResNet50", "MobileCLIP-S2", "ViT-L/14"]); ax.set_ylabel("CPU latency per photo (ms, 8 threads)")
    ax.set_ylim(0, max(lat) * 1.3); ax.set_title("Speed"); ax.grid(axis="x", visible=False)
    return fig
save_fig(f_encoders, "r_encoders", F)

def f_transformer(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9), gridspec_kw=dict(wspace=0.32))
    ax = axes[0]
    h = np.array(tfs["pretrain_hist"])
    ax.plot(h[:, 0], h[:, 2], "o-", color=T["s"][0], ms=4, label="held-out T1D subjects")
    ax.set_xlabel("pretraining step (batch 128)"); ax.set_ylabel("pinball loss (curve / 50 mg/dL)"); ax.set_title("Pretraining on 671k T1D meals")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    if "finetune_curves" in summ:
        for (lr, hs), col in zip(summ["finetune_curves"].items(), [T["s"][1], T["s"][0], T["s"][2]]):
            H_ = np.array([[x[2] for x in h] for h in hs]); stp = [x[0] for x in hs[0]]
            ax.plot(stp, H_.mean(0), color=col, lw=2, label=f"lr {float(lr):g}" + (" (chosen)" if float(lr) == summ["finetune_lr"] else ""))
        ax.axhline(summ["zero_shot_val"], color=T["ref"], ls=":", lw=1.2); ax.text(stp[-1], summ["zero_shot_val"] - 0.4, "no fine-tuning", ha="right", va="top", fontsize=8, color=T["ink2"])
        ax.set_ylim(min(np.min([[x[2] for x in h] for h in hs]) for hs in summ["finetune_curves"].values()) - 0.5, summ["zero_shot_val"] + 1)
    else:
        for key, col, lab in [("scratch_hist", T["ref"], "from scratch"), ("finetune_hist", T["s"][0], "T1D-pretrained")]:
            h = np.array(tfs[key]); ax.plot(h[:, 0], h[:, 2], "o-", color=col, ms=4, label=lab)
    ax.set_xlabel("fine-tuning step on CGMacros (batch 64)"); ax.set_ylabel("curve MAE, days 6–7 (mg/dL)"); ax.set_title("Fine-tuning: validation (mean of 3 seeds)")
    ax.legend(fontsize=8.5)
    ax = axes[2]
    q = tfm[(tfm.subset == "unseen") & tfm.model.isin(["transformer_scratch", "transformer_pretrained"])]
    tg = ["G60", "dpeak", "iauc"]
    for k, (m, col, lab) in enumerate([("transformer_scratch", T["ref"], "from scratch"), ("transformer_pretrained", T["s"][0], "T1D-pretrained")]):
        base = q[q.model == "transformer_scratch"].set_index("target").mae
        v = [q[(q.model == m) & (q.target == t)].mae.iloc[0] / base[t] * 100 for t in tg]
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, v, width=0.34, color=col, label=lab)
    ax.axhline(100, color=T["ink2"], lw=0.8); ax.set_ylim(85, 105)
    ax.set_xticks(range(3), ["glucose 60 min", "peak rise", "iAUC"]); ax.set_ylabel("test MAE, % of from-scratch")
    ax.set_title("Does pretraining help? (day 8+)"); ax.legend(fontsize=8.5); ax.grid(axis="x", visible=False)
    return fig
save_fig(f_transformer, "r_transformer", F)

def f_meds(T):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    order = ["population, no insulin", "population + insulin", "personalized, no insulin", "personalized + insulin"]
    cols = [T["ref"], T["s"][1], T["s"][2], T["s"][0]]
    for ax, t, lab in [(axes[0], "dpeak", "Peak rise (mg/dL)"), (axes[1], "iauc", "iAUC 0–120 min (mg/dL·min)")]:
        q = med[(med.subset == "all") & (med.target == t)].set_index("model").reindex(order)
        ax.hlines(range(4), q.mae.min() * 0.985, q.mae, color=T["grid"], lw=1.2)
        ax.scatter(q.mae, range(4), c=cols, s=80, edgecolor=T["panel"], lw=1.5, zorder=3)
        for yi, v in enumerate(q.mae): ax.text(v + (q.mae.max() - q.mae.min()) * 0.06, yi, f"{v:,.1f}" if t == "dpeak" else f"{v:,.0f}", va="center", fontsize=9, color=T["ink"])
        ax.set_xlim(q.mae.min() * 0.985, q.mae.max() * 1.01); ax.set_title(lab + " · dot plot, axis not at zero"); ax.set_xlabel("MAE on days 8–10")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(4), order); axes[0].invert_yaxis()
    return fig
save_fig(f_meds, "r_meds", F)

def f_examples(T):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, gridspec_kw=dict(hspace=0.42, wspace=0.22))
    tei = fp["te_index"]
    cand = [k for k, i in enumerate(tei) if not TE.loc[i, "seen"]]
    rng = np.random.default_rng(4)
    picks = []
    for g in GROUPS:
        ks = [k for k in cand if TE.loc[tei[k], "group"] == g]
        picks += list(rng.choice(ks, 3 if g != "Healthy" else 2, replace=False))
    tt = np.arange(0, 121, 5); pt = np.arange(-120, 1, 5)
    for ax, k in zip(axes.flat, picks):
        i = tei[k]; r = TE.loc[i]; g0 = r.G0
        ax.plot(pt, g0 + PRE[i], color=T["ref"], lw=1.4)
        ax.fill_between(tt, g0 + fp["band_lo"][k], g0 + fp["band_hi"][k], color=T["s"][0], alpha=T["band_alpha"], lw=0)
        ax.plot(tt, g0 + fp["curve"][k], color=T["s"][0], lw=2.2, label="predicted (80% band)")
        ax.plot(tt, g0 + CURV[i], color=T["ink"], lw=1.6, ls="--", label="observed")
        ax.axvline(0, color=T["ref"], ls=":", lw=1)
        ax.set_title(f"#{r.pid} · {r.group} · {r.meal_type}\nest. {r.est_carbs:.0f} g carbs (logged {r.carbs:g})", fontsize=9.5)
    for ax in axes[1]: ax.set_xlabel("minutes from meal")
    for ax in axes[:, 0]: ax.set_ylabel("glucose (mg/dL)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    return fig
save_fig(f_examples, "r_examples", F)

def f_scatter(T):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(wspace=0.28))
    tei = fp["te_index"]; obs = TE.loc[tei, "dpeak"].to_numpy(); grp = TE.loc[tei, "group"].to_numpy()
    ax = axes[0]
    for g, c in zip(GROUPS, T["s"][:3]):
        m = grp == g
        ax.scatter(obs[m], fp["blend"]["dpeak"][m], s=12, color=c, alpha=0.7, lw=0, label=g)
    mx = np.percentile(obs, 99); ax.plot([0, mx], [0, mx], color=T["ink2"], ls="--", lw=1)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_xlabel("observed peak rise (mg/dL)"); ax.set_ylabel("predicted peak rise (mg/dL)")
    r = np.corrcoef(obs, fp["blend"]["dpeak"])[0, 1]
    ax.set_title(f"Every day-8+ meal (r = {r:.2f})"); ax.legend(fontsize=8.5)
    ax = axes[1]
    df = pd.DataFrame(dict(pid=TE.loc[tei, "pid"].to_numpy(), o=obs, p=fp["blend"]["dpeak"], g=grp)).groupby("pid").agg(o=("o", "mean"), p=("p", "mean"), g=("g", "first"))
    for g, c in zip(GROUPS, T["s"][:3]):
        q = df[df.g == g]; ax.scatter(q.o, q.p, s=40, color=c, edgecolor=T["panel"], lw=1.2, label=g)
    mx = df[["o", "p"]].max().max() * 1.1; ax.plot([0, mx], [0, mx], color=T["ink2"], ls="--", lw=1)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_xlabel("person's mean observed rise"); ax.set_ylabel("person's mean predicted rise")
    ax.set_title(f"Per person (r = {np.corrcoef(df.o, df.p)[0, 1]:.2f})")
    return fig
save_fig(f_scatter, "r_scatter", F)
print("site figures written")
