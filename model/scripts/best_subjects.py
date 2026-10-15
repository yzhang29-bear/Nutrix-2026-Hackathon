"""Per-person accuracy of the final model on day-8+ test meals (predicted vs logged CGM), ranked to pick showcase subjects.
usage: python -I best_subjects.py <results dir> <out dir>
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
st = pickle.load(open(os.path.join(R, "final/stage1.pkl"), "rb"))
fp = pickle.load(open(os.path.join(R, "final_blend/final_preds.pkl"), "rb"))
TE = st["TE"]
obs = st["outer"]["CURV"][fp["te_index"]]          # glucose change from meal start, 0..120 min every 5 min
pred, lo, hi = fp["curve"], fp["band_lo"], fp["band_hi"]
bl = fp["blend"]
assert (TE.index.to_numpy() == fp["te_index"]).all()

rows = []
for pid, g in TE.groupby("pid"):
    i = TE.index.get_indexer(g.index)
    o, p = obs[i], pred[i]
    ok = np.isfinite(o)
    rows.append(dict(
        pid=pid, group=g.group.iloc[0], n_meals=len(g), n_unseen=int((~g.seen).sum()),
        curve_mae=np.abs(o - p)[ok].mean(),
        dpeak_mae=np.abs(np.asarray(bl["dpeak"])[i] - g.dpeak.to_numpy()).mean(),
        iauc_mae=np.abs(np.asarray(bl["iauc"])[i] - g.iauc.to_numpy()).mean(),
        band_cov=((o >= lo[i]) & (o <= hi[i]))[ok].mean(),
        curve_r=stats.pearsonr(o[ok], p[ok])[0],
        dpeak_r=stats.pearsonr(np.asarray(bl["dpeak"])[i], g.dpeak)[0] if len(g) >= 4 else np.nan,
        obs_dpeak_mean=g.dpeak.mean(), obs_dpeak_max=g.dpeak.max(),
    ))
P = pd.DataFrame(rows)
# composite: average percentile rank of the two headline errors (lower = better)
P["score"] = (P.curve_mae.rank(pct=True) + P.dpeak_mae.rank(pct=True)) / 2
P = P.sort_values("score").reset_index(drop=True)
P.to_csv(os.path.join(OUT, "per_person_accuracy.csv"), index=False, float_format="%.3f")
pd.set_option("display.width", 200)
print(P.round(2).to_string())
print("\ncohort medians:", P[["curve_mae", "dpeak_mae", "iauc_mae", "band_cov", "curve_r"]].median().round(2).to_dict())

# ------------------------------------------------------------------ showcase pick: best composite among people whose curve shape is
# also tracked (r >= .7) and who have >= 6 test meals; top 3 overall + the best T2D person so all three groups appear
SHOW_R, SHOW_N = 0.7, 6
elig = P[(P.curve_r >= SHOW_R) & (P.n_meals >= SHOW_N)]
pick = list(elig.pid[:3])
pick += [elig[elig.group == "T2D"].pid.iloc[0]] if not elig.head(3).group.eq("T2D").any() else []
print("showcase:", pick)
P.assign(showcase=P.pid.isin(pick)).to_csv(os.path.join(OUT, "per_person_accuracy.csv"), index=False, float_format="%.3f")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_common import save_fig, GROUPS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = st["outer"]["PRE"][fp["te_index"]]
med = P[["curve_mae", "dpeak_mae"]].median()


def f_overview(T):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for g, c in zip(GROUPS, T["s"][:3]):
        q = P[P.group == g]
        ax.scatter(q.curve_mae, q.dpeak_mae, s=46, color=c, edgecolor=T["panel"], lw=1.4, label=g, zorder=2)
    q = P[P.pid.isin(pick)]
    ax.scatter(q.curve_mae, q.dpeak_mae, s=150, facecolor="none", edgecolor=T["ink"], lw=1.6, zorder=3)
    for _, r in q.iterrows():
        ax.annotate(f"#{r.pid}", (r.curve_mae, r.dpeak_mae), xytext=(9, -3), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.axvline(med.curve_mae, color=T["ref"], ls=":", lw=1); ax.axhline(med.dpeak_mae, color=T["ref"], ls=":", lw=1)
    ax.text(med.curve_mae, ax.get_ylim()[1], " cohort median", fontsize=8.5, color=T["ink2"], va="top")
    ax.set_xlabel("curve error: mean |predicted − logged| over 0–120 min (mg/dL)")
    ax.set_ylabel("peak-rise error, mean |predicted − logged| (mg/dL)")
    ax.set_title(f"Per-person accuracy on unseen days (day 8+, {len(P)} people)\ncircled = showcase subjects (lower-left is better)")
    ax.legend(loc="lower right")
    return fig


def f_person(T, pid):
    g = TE[TE.pid == pid].sort_values("time")
    pr = P.set_index("pid").loc[pid]
    n = len(g); nr = 1 if n <= 5 else 2; nc = int(np.ceil(n / nr))
    # calendar study day (TE.day counts 24 h windows from the first sample, which splits dates)
    d0 = g.time.iloc[0].normalize(); cal_day = g.day.iloc[0] + (g.time.dt.normalize() - d0).dt.days
    fig = plt.figure(figsize=(3.3 * nc, 4.9 * nr + 0.5))
    outer = fig.add_gridspec(nr, 1, hspace=0.2, top=0.9, bottom=0.06, left=0.05, right=0.99)
    gs = [outer[i].subgridspec(2, nc, height_ratios=[1.0, 1.35], hspace=0.12, wspace=0.26) for i in range(nr)]
    tt = np.arange(0, 121, 5); pt = np.arange(-120, 1, 5)
    ymax = max(np.nanmax(g.G0.to_numpy()[:, None] + obs[TE.index.get_indexer(g.index)]),
               np.nanmax(g.G0.to_numpy()[:, None] + hi[TE.index.get_indexer(g.index)])) + 24
    ymin = min(np.nanmin(g.G0.to_numpy()[:, None] + PRE[TE.index.get_indexer(g.index)]),
               np.nanmin(g.G0.to_numpy()[:, None] + lo[TE.index.get_indexer(g.index)])) - 8
    for j, (ix, r) in enumerate(g.iterrows()):
        k = TE.index.get_loc(ix); row, col = divmod(j, nc)
        axp = fig.add_subplot(gs[row][0, col])
        im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, r.photo))).convert("RGB")
        axp.imshow(ImageOps.fit(im, (400, 300))); axp.axis("off")
        axp.set_title(f"{r.meal_type} · day {cal_day[ix]} {r.time:%H:%M}" + ("  (repeat recipe)" if r.seen else ""), fontsize=9.5)
        ax = fig.add_subplot(gs[row][1, col])
        ax.plot(pt, r.G0 + PRE[k], color=T["ref"], lw=1.3)
        ax.fill_between(tt, r.G0 + lo[k], r.G0 + hi[k], color=T["s"][0], alpha=T["band_alpha"], lw=0)
        ax.plot(tt, r.G0 + pred[k], color=T["s"][0], lw=2.2, label="predicted (80% band)")
        ax.plot(tt, r.G0 + obs[k], color=T["ink"], lw=1.6, ls="--", label="logged CGM")
        ax.axvline(0, color=T["ref"], ls=":", lw=1)
        ax.set_ylim(ymin, ymax); ax.set_xticks([-120, -60, 0, 60, 120])
        ax.text(0.03, 0.97, f"peak rise  pred {bl['dpeak'][k]:.0f} / logged {r.dpeak:.0f}\ncarbs  photo est. {r.est_carbs:.0f} g / logged {r.carbs:g} g",
                transform=ax.transAxes, va="top", fontsize=8.3, color=T["ink2"])
        if col == 0: ax.set_ylabel("glucose (mg/dL)")
        if row == nr - 1: ax.set_xlabel("minutes from meal")
    fig.text(0.01, 0.995, f"Subject #{pid} · {pr.group} · {n} meals on unseen days (day 8+)", ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.965, f"curve error {pr.curve_mae:.1f} mg/dL (cohort median {med.curve_mae:.1f})  ·  peak-rise error {pr.dpeak_mae:.1f} mg/dL "
             f"(median {med.dpeak_mae:.1f})  ·  {pr.band_cov:.0%} of logged points inside the 80% band  ·  grey = 2 h before the meal",
             ha="left", va="top", fontsize=10.5, color=T["ink2"])
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.995), fontsize=9.5)
    return fig


save_fig(f_overview, "overview", OUT)
for pid in pick:
    save_fig(f_person, f"subject_{pid}", OUT, pid=pid)
print("figures written to", OUT)

# ------------------------------------------------------------------ showcase sheet: per showcase subject, only the K meals whose predicted
# curve is closest to the logged CGM (smallest mean |predicted − logged| over 0–120 min)
K = 3
meal_mae = np.nanmean(np.abs(obs - pred), 1)
best = {pid: TE.index[TE.pid.to_numpy() == pid][np.argsort(meal_mae[TE.pid.to_numpy() == pid])[:K]] for pid in pick}
pd.DataFrame([dict(pid=pid, meal_index=ix, time=TE.loc[ix, "time"], meal_type=TE.loc[ix, "meal_type"], curve_mae=meal_mae[TE.index.get_loc(ix)],
                   pred_dpeak=bl["dpeak"][TE.index.get_loc(ix)], logged_dpeak=TE.loc[ix, "dpeak"]) for pid in pick for ix in best[pid]]
             ).to_csv(os.path.join(OUT, "showcase_meals.csv"), index=False, float_format="%.2f")


def f_showcase(T):
    nr = len(pick)
    H = 3.25 * nr + 0.8
    fig = plt.figure(figsize=(6.1 * K, H))
    outer = fig.add_gridspec(2 * nr, 1, height_ratios=[0.2, 1.0] * nr, hspace=0.32, top=1 - 0.75 / H, bottom=0.045, left=0.012, right=0.995)
    tt = np.arange(0, 121, 5); pt = np.arange(-120, 1, 5)
    for row, pid in enumerate(pick):
        g = TE.loc[best[pid]]; kk = TE.index.get_indexer(g.index); pr = P.set_index("pid").loc[pid]
        d0 = TE[TE.pid == pid].time.min().normalize(); dfirst = TE[TE.pid == pid].day.min()
        axh = fig.add_subplot(outer[2 * row]); axh.axis("off")
        axh.text(0, 0.1, f"Subject #{pid} · {pr.group}", fontsize=12, fontweight="bold", va="bottom", transform=axh.transAxes)
        axh.text(0.17, 0.1, f"all {pr.n_meals:.0f} unseen-day meals: curve error {pr.curve_mae:.1f} mg/dL, peak-rise error {pr.dpeak_mae:.1f} mg/dL",
                 fontsize=9.5, color=T["ink2"], va="bottom", transform=axh.transAxes)
        gs = outer[2 * row + 1].subgridspec(1, 2 * K, width_ratios=[1.0, 1.3] * K, wspace=0.14)
        ymax = np.nanmax(g.G0.to_numpy()[:, None] + np.maximum(obs[kk], hi[kk])) + 26
        ymin = np.nanmin(g.G0.to_numpy()[:, None] + np.minimum(np.minimum(obs[kk], lo[kk]), PRE[kk])) - 6
        for j, (ix, r) in enumerate(g.iterrows()):
            k = TE.index.get_loc(ix)
            axp = fig.add_subplot(gs[0, 2 * j])
            im = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, r.photo))).convert("RGB")
            axp.imshow(ImageOps.fit(im, (360, 360))); axp.axis("off")
            day = dfirst + (r.time.normalize() - d0).days
            axp.set_title(f"{r.meal_type} · day {day} {r.time:%H:%M}", fontsize=9.5)
            ax = fig.add_subplot(gs[0, 2 * j + 1])
            ax.plot(pt, r.G0 + PRE[k], color=T["ref"], lw=1.3)
            ax.fill_between(tt, r.G0 + lo[k], r.G0 + hi[k], color=T["s"][0], alpha=T["band_alpha"], lw=0)
            ax.plot(tt, r.G0 + pred[k], color=T["s"][0], lw=2.2, label="predicted (80% band)")
            ax.plot(tt, r.G0 + obs[k], color=T["ink"], lw=1.6, ls="--", label="logged CGM")
            ax.axvline(0, color=T["ref"], ls=":", lw=1)
            ax.set_ylim(ymin, ymax); ax.set_xticks([-120, -60, 0, 60, 120]); ax.tick_params(labelsize=8.5)
            ax.text(0.03, 0.97, f"curve error {meal_mae[k]:.1f} mg/dL\npeak rise  pred {bl['dpeak'][k]:.0f} / logged {r.dpeak:.0f}",
                    transform=ax.transAxes, va="top", fontsize=8.3, color=T["ink2"])
            if row == nr - 1: ax.set_xlabel("minutes from meal", fontsize=9)
    fig.text(0.012, 1 - 0.12 / H, "Most accurate day-8+ meals of the four showcase subjects", fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 1 - 0.42 / H, f"glucose in mg/dL · grey = 2 h before the meal · median over all 402 test meals: curve error {np.nanmedian(meal_mae):.1f} mg/dL",
             fontsize=10, color=T["ink2"], va="top")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1 - 0.08 / H), fontsize=10)
    return fig


save_fig(f_showcase, "showcase_best_meals", OUT)
print("showcase sheet written")
