"""T1D harmonized parquet EDA, figure + summary pass (reads the compute-pass outputs).
usage: python -I eda_t1d_figs.py <eda dir>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_common import save_fig
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

E = sys.argv[1]
FIG = os.path.join(E, "fig")
S = pd.read_csv(os.path.join(E, "t1d_subjects.csv"), dtype={"id": str})
A = np.load(os.path.join(E, "t1d_arrays.npz"), allow_pickle=True)
J = json.load(open(os.path.join(E, "t1d_stats.json")))
PM = pd.read_csv(os.path.join(E, "t1d_pp_meta.csv"))
srcs = list(A["srcs"])
OUT = {}

MODS = ["AID", "SAP", "MDI"]
MODC = lambda T: dict(zip(MODS, T["s"][:3]))
AGE_BINS = [0, 6, 13, 18, 26, 50, 120]
AGE_LAB = ["<6", "6–12", "13–17", "18–25", "26–49", "50+"]
S["age_grp"] = pd.cut(S.age, AGE_BINS, right=False, labels=AGE_LAB)
S["years"] = S.days / 365.25
S["weight_kg"] = S.weight / 2.20462
S["tdd_kg"] = S.tdd / S.weight_kg
S["basal_share"] = S.basal_day / (S.basal_day + S.bolus_day)
order = S.groupby("source").size().sort_values(ascending=True).index.tolist()     # barh bottom→top
mod_of = S.groupby("source").insulin_delivery_modality.agg(lambda x: x.value_counts().index[0] if x.notna().any() else "SAP")

def ramp(T, n):
    return LinearSegmentedColormap.from_list("r", T["seq"][1:] if n <= 4 else T["seq"])(np.linspace(0, 1, n))

# ------------------------------------------------------------------ summary numbers for the report
OUT["n_subjects"] = len(S)
OUT["n_id_strings"] = int(S.id.nunique())
OUT["ids_in_multiple_sources"] = int((S.groupby("id").source.nunique() > 1).sum())
OUT["subject_years"] = float(S.years.sum())
OUT["cgm_years"] = float(S.cgm_days.sum() / 365.25)
OUT["loop_share_subjects"] = float((S.source == "Loop").mean())
OUT["loop_share_years"] = float(S.loc[S.source == "Loop", "years"].sum() / S.years.sum())
OUT["modality_n"] = S.insulin_delivery_modality.value_counts(dropna=False).to_dict()
OUT["age_grp_n"] = S.age_grp.value_counts().reindex(AGE_LAB).to_dict()
OUT["pediatric_share"] = float((S.age < 18).mean())
OUT["gender"] = S.gender.value_counts(dropna=False).to_dict()
OUT["split_flag_subjects"] = int(S.split_flag.sum())
OUT["pregnant_subjects"] = int(S.is_pregnant.sum())
OUT["future_end_subjects"] = S.loc[S.end > "2026-10-10"].groupby("source").size().to_dict()
OUT["weight_timevarying"] = int((S.weight_nvals > 1).sum())
OUT["modality_changes"] = int((S.insulin_delivery_modality_nvals > 1).sum())
OUT["bolus_negative"] = dict(subjects=int((S.bolus_neg > 0).sum()), values=int(S.bolus_neg.sum()))
OUT["nonint_cgm"] = {k: round(v, 3) for k, v in J["nonint"].items()}
gm = A["gap_minutes"]
OUT["gap_share_by_len"] = dict(zip(J["gap_labels"], (gm.sum(0) / gm.sum()).round(3).tolist()))
h = A["cgm_hist"]; b = A["gbins"]
OUT["cgm_lt40_by_src"] = dict(zip(srcs, (h[:, b[:-1] < 40].sum(1) / h.sum(1) * 100).round(3).tolist()))
OUT["cgm_gt400_by_src"] = dict(zip(srcs, (h[:, b[:-1] >= 400].sum(1) / h.sum(1) * 100).round(3).tolist()))
OUT["overall"] = dict(tir=float(S.tir.median()), tbr70=float(S.tbr70.median()), tbr54=float(S.tbr54.median()), tar180=float(S.tar180.median()),
                      cv=float(S.cv.median()), mean_g=float(S.mean_g.median()),
                      meet_tir70=float((S.tir >= 70).mean()), meet_tbr4=float((S.tbr70 < 4).mean()), meet_both=float(((S.tir >= 70) & (S.tbr70 < 4)).mean()))
OUT["by_modality"] = S.groupby("insulin_delivery_modality")[["tir", "tbr70", "tar180", "cv", "mean_g"]].median().round(1).to_dict("index")
OUT["by_agegrp"] = S.groupby("age_grp", observed=True)[["tir", "tbr70", "tar180", "cv", "tdd_kg"]].median().round(2).to_dict("index")
src_tab = S.groupby("source").agg(n=("key", "size"), years=("years", "sum"), days_med=("days", "median"), cgm_cov=("cgm_cov", "median"),
                                   age_med=("age", "median"), age_min=("age", "min"), age_max=("age", "max"), tir=("tir", "median"),
                                   tbr=("tbr70", "median"), cv=("cv", "median"), carb_entries=("carb_entries_day", "median"),
                                   hr=("hr_cov", "median"), split=("split_flag", "sum"))
src_tab["modality"] = mod_of
src_tab["device"] = S.groupby("source").cgm_device.agg(lambda x: x.value_counts().index[0] if x.notna().any() else "—")
src_tab["start"] = S.groupby("source").start.min(); src_tab["end"] = S.groupby("source").end.max()
OUT["source_table"] = src_tab.sort_values("n", ascending=False).round(2).reset_index().to_dict("records")
cs = A["carb_sizes"]
OUT["carb_entries_total"] = int(len(cs))
OUT["carb_mult5"] = float(np.mean(np.isclose(cs % 5, 0))); OUT["carb_mult10"] = float(np.mean(np.isclose(cs % 10, 0)))
OUT["carb_median"] = float(np.median(cs))
OUT["pp_events"] = int(len(PM)); OUT["pp_by_src"] = PM.source.value_counts().to_dict()

# ------------------------------------------------------------------ T1 cohort composition
def f_comp(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=True, gridspec_kw=dict(wspace=0.08, width_ratios=[1, 1, 1.1]))
    y = np.arange(len(order)); MC = MODC(T)
    n = S.groupby("source").size().reindex(order); yr = S.groupby("source").years.sum().reindex(order)
    cols = [MC.get(mod_of[s], T["ref"]) for s in order]
    for ax, v, lab in [(axes[0], n, "participants"), (axes[1], yr, "person-years of records")]:
        ax.barh(y, v, color=cols, height=0.72)
        for yi, vi in zip(y, v):
            ax.text(vi + v.max() * 0.015, yi, f"{vi:,.0f}" if vi >= 10 else f"{vi:.1f}", va="center", fontsize=8.5, color=T["ink"])
        ax.set_xlabel(lab); ax.grid(axis="y", visible=False); ax.set_xlim(0, v.max() * 1.16)
    axes[0].set_yticks(y, order); axes[0].set_title(f"{len(S):,} participants from {len(order)} studies")
    axes[1].set_title(f"{S.years.sum():,.0f} person-years ({OUT['loop_share_years'] * 100:.0f}% from Loop)")
    axes[0].legend(handles=[Patch(color=MC[m], label=m) for m in MODS], title="main insulin delivery", fontsize=8.5, title_fontsize=8.5, loc="lower right")
    ax = axes[2]
    for yi, s in zip(y, order):
        a = S.loc[S.source == s, "age"].dropna()
        if a.empty:
            ax.text(1, yi, "age not provided", va="center", fontsize=8.5, color=T["ink2"]); continue
        ax.plot([a.min(), a.max()], [yi, yi], color=T["ref"], lw=1.2)
        ax.plot([a.quantile(.25), a.quantile(.75)], [yi, yi], color=T["s"][0], lw=6, solid_capstyle="butt")
        ax.plot(a.median(), yi, "o", color=T["panel"], ms=4.5, mec=T["ink"], mew=1)
    ax.axvline(18, color=T["ink2"], ls=":", lw=1); ax.text(18.8, len(order) - 0.6, "18 y", fontsize=8.5, color=T["ink2"])
    ax.set_xlabel("age (years): range, IQR, median"); ax.set_title(f"Age spans toddlers to 80s ({OUT['pediatric_share'] * 100:.0f}% under 18)")
    ax.grid(axis="y", visible=False)
    return fig
save_fig(f_comp, "t1_composition", FIG)

# ------------------------------------------------------------------ T2 variable availability by source
VARS = [("CGM", "CGM"), ("basal", "basal"), ("bolus", "bolus"), ("insulin", "insulin"), ("carbs", "carbs"), ("heartrate", "heart rate"),
        ("steps", "steps"), ("age", "age"), ("gender", "gender"), ("ethnicity", "ethnicity"), ("weight", "weight"), ("height", "height"),
        ("age_of_diagnosis", "age at dx"), ("is_pregnant", "pregnancy"), ("treatment_group", "trial arm"), ("cgm_device", "CGM device"),
        ("insulin_delivery_modality", "delivery mode")]
def f_avail(T):
    order_t = order[::-1]
    M_ = np.array([[J["avail"][s].get(c, np.nan) for c, _ in VARS] for s in order_t]) * 100
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    cmap = LinearSegmentedColormap.from_list("a", [T["panel"], T["seq"][1], T["seq"][3]])
    ax.imshow(M_, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M_.shape[0]):
        for j in range(M_.shape[1]):
            v = M_[i, j]
            ax.text(j, i, "·" if v < 0.5 else f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color=T["ink2"] if v < 0.5 else (T["panel"] if v > 70 else T["ink"]))
    ax.set_xticks(range(len(VARS)), [l for _, l in VARS], rotation=35, ha="right"); ax.set_yticks(range(len(order_t)), order_t)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("Share of 5-min rows with a non-missing value (%), by study  ·  “·” = never recorded")
    return fig
save_fig(f_avail, "t1_availability", FIG)

# ------------------------------------------------------------------ T3 CGM value quality
def f_cgmvals(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1.35, 1, 1]))
    ax = axes[0]
    tot = h.sum(0)
    ax.bar(b[:-1] + 2.5, tot / tot.sum() * 100, width=5, color=T["s"][0], lw=0)
    ax.set_yscale("log")
    for xv, lab in [(40, "40"), (70, "70"), (180, "180"), (400, "400")]:
        ax.axvline(xv, color=T["ink2"], ls=":", lw=1); ax.text(xv + 3, 3.2, lab, fontsize=8.5, color=T["ink2"])
    ax.set_xlabel("CGM glucose (mg/dL), 5 mg/dL bins"); ax.set_ylabel("% of readings (log)")
    ax.set_title(f"{tot.sum() / 1e6:.0f} M readings pooled")
    ax = axes[1]
    ni = pd.Series(J["nonint"]).reindex(order)
    ax.barh(range(len(order)), ni * 100, color=[T["s"][1] if v > 0.2 else T["s"][0] for v in ni], height=0.7)
    ax.set_yticks(range(len(order)), order, fontsize=8.6); ax.set_xlabel("% of readings that are not whole mg/dL")
    ax.set_title("Converted from mmol/L?"); ax.grid(axis="y", visible=False); ax.set_xlim(0, 105)
    ax = axes[2]
    lo = pd.Series(OUT["cgm_lt40_by_src"]).reindex(order); hi = pd.Series(OUT["cgm_gt400_by_src"]).reindex(order)
    ax.scatter(lo, range(len(order)), s=36, color=T["s"][7], label="< 40 mg/dL", zorder=3, edgecolor=T["panel"])
    ax.scatter(hi, range(len(order)), s=36, color=T["s"][3], label="≥ 400 mg/dL", zorder=3, marker="D", edgecolor=T["panel"])
    ax.set_yticks(range(len(order)), [""] * len(order)); ax.set_xlabel("% of readings below 40 or at/above 400 mg/dL")
    ax.set_title("Readings at the sensor limits"); ax.legend(fontsize=8.5, loc="lower right"); ax.grid(axis="y", visible=False)
    return fig
save_fig(f_cgmvals, "t1_cgm_values", FIG)

# ------------------------------------------------------------------ T4 CGM missingness
def f_missing(T):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True, gridspec_kw=dict(wspace=0.06, width_ratios=[1, 1.15]))
    ax = axes[0]
    for yi, s in enumerate(order):
        c = S.loc[S.source == s, "cgm_cov"] * 100
        ax.boxplot(c, positions=[yi], vert=False, widths=0.6, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=T["panel"], edgecolor=T["s"][0]), medianprops=dict(color=T["s"][0], lw=2),
                   whiskerprops=dict(color=T["ink2"]), capprops=dict(color=T["ink2"]))
        ax.scatter(c, yi + np.random.default_rng(yi).uniform(-0.22, 0.22, len(c)), s=4, color=T["s"][0], alpha=0.35, lw=0)
    ax.set_yticks(range(len(order)), order); ax.set_xlabel("CGM coverage per participant (% of 5-min slots)")
    ax.set_title("How complete is each participant's CGM trace?"); ax.grid(axis="y", visible=False)
    ax = axes[1]
    gm_ = pd.DataFrame(gm, index=srcs, columns=J["gap_labels"]).reindex(order)
    frac = gm_.div(gm_.sum(1).replace(0, np.nan), axis=0).fillna(0) * 100
    left = np.zeros(len(order)); cols = ramp(T, len(J["gap_labels"]))
    for k, lab in enumerate(J["gap_labels"]):
        ax.barh(range(len(order)), frac[lab], left=left, color=cols[k], height=0.72, label=lab); left += frac[lab].to_numpy()
    ax.set_xlabel("share of missing CGM time, by length of the gap (%)"); ax.set_xlim(0, 100)
    ax.set_title(f"Missing time is mostly long outages ({OUT['gap_share_by_len']['>24 h'] * 100:.0f}% in gaps > 24 h)")
    ax.legend(ncol=7, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.06), borderaxespad=0, handlelength=1.2, columnspacing=0.9)
    ax.grid(axis="y", visible=False)
    return fig
save_fig(f_missing, "t1_missingness", FIG)

# ------------------------------------------------------------------ T5 glycaemic outcomes
def f_outcomes(T):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw=dict(wspace=0.25, width_ratios=[1.15, 1]))
    ax = axes[0]; MC = MODC(T)
    o2 = S.groupby("source").tir.median().sort_values().index.tolist()
    for yi, s in enumerate(o2):
        q = S[(S.source == s) & S.tir.notna()]
        ax.boxplot(q.tir, positions=[yi], vert=False, widths=0.62, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=T["panel"], edgecolor=T["ink2"]), medianprops=dict(color=T["ink"], lw=2),
                   whiskerprops=dict(color=T["ink2"]), capprops=dict(color=T["ink2"]))
        ax.scatter(q.tir, yi + np.random.default_rng(yi).uniform(-0.24, 0.24, len(q)), s=5, lw=0, alpha=0.55,
                   color=[MC.get(m, T["ref"]) for m in q.insulin_delivery_modality])
    ax.axvline(70, color=T["s"][2], ls="--", lw=1.2); ax.text(70.5, -0.9, "consensus target ≥ 70%", fontsize=8.5, color=T["ink2"])
    ax.set_yticks(range(len(o2)), o2); ax.set_xlabel("time in range 70–180 mg/dL (%)"); ax.grid(axis="y", visible=False)
    ax.set_title("Time in range by study (dots = participants)"); ax.set_ylim(-1.3, len(o2) - 0.4)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MC[m], label=m) for m in MODS], fontsize=8.5, loc="lower left")
    ax = axes[1]
    for m in MODS[::-1]:
        q = S[S.insulin_delivery_modality == m]
        ax.scatter(q.tar180, q.tbr70, s=9 if m == "AID" else 16, color=MC[m], alpha=0.5 if m == "AID" else 0.8, lw=0,
                   label=f"{m} (n={len(q)}): median TIR {q.tir.median():.0f}%")
    ax.axhline(4, color=T["ink2"], ls=":", lw=1); ax.axvline(25, color=T["ink2"], ls=":", lw=1)
    ax.text(26, 19, "TAR target < 25%", fontsize=8.5, color=T["ink2"]); ax.text(78, 4.4, "TBR target < 4%", fontsize=8.5, color=T["ink2"], ha="right")
    ax.set_xlim(0, 80); ax.set_ylim(0, 20); ax.set_xlabel("time above 180 mg/dL (%)"); ax.set_ylabel("time below 70 mg/dL (%)")
    ax.set_title(f"Hypo vs hyper exposure ({OUT['overall']['meet_both'] * 100:.0f}% meet both TIR and TBR targets)")
    ax.legend(fontsize=8.5, loc="upper right")
    return fig
save_fig(f_outcomes, "t1_outcomes", FIG)

# ------------------------------------------------------------------ T6 diurnal profiles
HR_ = A["hourly"]
def f_diurnal(T):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    ax = axes[0]; cols = ramp(T, len(AGE_LAB))
    for k, g in enumerate(AGE_LAB):
        idx = S.loc[S.age_grp == g, "hourly_idx"].to_numpy()
        ax.plot(np.arange(24) + 0.5, np.nanmedian(HR_[idx], 0), color=cols[k], lw=2.2, label=f"{g} y (n={len(idx)})")
    ax.set_title("Median hourly glucose by age group"); ax.set_ylabel("CGM glucose (mg/dL)")
    ax.legend(fontsize=8.3, ncol=2, loc="upper right")
    ax = axes[1]; MC = MODC(T)
    for m in MODS:
        idx = S.loc[S.insulin_delivery_modality == m, "hourly_idx"].to_numpy()
        ax.fill_between(np.arange(24) + 0.5, np.nanpercentile(HR_[idx], 25, 0), np.nanpercentile(HR_[idx], 75, 0), color=MC[m], alpha=T["band_alpha"], lw=0)
        ax.plot(np.arange(24) + 0.5, np.nanmedian(HR_[idx], 0), color=MC[m], lw=2.2, label=f"{m} (n={len(idx)})")
    ax.set_title("By insulin delivery mode (median, IQR of participants)"); ax.legend(fontsize=8.5, loc="upper right")
    for ax in axes:
        ax.set_xticks([0, 6, 12, 18, 24], ["00", "06", "12", "18", "24"]); ax.set_xlabel("hour of day (local clock as recorded)")
        ax.axhspan(70, 180, color=T["s"][2], alpha=0.06, lw=0)
    return fig
save_fig(f_diurnal, "t1_diurnal", FIG)

# ------------------------------------------------------------------ T7 insulin & carbs
def f_ins_carbs(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    q = S[(S.ins_cov > 0.8) & S.tdd_kg.between(0.05, 3)]
    cols = ramp(T, len(AGE_LAB))
    for k, g in enumerate(AGE_LAB):
        x = q.loc[q.age_grp == g, "tdd_kg"]
        ax.boxplot(x, positions=[k], widths=0.6, showfliers=False, patch_artist=True, boxprops=dict(facecolor=cols[k], edgecolor=cols[k]),
                   medianprops=dict(color=T["panel"], lw=2), whiskerprops=dict(color=T["ink2"]), capprops=dict(color=T["ink2"]))
    ax.set_xticks(range(len(AGE_LAB)), AGE_LAB); ax.set_xlabel("age group (y)"); ax.set_ylabel("total daily insulin (U/kg/day)")
    ax.set_title(f"Insulin need peaks in adolescence (n={len(q)})"); ax.grid(axis="x", visible=False)
    ax = axes[1]
    ch = pd.DataFrame(A["carb_hour"], index=srcs)
    ch = ch[ch.sum(1) > 0]
    pooled = ch.sum(0) / ch.sum().sum() * 100
    ax.bar(np.arange(24) + 0.5, pooled, width=0.85, color=T["s"][0], lw=0)
    ax.set_xticks([0, 6, 12, 18, 24], ["00", "06", "12", "18", "24"]); ax.set_xlabel("hour of day"); ax.set_ylabel("% of carb entries")
    ax.set_title(f"When carbs are logged ({len(ch)} studies, {OUT['carb_entries_total'] / 1e6:.2f} M entries)"); ax.grid(axis="x", visible=False)
    ax = axes[2]
    x = np.clip(cs, 0, 150)
    vals, cnt = np.unique(np.round(x), return_counts=True)
    m10 = (vals % 10 == 0); m5 = (vals % 5 == 0) & ~m10
    ax.bar(vals[~m5 & ~m10], cnt[~m5 & ~m10] / len(cs) * 100, width=0.9, color=T["ref"], lw=0, label="other values")
    ax.bar(vals[m5], cnt[m5] / len(cs) * 100, width=0.9, color=T["s"][1], lw=0, label="multiples of 5")
    ax.bar(vals[m10], cnt[m10] / len(cs) * 100, width=0.9, color=T["s"][0], lw=0, label="multiples of 10")
    ax.set_xlim(0, 120); ax.set_xlabel("carbs per entry (g)"); ax.set_ylabel("% of entries")
    ax.set_title(f"Carb counts heap on round numbers ({OUT['carb_mult5'] * 100:.0f}% are ×5)"); ax.legend(fontsize=8.5)
    return fig
save_fig(f_ins_carbs, "t1_insulin_carbs", FIG)

# ------------------------------------------------------------------ T8 postprandial
PP = A["pp"]; tmin = np.arange(-6, 37) * 5
PM["mod"] = PM.key.map(S.set_index("key").insulin_delivery_modality)
PM["bolused"] = PM.bolus_pm15 > 0
OUT["pp_peak_by_mod"] = {m: float(np.nanmax(np.nanmedian(PP[(PM["mod"] == m).to_numpy()], 0))) for m in MODS if (PM["mod"] == m).any()}
CB = [(20, 40, "20–39 g"), (40, 60, "40–59 g"), (60, 90, "60–89 g"), (90, 1e9, "≥ 90 g")]
def f_pp(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True, gridspec_kw=dict(wspace=0.07))
    def band(ax, sel, col, lab):
        X = PP[sel]
        ax.fill_between(tmin, np.nanpercentile(X, 25, 0), np.nanpercentile(X, 75, 0), color=col, alpha=T["band_alpha"], lw=0)
        ax.plot(tmin, np.nanmedian(X, 0), color=col, lw=2.2, label=f"{lab} (n={sel.sum():,})")
    MC = MODC(T)
    for m in MODS:
        sel = (PM["mod"] == m).to_numpy()
        if sel.sum() > 100: band(axes[0], sel, MC[m], m)
    axes[0].set_title("By insulin delivery mode")
    cols = ramp(T, len(CB))
    for k, (a, b_, lab) in enumerate(CB):
        sel = ((PM.carbs >= a) & (PM.carbs < b_)).to_numpy()
        axes[1].plot(tmin, np.nanmedian(PP[sel], 0), color=cols[k], lw=2.2, label=f"{lab} (n={sel.sum():,})")
    axes[1].set_title("By carb amount (median)")
    has_bol = PM.source.isin(S.groupby("source").bolus_day.median().loc[lambda x: x > 0].index)
    for flag, col, lab in [(True, T["s"][0], "bolus within ±15 min"), (False, T["s"][1], "no bolus within ±15 min")]:
        sel = (has_bol & (PM.bolused == flag)).to_numpy()
        band(axes[2], sel, col, lab)
    axes[2].set_title("With vs without a meal bolus")
    for ax in axes:
        ax.axvline(0, color=T["ref"], ls=":", lw=1); ax.axhline(0, color=T["ref"], lw=0.8)
        ax.set_xlabel("minutes after carb entry"); ax.legend(fontsize=8.3, loc="upper right"); ax.set_xticks([-30, 0, 60, 120, 180])
    axes[0].set_ylabel("Δ CGM from entry time (mg/dL)")
    return fig
save_fig(f_pp, "t1_postprandial", FIG)

# ------------------------------------------------------------------ T9 demographics
def norm_eth(x):
    if not isinstance(x, str): return "Not reported"
    t = [p.strip().lower().replace("-", " ") for p in x.split(",")]
    races = set()
    for p in t:
        if "hispanic" in p: continue
        if "white" in p: races.add("White")
        elif "black" in p or "african" in p: races.add("Black")
        elif "asian" in p or "pakistani" in p or "middle eastern" in p: races.add("Asian")
        elif "indian" in p or "alaska" in p: races.add("AI/AN")
        elif "hawaiian" in p or "pacific" in p: races.add("NH/PI")
        elif "more than one" in p: races.add("Multi"); races.add("Multi2")
        elif "unknown" in p or "not reported" in p: pass
        else: races.add("Other")
    if len(races) >= 2: return "Multiracial"
    if len(races) == 1: return races.pop()
    return "Hispanic only" if any("hispanic" in p for p in t) else "Not reported"
S["race"] = S.ethnicity.map(norm_eth)
S["hispanic"] = S.ethnicity.fillna("").str.contains("Hispanic")
OUT["race"] = S.race.value_counts().to_dict(); OUT["hispanic_share"] = float(S.hispanic[S.ethnicity.notna()].mean())
OUT["ethnicity_raw_labels"] = int(S.ethnicity.nunique())
def f_demo(T):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1.1, 1]))
    ax = axes[0]
    q = S.dropna(subset=["age", "age_of_diagnosis"])
    ax.hexbin(q.age, q.age_of_diagnosis, gridsize=30, bins="log", mincnt=1, cmap=LinearSegmentedColormap.from_list("q", [T["seq"][0], T["seq"][4]] if T["panel"] == "#ffffff" else [T["seq"][0], T["seq"][4]]))
    ax.plot([0, 82], [0, 82], color=T["ref"], ls="--", lw=1)
    ax.set_xlabel("age (y)"); ax.set_ylabel("age at diagnosis (y)"); ax.set_title(f"Diabetes duration (median {np.median(q.age - q.age_of_diagnosis):.0f} y)")
    ax = axes[1]
    g = S.assign(gender=S.gender.fillna("Not reported")).groupby("source").gender.value_counts(normalize=True).unstack().reindex(order).fillna(0) * 100
    left = np.zeros(len(order))
    for k, (lab, col) in enumerate([("Female", T["s"][0]), ("Male", T["s"][1]), ("Non-binary", T["s"][3]), ("Not reported", T["grid"])]):
        if lab not in g: continue
        ax.barh(range(len(order)), g[lab], left=left, color=col, height=0.72, label=lab); left += g[lab].to_numpy()
    ax.set_yticks(range(len(order)), order, fontsize=8.6); ax.set_xlabel("% of participants"); ax.set_xlim(0, 100)
    ax.set_title("Gender by study"); ax.legend(fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.06), borderaxespad=0, handlelength=1, columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    ax = axes[2]
    r = S.race.value_counts()
    ax.barh(range(len(r))[::-1], r.values, color=T["s"][0], height=0.7)
    for yi, v in zip(range(len(r))[::-1], r.values):
        ax.text(v + r.max() * 0.015, yi, f"{v} ({v / len(S) * 100:.0f}%)", va="center", fontsize=8.5, color=T["ink"])
    ax.set_yticks(range(len(r))[::-1], r.index); ax.set_xlim(0, r.max() * 1.3); ax.set_xlabel("participants")
    ax.set_title(f"Race ({OUT['ethnicity_raw_labels']} raw labels → 8)"); ax.grid(axis="y", visible=False)
    return fig
save_fig(f_demo, "t1_demographics", FIG)

json.dump(OUT, open(os.path.join(E, "t1d_summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: v for k, v in OUT.items() if k not in ("source_table",)}, indent=0, default=str)[:6000])
