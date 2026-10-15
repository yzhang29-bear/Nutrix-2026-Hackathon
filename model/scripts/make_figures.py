"""CGMacros overview figures. usage: python -I make_figures.py <CGMacros dir> <out dir>"""
import sys, os, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.facecolor": "white"})

def norm_meal(s):
    s = str(s).strip().lower()
    return "snack" if s.startswith("snack") else s

def photo_time(fn):
    m = re.search(r"PHOTO-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", fn)
    return pd.Timestamp(*map(int, m.groups())) if m else None

def load_thumb(path, h=360, aspect=0.75):
    """center-crop to a fixed portrait aspect so landscape photos don't break the grid"""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return ImageOps.fit(im, (int(h * aspect), h))

data = {}
for f in sorted(glob.glob(os.path.join(ROOT, "CGMacros-*", "CGMacros-*.csv"))):
    d = pd.read_csv(f)
    d.columns = [c.strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    d["meal"] = d["Meal Type"].map(lambda x: norm_meal(x) if pd.notna(x) else np.nan)
    data[os.path.basename(f)[9:12]] = d

def after_photo(sid, before_rel):
    """the next photo in the folder within 2 h of the before-photo = after-meal photo"""
    pdir = os.path.join(ROOT, f"CGMacros-{sid}", "photos")
    files = sorted(os.listdir(pdir))
    b = os.path.basename(before_rel)
    if b not in files: return None
    i = files.index(b)
    meal_paths = set(os.path.basename(p) for p in data[sid].loc[data[sid]["meal"].notna(), "Image path"].dropna())
    if i + 1 < len(files):
        nxt = files[i + 1]
        t0, t1 = photo_time(b), photo_time(nxt)
        if nxt not in meal_paths and t0 and t1 and 0 <= (t1 - t0).total_seconds() <= 7200:
            return os.path.join(pdir, nxt)
    return None

# ---------- Fig 1: meal photo wall (before | after) ----------
meals = []
for sid, d in data.items():
    m = d[d["meal"].notna() & d["Image path"].notna()]
    for _, r in m.iterrows():
        meals.append(dict(sid=sid, meal=r["meal"], kcal=r["Calories"], C=r["Carbs"], P=r["Protein"],
                          F=r["Fat"], Fi=r["Fiber"], before=os.path.join(ROOT, f"CGMacros-{sid}", r["Image path"])))
M = pd.DataFrame(meals)
rng = np.random.default_rng(7)
pick = []
for meal, n in [("breakfast", 4), ("lunch", 4), ("dinner", 4), ("snack", 4)]:
    sub = M[(M.meal == meal)].copy()
    sub = sub.groupby("sid").head(1)                       # one per person -> diversity
    sub = sub.iloc[rng.permutation(len(sub))]
    got = 0
    for _, r in sub.iterrows():
        a = after_photo(r.sid, r.before)
        if a and os.path.exists(r.before):
            pick.append((r, a)); got += 1
        if got == n: break

ncol_pairs, nrow = 4, 4
fig, axes = plt.subplots(nrow, ncol_pairs * 2, figsize=(16, 3.45 * nrow), gridspec_kw=dict(wspace=0.03, hspace=0.32))
for k, (r, a) in enumerate(pick):
    i, j = divmod(k, ncol_pairs)
    for jj, (p, tag) in enumerate([(r.before, "before"), (a, "after")]):
        ax = axes[i, 2 * j + jj]
        ax.imshow(load_thumb(p)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(tag, fontsize=8, color=INK2, pad=2)
    axes[i, 2 * j].text(1.015, -0.03, f"S{r.sid} · {r.meal} · {r.kcal:.0f} kcal\ncarb {r.C:.0f}g  protein {r.P:.0f}g  fat {r.F:.0f}g  fiber {r.Fi:.0f}g",
                        transform=axes[i, 2 * j].transAxes, ha="center", va="top", fontsize=8.5, color=INK)
fig.suptitle("CGMacros meal photos (before / after eating), with logged macronutrients", fontsize=14, color=INK, y=0.94)
fig.savefig(os.path.join(OUT, "fig1_meal_photos.jpg"), dpi=90, bbox_inches="tight", pil_kwargs={"quality": 85})
plt.close(fig)

# ---------- Fig 2: one person, one day ----------
sid = "001"; d = data[sid]
day = pd.Timestamp("2020-05-03")
dd = d[(d.Timestamp >= day) & (d.Timestamp < day + pd.Timedelta("1D"))]
mm = dd[dd["meal"].notna()]
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 2.4, 1], hspace=0.15)
axg = fig.add_subplot(gs[1]); axh = fig.add_subplot(gs[2], sharex=axg)
axg.plot(dd.Timestamp, dd["Libre GL"], color=BLUE, lw=2, label="Libre Pro (arm, 15-min, interpolated to 1-min)")
dx = dd[dd["Dexcom GL"].notna()]
axg.plot(dx.Timestamp, dx["Dexcom GL"], color=ORANGE, lw=2, label="Dexcom G6 Pro (abdomen, 5-min)")
axg.set_ylabel("glucose (mg/dL)")
axg.legend(loc="lower right", frameon=False, fontsize=9)
axh.plot(dd.Timestamp, dd["HR"], color=INK2, lw=1)
axh.set_ylabel("heart rate (bpm)\nFitbit Sense")
for ax in (axg, axh):
    for t in mm.Timestamp: ax.axvline(t, color=INK2, lw=0.8, ls=":")
plt.setp(axg.get_xticklabels(), visible=False)
import matplotlib.dates as mdates
axh.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# photo strip on top, each photo anchored to its meal time
axp = fig.add_subplot(gs[0], sharex=axg); axp.axis("off")
x0, x1 = mdates.date2num(day), mdates.date2num(day + pd.Timedelta("1D"))
axg.set_xlim(x0, x1)
for k, (_, r) in enumerate(mm.iterrows()):
    t = mdates.date2num(r.Timestamp)
    axg.annotate(f"{r['meal']}\n{r['Calories']:.0f} kcal", (t, axg.get_ylim()[1]), xytext=(3, -4 - 24 * (k % 2)), textcoords="offset points",
                 fontsize=8, color=INK, va="top")
    if pd.notna(r["Image path"]):
        p = os.path.join(ROOT, f"CGMacros-{sid}", r["Image path"])
        if os.path.exists(p):
            im = load_thumb(p, 240)
            ia = axp.inset_axes([(t - x0) / (x1 - x0) - 0.025, 0.02, 0.05, 0.96])
            ia.imshow(im); ia.axis("off")
fig.suptitle(f"Participant {sid}, one day: CGM glucose + heart rate + meals (photos on top)", fontsize=13, color=INK, y=0.93)
fig.savefig(os.path.join(OUT, "fig2_one_day.png"), dpi=110, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 3: post-meal glucose response by diabetes status ----------
bio = pd.read_csv(os.path.join(ROOT, "bio.csv")); bio.columns = [c.strip() for c in bio.columns]
bio["sid"] = bio["subject"].map(lambda x: f"{int(x):03d}")
a1c = bio.set_index("sid")["A1c PDL (Lab)"]
grp = lambda v: "healthy (A1c<5.7)" if v < 5.7 else ("prediabetes (5.7-6.4)" if v <= 6.4 else "type 2 diabetes (>6.4)")
G = {g: c for g, c in zip(["healthy (A1c<5.7)", "prediabetes (5.7-6.4)", "type 2 diabetes (>6.4)"], [BLUE, AQUA, ORANGE])}
win = np.arange(-15, 181)
curves = {(g, ml): [] for g in G for ml in ["breakfast", "lunch"]}
for sid, d in data.items():
    if sid not in a1c.index or pd.isna(a1c[sid]): continue
    g = grp(a1c[sid]); s = d.set_index("Timestamp")["Libre GL"]
    for _, r in d[d["meal"].isin(["breakfast", "lunch"])].iterrows():
        idx = r.Timestamp + pd.to_timedelta(win, unit="min")
        v = s.reindex(idx).to_numpy()
        if np.isnan(v).mean() > 0.1: continue
        curves[(g, r["meal"])].append(v - np.nanmean(v[:16]))
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, ml in zip(axes, ["breakfast", "lunch"]):
    for g, c in G.items():
        A = np.array(curves[(g, ml)])
        if len(A) == 0: continue
        med = np.nanmedian(A, 0); q1, q3 = np.nanpercentile(A, [25, 75], 0)
        ax.fill_between(win, q1, q3, color=c, alpha=0.15, lw=0)
        ax.plot(win, med, color=c, lw=2, label=f"{g}  (n={len(A)} meals)")
    ax.axvline(0, color=INK2, lw=0.8, ls=":"); ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(f"{ml}", color=INK); ax.set_xlabel("minutes after meal start")
axes[0].set_ylabel("Δ glucose from pre-meal (mg/dL), Libre")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Post-meal glucose response by diabetes status (median, band = IQR)", fontsize=13, color=INK)
fig.savefig(os.path.join(OUT, "fig3_ppgr_by_group.png"), dpi=110, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 4: cohort + meal macro overview ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ax = axes[0]
for g, c in G.items():
    b = bio[bio.sid.map(lambda s: s in a1c.index and grp(a1c[s]) == g)]
    ax.scatter(b["BMI"], b["A1c PDL (Lab)"], s=40, color=c, edgecolor="white", lw=1.5, label=g, zorder=3)
ax.set_xlabel("BMI (kg/m²)"); ax.set_ylabel("HbA1c (%)"); ax.set_title("45 participants", color=INK)
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
Mall = pd.concat([d[d["meal"].notna()].assign(sid=s) for s, d in data.items()])
order = ["breakfast", "lunch", "dinner", "snack"]
cnt = Mall["meal"].value_counts().reindex(order)
ax.bar(order, cnt.values, color=BLUE, width=0.6)
for i, v in enumerate(cnt.values): ax.text(i, v + 8, str(v), ha="center", fontsize=9, color=INK)
ax.set_title(f"{len(Mall)} logged meals", color=INK); ax.set_ylabel("count"); ax.grid(axis="x", visible=False)
ax = axes[2]
for i, ml in enumerate(order):
    v = Mall.loc[Mall.meal == ml, "Calories"].clip(upper=2000)
    ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE), medianprops=dict(color=BLUE, lw=2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
ax.set_xticks(range(4), order); ax.set_ylabel("kcal"); ax.set_title("calories per meal", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_overview.png"), dpi=110, bbox_inches="tight")
print("picked", len(pick), "meal pairs; curves:", {k: len(v) for k, v in curves.items()})
