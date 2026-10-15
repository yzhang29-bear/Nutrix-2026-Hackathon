"""CGMacros data inventory: every data type, which participants have it (and how complete), what it looks like.
Writes eda/fig/cg_inventory.png, eda/fig/cg_oneday.png, eda/inventory.json and inserts (or replaces) the
"Data inventory" section at the top of eda/cgmacros_eda.html.
usage: python -I eda_inventory.py <CGMacros dir> <eda dir>
"""
import sys, os, re, glob, json, html
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps

DATA, EDA = sys.argv[1], sys.argv[2]
FIG = os.path.join(EDA, "fig")
os.makedirs(FIG, exist_ok=True)
bio = pd.read_csv(os.path.join(DATA, "bio.csv")); bio.columns = [c.strip() for c in bio.columns]
gut = pd.read_csv(os.path.join(DATA, "gut_health_test.csv")); gut.columns = [c.strip() for c in gut.columns]
mic = pd.read_csv(os.path.join(DATA, "microbes.csv")); mic.columns = [c.strip() for c in mic.columns]
BIO_FIELDS = [c for c in bio.columns if c != "subject"]

rows, minute_examples, vals = [], None, {k: [] for k in ["Dexcom GL", "Libre GL", "HR", "METs", "Calories (Activity)"]}
for f in sorted(glob.glob(os.path.join(DATA, "CGMacros-*", "CGMacros-*.csv"))):
    sid = os.path.basename(f)[9:12]
    d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    span = (d.Timestamp.max() - d.Timestamp.min()).total_seconds() / 86400
    meal = d[d["Meal Type"].notna()]
    img = d["Image path"].notna()
    exists = lambda p: isinstance(p, str) and os.path.exists(os.path.join(DATA, f"CGMacros-{sid}", p))
    act_col = "METs" if "METs" in d else ("Intensity" if "Intensity" in d else None)
    b = bio[bio.subject == int(sid)]
    r = dict(subject=sid, days=round(span, 1), minutes=len(d),
             dexcom=float(d["Dexcom GL"].notna().mean()), libre=float(d["Libre GL"].notna().mean()), hr=float(d["HR"].notna().mean()),
             activity=float(d[act_col].notna().mean()) if act_col else 0.0, activity_kind=act_col or "none",
             active_kcal=float(d["Calories (Activity)"].notna().mean()) if "Calories (Activity)" in d else 0.0,
             meals=int(len(meal)), meals_macros=int((meal[["Carbs", "Protein", "Fat"]].notna().all(1)).sum()),
             meals_photo=int(sum(exists(p) for p in meal["Image path"])), after_photos=int(sum(exists(p) for p in d.loc[img & d["Meal Type"].isna(), "Image path"])),
             breakfast=int((meal["Meal Type"].str.lower().str.strip() == "breakfast").sum()), lunch=int((meal["Meal Type"].str.lower().str.strip() == "lunch").sum()),
             dinner=int((meal["Meal Type"].str.lower().str.strip() == "dinner").sum()), snack=int(meal["Meal Type"].str.lower().str.startswith("snack").sum()),
             bio_fields=int(b[BIO_FIELDS].notna().sum(1).iloc[0]) if len(b) else 0, a1c=bool(len(b) and pd.notna(b["A1c PDL (Lab)"].iloc[0])),
             gut=bool((gut.subject == int(sid)).any()), microbes=bool((mic.subject == int(sid)).any()), steps="Steps" in d)
    rows.append(r)
    for k in vals:
        if k in d:
            vals[k].append(d[k].dropna().to_numpy(float))
    if sid == "001":
        minute_examples = d
INV = pd.DataFrame(rows)
N = len(INV)
q = {k: np.percentile(np.concatenate(v), [1, 50, 99]).round(1).tolist() for k, v in vals.items() if v}

# ---------------------------------------------------------------- figure 1: who has what
cols = [("days", "days recorded", lambda x: x.days / 12), ("dexcom", "Dexcom CGM", None), ("libre", "Libre CGM", None), ("hr", "heart rate", None),
        ("activity", "activity (METs or level)", None), ("active_kcal", "active kcal/min", None),
        ("meals", "meals logged", lambda x: x.meals / INV.meals.max()), ("meals_photo", "meals with photo", lambda x: x.meals_photo / x.meals.clip(lower=1)),
        ("meals_macros", "meals with macros", lambda x: x.meals_macros / x.meals.clip(lower=1)), ("bio_fields", "bio.csv fields", lambda x: x.bio_fields / len(BIO_FIELDS)),
        ("gut", "gut-health test", lambda x: x.gut.astype(float)), ("microbes", "microbiome", lambda x: x.microbes.astype(float))]
M = np.column_stack([(fn(INV) if fn else INV[c]).to_numpy(float) for c, _, fn in cols]).clip(0, 1)
lab = np.empty(M.shape, dtype=object)
for j, (c, _, _) in enumerate(cols):
    for i, r in INV.iterrows():
        v = r[c]
        lab[i, j] = (f"{v:.0f}" if c in ("days", "meals", "meals_photo", "meals_macros", "bio_fields") else ("✓" if v is True else ("✗" if v is False else f"{v * 100:.0f}")))
fig, ax = plt.subplots(figsize=(11, 12.5))
ax.imshow(M, cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, lab[i, j], ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.7 else "#333")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([n for _, n, _ in cols], rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(N)); ax.set_yticklabels([f"{s}  ({k})" if k == "Intensity" else s for s, k in zip(INV.subject, INV.activity_kind)], fontsize=7.5)
ax.set_title("Who has what: CGMacros, 45 participants\n(sensor columns: % of recorded minutes with a value; counts for meals and bio fields)", fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cg_inventory.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- figure 2: one participant, one day, every stream
d = minute_examples
day = d[(d.Timestamp >= "2020-05-02 06:00") & (d.Timestamp < "2020-05-03 00:00")]
fig, axs = plt.subplots(3, 1, figsize=(12, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
axs[0].plot(day.Timestamp, day["Dexcom GL"], color="#2b6cb0", lw=1.6, label="Dexcom (5-min, interpolated to 1 min)")
axs[0].plot(day.Timestamp, day["Libre GL"], color="#dd6b20", lw=1.2, label="Libre (15-min, interpolated to 1 min)")
axs[0].set_ylabel("glucose (mg/dL)"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
axs[1].plot(day.Timestamp, day["HR"], color="#c53030", lw=0.8); axs[1].set_ylabel("heart rate (bpm)")
axs[2].plot(day.Timestamp, day["METs"] / 10, color="#2f855a", lw=0.8, label="METs (column value / 10)")
axs[2].set_ylabel("METs"); ax2 = axs[2].twinx(); ax2.plot(day.Timestamp, day["Calories (Activity)"], color="#718096", lw=0.6, alpha=0.7)
ax2.set_ylabel("active kcal/min", color="#718096"); axs[2].legend(frameon=False, fontsize=8, loc="upper left")
lo_, hi_ = day[["Dexcom GL", "Libre GL"]].min().min(), day[["Dexcom GL", "Libre GL"]].max().max()
axs[0].set_ylim(lo_ - 5, hi_ + (hi_ - lo_) * 0.75)
top = axs[0].get_ylim()[1]
for _, m in day[day["Meal Type"].notna()].iterrows():
    for a in axs: a.axvline(m.Timestamp, color="#444", lw=0.8, ls=":")
    axs[0].text(m.Timestamp, top - 2, f" {m['Meal Type']}, {m.Carbs:.0f} g carbs", fontsize=8, va="top")
    p = os.path.join(DATA, "CGMacros-001", str(m["Image path"]))
    if os.path.exists(p):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); im.thumbnail((90, 90))
        axs[0].add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=0.55), (m.Timestamp, hi_ + (hi_ - lo_) * 0.38), frameon=False, xybox=(30, 0), boxcoords="offset points"))
for a in axs:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
axs[0].set_title("Participant 001, 2 May 2020: everything recorded in one day", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cg_oneday.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------- HTML section
def pct_people(col, thr=0.5):
    return int((INV[col] >= thr).sum())

e = lambda s: html.escape(str(s))
n_int = int((INV.activity_kind == "Intensity").sum())
med = lambda c: f"{INV[c].median() * 100:.0f}%"
types = [
    ("Dexcom glucose", "<code>Dexcom GL</code>", "5 min, filled to 1 min", "mg/dL", q["Dexcom GL"], f"{pct_people('dexcom')} of {N}", med("dexcom"), "101.8"),
    ("Libre glucose", "<code>Libre GL</code>", "15 min, filled to 1 min", "mg/dL", q["Libre GL"], f"{pct_people('libre')} of {N}", med("libre"), "88.4"),
    ("Heart rate", "<code>HR</code>", "every minute (Fitbit)", "beats/min", q["HR"], f"{pct_people('hr')} of {N}", med("hr"), "81"),
    ("Activity intensity", "<code>METs</code> (×10) or <code>Intensity</code>", "every minute (Fitbit)", "METs × 10, or level 0–3", q["METs"], f"{N - n_int} METs, {n_int} level only", med("activity"), "30 (= 3.0 METs, walking)"),
    ("Active calories", "<code>Calories (Activity)</code>", "every minute (Fitbit)", "kcal/min", q["Calories (Activity)"], f"{pct_people('active_kcal')} of {N}", med("active_kcal"), "3.15"),
    ("Meal", "<code>Meal Type</code>, <code>Calories</code>, <code>Carbs</code>, <code>Protein</code>, <code>Fat</code>, <code>Fiber</code>, <code>Amount Consumed</code>", "one row per meal", "g, kcal, %", None, f"{N} of {N}", f"{int(INV.meals.median())} meals each", "Breakfast: 448 kcal, 66 g carbs, 22 g protein, 10.5 g fat"),
    ("Meal photos", "<code>Image path</code> → <code>photos/*.jpg</code>", "before + after each meal", "JPEG", None, f"{int((INV.meals_photo > 0).sum())} of {N}", f"{INV.meals_photo.sum() / INV.meals.sum() * 100:.0f}% of meals", "photos/…-PHOTO-2020-5-2-8-18-0.jpg"),
    ("Demographics and labs", "<code>bio.csv</code>", "once per person", "years, lb, in, %, mg/dL, µU/mL", None, f"{int((INV.bio_fields > 0).sum())} of {N}", f"{int(INV.bio_fields.median())} of {len(BIO_FIELDS)} fields", "27 y, M, BMI 22.3, HbA1c 5.4%"),
    ("Gut-health scores", "<code>gut_health_test.csv</code>", "once per person", "1 = not optimal, 2 = average, 3 = good", None, f"{int(INV.gut.sum())} of {N}", "22 scores", "Gut Lining Health 2, Metabolic Fitness 3"),
    ("Microbiome", "<code>microbes.csv</code>", "once per person", "present (1) / absent (0)", None, f"{int(INV.microbes.sum())} of {N}", "1,979 taxa", f"{int((mic.iloc[0].drop('subject') == 1).sum())} of {mic.shape[1] - 1:,} taxa present"),
]
trs = "".join(f"<tr><td><b>{a}</b></td><td>{b_}</td><td>{c} · {d_}</td><td class='num'>{'' if r is None else f'{r[0]:g} / {r[1]:g} / {r[2]:g}'}</td><td>{p}</td><td>{m}</td><td>{e(x)}</td></tr>"
              for a, b_, c, d_, r, p, m, x in types)
gaps = []
low = INV[INV.dexcom < 0.8]
if len(low): gaps.append(f"Dexcom below 80% of minutes: {', '.join(low.subject)}")
lowh = INV[INV.hr < 0.5]
if len(lowh): gaps.append(f"heart rate below 50% of minutes: {', '.join(lowh.subject)}")
gaps.append(f"activity only as a 0–3 level (no METs): {', '.join(INV[INV.activity_kind == 'Intensity'].subject)}")
nophoto = INV[INV.meals_photo < 0.8 * INV.meals]
if len(nophoto): gaps.append(f"photos for fewer than 80% of meals: {', '.join(f'{s} ({p}/{m})' for s, p, m in zip(nophoto.subject, nophoto.meals_photo, nophoto.meals))}")
nok = INV[INV.active_kcal < 0.5]
if len(nok): gaps.append(f"active calories missing: {', '.join(f'{x} (has a Steps column instead)' if st else x for x, st in zip(nok.subject, nok.steps))}")
nog = INV[~INV.gut]; nom = INV[~INV.microbes]
if len(nog): gaps.append(f"no gut-health test: {', '.join(nog.subject)}")
if len(nom): gaps.append(f"no microbiome row: {', '.join(nom.subject)}")
gaps.append("gut-health table also has 2 IDs without any CGM folder (024, 025); IDs 024, 025, 037 and 040 have no data folder")
ex_rows = minute_examples.iloc[1305:1311][["Timestamp", "Libre GL", "Dexcom GL", "HR", "Calories (Activity)", "METs", "Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed", "Image path"]]
ex_rows = ex_rows.astype(object).where(ex_rows.notna(), "")
ex_txt = ex_rows.to_string(index=False, float_format=lambda v: f"{v:g}")
b1 = bio.iloc[0][BIO_FIELDS]
bio_txt = "\n".join(f"{k:<28} {v}" for k, v in b1.items())
g1 = gut.iloc[0].drop("subject"); gut_txt = "\n".join(f"{k:<44} {v:g}" for k, v in g1.items())
m1 = mic.iloc[0].drop("subject"); present = m1[m1 == 1].index.tolist()
mic_txt = f"{len(present)} of {len(m1)} taxa present, for example:\n" + "\n".join(f"  {t}" for t in present[:6])
SEC = f'''    <section id="inventory" aria-labelledby="h-inventory">
      <div class="sec-head"><span class="sec-num">01</span><h2 id="h-inventory">Data inventory: every data type, who has it, what it looks like</h2></div>
      <p>CGMacros has {N} participants with about {INV.days.median():.0f} days each. The time series sit in one CSV per person, one row per minute. Meals are extra columns on the minute rows, and photos are separate files. Demographics, labs, gut-health scores and the microbiome are one row per person in three shared tables.</p>
      <div class="tbl"><table>
        <thead><tr><th>Data type</th><th>Where (column / file)</th><th>Sampling · unit</th><th class="num">1st / 50th / 99th pct</th><th>Who has it</th><th>Completeness</th><th>Example</th></tr></thead>
        <tbody>{trs}</tbody>
      </table></div>
      <figure class="fig"><img src="fig/cg_inventory.png" alt="Heat map of 45 participants by 12 data types showing completeness" loading="lazy">
        <figcaption>Who has what. Every participant has both CGMs, heart rate, meals and the bio table. The differences are in completeness, the kind of activity data and the side tables.</figcaption></figure>
      <div class="callout"><b>Gaps to know about</b><ul class="plain" style="margin-top:6px">{"".join(f"<li>{e(g)}</li>" for g in gaps)}</ul></div>
      <figure class="fig"><img src="fig/cg_oneday.png" alt="One day of participant 001: Dexcom and Libre glucose, heart rate, METs and active calories, with meal times and photos" loading="lazy">
        <figcaption>One day of one participant, with every stream on a shared time axis. The Libre reads lower than the Dexcom throughout (section 04).</figcaption></figure>
      <details><summary>Raw examples: minute rows, bio.csv, gut-health and microbiome rows (participant 001)</summary>
        <p><b>Minute rows around a meal</b> (<code>CGMacros-001.csv</code>). Only the meal row carries meal type, nutrition and the photo path. METs are stored ×10.</p>
<pre class="mono" style="overflow-x:auto;font-size:12px">{e(ex_txt)}</pre>
        <p><b>bio.csv</b> (one row per person; weight in lb, height in inches):</p>
<pre class="mono" style="overflow-x:auto;font-size:12px">{e(bio_txt)}</pre>
        <p><b>gut_health_test.csv</b> (22 scores, 1 = not optimal, 2 = average, 3 = good):</p>
<pre class="mono" style="overflow-x:auto;font-size:12px">{e(gut_txt)}</pre>
        <p><b>microbes.csv</b> (presence of each taxon):</p>
<pre class="mono" style="overflow-x:auto;font-size:12px">{e(mic_txt)}</pre>
      </details>
    </section>
'''
p = os.path.join(EDA, "cgmacros_eda.html")
s = open(p).read()
if 'id="inventory"' in s:                               # replace an earlier version
    a = s.index('    <section id="inventory"'); b = s.index("</section>", a) + len("</section>\n")
    s = s[:a] + s[b:]
else:                                                    # shift the existing section numbers by one
    s = re.sub(r'<span class="sec-num">(\d\d)</span>', lambda m: f'<span class="sec-num">{int(m.group(1)) + 1:02d}</span>', s)
a = s.index('    <section id="structure"')
s = s[:a] + SEC + "\n" + s[a:]
open(p, "w").write(s)
INV.to_csv(os.path.join(EDA, "inventory_by_participant.csv"), index=False)
json.dump(dict(n=N, percentiles=q, gaps=gaps), open(os.path.join(EDA, "inventory.json"), "w"), indent=1)
print(INV.describe().round(2).T[["mean", "min", "max"]].to_string()); print("\n".join(gaps))
