# Pick up to 4 day-8+ CGMacros meals per participant (one per meal type first) with Dexcom coverage.
# Usage: python ops/data/make_test_images.py data/CGMacros <out_dir>  (out_dir must contain photos/)
import sys, shutil, pandas as pd
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for f in sorted(root.glob("CGMacros-*/CGMacros-*.csv")):
    pid = f.parent.name.split("-")[1]
    d = pd.read_csv(f, parse_dates=["Timestamp"])
    day1 = d["Timestamp"].dt.normalize().min()
    m = d[d["Meal Type"].notna() & d["Image path"].notna()].copy()
    m["day"] = (m["Timestamp"].dt.normalize() - day1).dt.days + 1
    m = m[m["day"] >= 8]
    # Dexcom must exist at meal time and 120 min after
    dex = d.set_index("Timestamp")["Dexcom GL"]
    def ok(t):
        w = dex[t - pd.Timedelta("10min"): t + pd.Timedelta("120min")]
        return w.notna().mean() > 0.05
    m = m[m["Timestamp"].apply(ok)]
    m["mt"] = m["Meal Type"].str.strip().str.lower().replace({"snacks": "snack"})
    pick = m.groupby("mt", group_keys=False).head(1)          # one per meal type first
    pick = pd.concat([pick, m.drop(pick.index)]).head(4)      # top up to 4
    for _, r in pick.iterrows():
        src = f.parent / r["Image path"].strip()
        if not src.exists(): continue
        name = f"p{pid}_{r['Timestamp']:%Y%m%dT%H%M}_{r['mt']}.jpg"
        shutil.copy2(src, out / "photos" / name)
        rows.append(dict(photo=f"photos/{name}", person_id=pid, day=r["day"], time=r["Timestamp"],
                         meal_type=r["mt"], carbs_g=r["Carbs"], protein_g=r["Protein"], fat_g=r["Fat"],
                         fiber_g=r["Fiber"], kcal=r["Calories"], source=f"{f.parent.name}/{r['Image path'].strip()}"))
t = pd.DataFrame(rows); t.to_csv(out / "test_meals.csv", index=False)
print(len(t), "photos from", t.person_id.nunique(), "people"); print(t.meal_type.value_counts().to_string())
