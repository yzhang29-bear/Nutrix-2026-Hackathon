import sys, glob, os
import pandas as pd
root = sys.argv[1]
subs = sorted(glob.glob(os.path.join(root, "CGMacros-*", "CGMacros-*.csv")))
rows = []
meals_all = []
for f in subs:
    d = pd.read_csv(f)
    sid = os.path.basename(f)[9:12]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    m = d[d["Meal Type"].notna()].copy()
    m["sid"] = sid
    meals_all.append(m)
    rows.append(dict(sid=sid, rows=len(d), days=round((d.Timestamp.max()-d.Timestamp.min()).total_seconds()/86400, 1),
                     libre=d["Libre GL"].notna().mean().round(2), dexcom=d["Dexcom GL"].notna().mean().round(2),
                     hr=d["HR"].notna().mean().round(2), meals=len(m),
                     photos=m["Image path"].notna().sum(),
                     n_photo_files=len(glob.glob(os.path.join(os.path.dirname(f), "photos", "*")))))
s = pd.DataFrame(rows)
print(s.to_string())
print("total meals", s.meals.sum(), "photo files", s.n_photo_files.sum())
M = pd.concat(meals_all)
print(M["Meal Type"].value_counts())
print(M[["Calories","Carbs","Protein","Fat","Fiber","Amount Consumed "]].describe().round(1).to_string())
print(M[["sid","Timestamp","Meal Type","Calories","Carbs","Protein","Fat","Fiber","Image path"]].head(15).to_string())
print(d.columns.tolist())
