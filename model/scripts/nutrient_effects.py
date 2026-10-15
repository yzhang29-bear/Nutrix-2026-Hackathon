"""Within-person nutrient effects for the diet guide.
Model: rise_ij = person_i + b_carb*carbs + b_prot*protein + b_fat*fat + b_fiber*fiber + meal-type + pre-meal glucose terms,
with person fixed effects (each person compared only with themselves) and a person-specific carb slope shrunk to the population.
Validation: estimated WITHOUT breakfasts, then compared with the effects measured in the breakfast-shake experiment.
usage: python -I nutrient_effects.py <stage1 dir> <out dir>
"""
import sys, os, json, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd

S1, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
M = st["outer"]["M"].copy()
M = M[M.label].copy()
M["fiber"] = M.fiber.fillna(0)
NUT = ["carbs", "protein", "fat", "fiber"]

def fit(D, k_shrink=10.0):
    X = D[NUT + ["G0", "is_lunch", "is_dinner", "is_snack"]].astype(float)
    y = D.dpeak.astype(float)
    Xd = X - X.groupby(D.pid).transform("mean"); yd = y - y.groupby(D.pid).transform("mean")     # within-person
    beta, *_ = np.linalg.lstsq(Xd.to_numpy(), yd.to_numpy(), rcond=None)
    b = dict(zip(X.columns, beta))
    res = yd - Xd.to_numpy() @ beta
    # person-specific carb slope: regress the person's residual on their (demeaned) carbs, shrink toward 0 deviation
    dev = {}
    for pid, g in D.groupby("pid"):
        xc = Xd.loc[g.index, "carbs"].to_numpy(); r = res.loc[g.index].to_numpy()
        n = len(g); sxx = float(np.sum(xc ** 2))
        dev[pid] = float(np.sum(xc * r) / (sxx + k_shrink * np.mean(Xd.carbs ** 2) * 1.0)) if n >= 3 and sxx > 0 else 0.0
    person = D.groupby("pid").agg(mean_rise=("dpeak", "mean"), mean_carbs=("carbs", "mean"), n=("dpeak", "size"))
    person["carb_slope"] = b["carbs"] + person.index.map(dev)
    return b, person

# 1) validation fit: everything except breakfasts (CGMacros days 1-7 + all BIG IDEAs)
hist = M[((M.source == "cgmacros") & M.in_hist) | (M.source == "bigideas")]
b_val, _ = fit(hist[hist.meal_type != "breakfast"])
shakes = M[(M.source == "cgmacros") & (M.meal_type == "breakfast")]
ref = shakes[(shakes.carbs == 66) & (shakes.protein == 22) & (shakes.fat == 10.5)]
def observed(mask):
    d = (shakes[mask].groupby("pid").dpeak.mean() - ref.groupby("pid").dpeak.mean()).dropna()
    return float(d.mean()), float(1.96 * d.sem()), int(len(d))
checks = {"carbs 66 -> 24 g": (dict(carbs=-42), (shakes.carbs == 24) & (shakes.protein == 22)),
          "+44 g protein": (dict(protein=44), (shakes.carbs == 66) & (shakes.protein == 66) & (shakes.fat == 10.5)),
          "+31.5 g fat": (dict(fat=31.5), (shakes.carbs == 66) & (shakes.protein == 22) & (shakes.fat == 42))}
val = {}
for name, (chg, mask) in checks.items():
    pred = sum(b_val[k] * v for k, v in chg.items())
    o, ci, n = observed(mask)
    val[name] = dict(predicted_change_mgdl=round(pred, 1), observed_change_mgdl=round(o, 1), observed_ci95=round(ci, 1), people=n)
# 2) deployment fit: all history meals (breakfasts included)
b_all, person = fit(hist)
json.dump(dict(validation_without_breakfasts=val, coef_validation={k: round(v, 3) for k, v in b_val.items()},
               coef_deployed={k: round(v, 3) for k, v in b_all.items()}), open(os.path.join(OUT, "nutrient_effects.json"), "w"), indent=1)
pickle.dump(dict(coef=b_all, person=person), open(os.path.join(OUT, "nutrient_effects.pkl"), "wb"))
print(json.dumps(val, indent=1)); print({k: round(v, 3) for k, v in b_all.items()})
print(person.carb_slope.describe().round(3).to_dict())

# ---------------------------------------------------------------------------------------------------------------
# 3) personal nutrient effects measured in the person's own shake contrasts during days 1-7 (designed, so not attenuated
#    by logging error), shrunk toward the population effect; validated on the shakes they drank on day 8+.
RECIPES = {"reference": (66, 22, 10.5, 0), "low carb": (24, 22, 10.5, 0), "+protein": (66, 66, 10.5, 0), "+fat": (66, 22, 42, 0),
           "+fat+fiber": (73, 22, 42, 7), "+protein+fat+fiber": (73, 66, 42, 7)}
S = M[(M.source == "cgmacros") & (M.meal_type == "breakfast")].copy()
S["recipe"] = None
for k, (c, p, f, fb) in RECIPES.items():
    S.loc[(S.carbs == c) & (S.protein == p) & (S.fat == f) & (S.fiber.round() == fb), "recipe"] = k
S = S.dropna(subset=["recipe"])
H7, T8 = S[S.in_hist], S[~S.in_hist]
r7 = H7.groupby(["pid", "recipe"]).dpeak.agg(["mean", "size"]).unstack()
contrasts = [k for k in RECIPES if k != "reference" and k in r7["mean"].columns]   # recipes served during days 1-7
pop = {k: float((r7["mean"][k] - r7["mean"]["reference"]).dropna().mean()) for k in contrasts}

def personal_effect(pid, k, kshrink=1.0):
    try:
        d = r7["mean"].loc[pid, k] - r7["mean"].loc[pid, "reference"]; n = min(r7["size"].loc[pid, k], r7["size"].loc[pid, "reference"])
    except KeyError:
        return pop[k]
    if not np.isfinite(d):
        return pop[k]
    return float((n * d + kshrink * pop[k]) / (n + kshrink))

base = r7["mean"]["reference"]
fp = pickle.load(open(os.path.join(os.path.dirname(S1.rstrip("/")), "final_blend", "final_preds.pkl"), "rb"))
main_pred = pd.Series(fp["blend"]["dpeak"], index=fp["te_index"])
rows = []
for i, r in T8.iterrows():
    if r.pid not in base.index or not np.isfinite(base.get(r.pid, np.nan)) or (r.recipe != "reference" and r.recipe not in pop):
        continue
    b0 = base[r.pid]
    eff_pop = 0.0 if r.recipe == "reference" else pop[r.recipe]
    eff_per = 0.0 if r.recipe == "reference" else personal_effect(r.pid, r.recipe)
    rows.append(dict(pid=r.pid, recipe=r.recipe, observed=r.dpeak, own_reference_only=b0, population_effects=b0 + eff_pop,
                     personal_effects=b0 + eff_per, main_model=main_pred.get(i, np.nan)))
V = pd.DataFrame(rows)
mae = {c: float(np.mean(np.abs(V[c] - V.observed))) for c in ["own_reference_only", "population_effects", "personal_effects", "main_model"] if V[c].notna().all()}
mae_nonref = {c: float(np.mean(np.abs(V[c] - V.observed)[V.recipe != "reference"])) for c in ["own_reference_only", "population_effects", "personal_effects"]}
out = dict(population_effects_days1_7={k: round(v, 1) for k, v in pop.items()}, n_test_shakes=int(len(V)), n_people=int(V.pid.nunique()),
           mae_test_shakes=mae, mae_test_shakes_excluding_reference=mae_nonref)
json.dump(out, open(os.path.join(OUT, "shake_guide_validation.json"), "w"), indent=1)
pickle.dump(dict(pop=pop, r7=r7, recipes=RECIPES), open(os.path.join(OUT, "shake_effects.pkl"), "wb"))
print(json.dumps(out, indent=1))
