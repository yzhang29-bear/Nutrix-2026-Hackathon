"""Nutrition-driven model for the diet guide. The main predictor leans on photo features, which stay fixed in a counterfactual
("same photo, fewer carbs"), so it under-reacts to nutrient changes. This model sees logged nutrition, pre-meal context and the
personal profile only, so nutrient changes move its prediction.
Checks: (1) accuracy on day-8+ meals using logged nutrition; (2) its counterfactual effects reproduce the within-person effects
measured in the CGMacros breakfast-shake experiment (carbs 66 -> 24 g, +44 g protein, +31.5 g fat).
usage: python -I train_guide_model.py <stage1 dir> <out dir>
"""
import sys, os, json, pickle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import numpy as np, pandas as pd
from scipy import stats
from pmrf.features import CONTEXT_FEATURES, MED_FEATURES, PROFILE_FEATURES
from pmrf.pipeline import MEALTYPE_FEATURES, LOGGED_MACROS, metrics, to_challenge_outputs
from pmrf.models.gbm import GBMResponse

S1, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
st = pickle.load(open(os.path.join(S1, "stage1.pkl"), "rb"))
TR, TE = st["TR"].copy(), st["TE"].copy()
for d in (TR, TE):
    d["fiber"] = d["fiber"].fillna(0)
FEATS = CONTEXT_FEATURES + MED_FEATURES + MEALTYPE_FEATURES + LOGGED_MACROS + PROFILE_FEATURES
YS = ["d30", "d60", "d90", "d120", "dpeak", "ttp", "iauc"]
m = GBMResponse(FEATS, targets=YS, personal_offset=False, params=st["params"], seeds=(0, 1, 2), log_targets=st.get("log_targets", {}),
                log_point=st.get("log_point", "median")).fit(TR)
P = m.predict(TE)
res = {"test_logged_nutrition": metrics(TE, to_challenge_outputs(TE, P), {"unseen": ~TE.seen.to_numpy()}).set_index("target")[["mae", "r"]].round(3).to_dict("index")}

# counterfactual check on the shake experiment: for each test-day reference shake (66 g carbs, 22 protein, 10.5 fat), predict the
# same meal with one nutrient changed, and compare the predicted change with the within-person change measured in all data
shakes = st["outer"]["M"]
shakes = shakes[(shakes.source == "cgmacros") & shakes.label & (shakes.meal_type == "breakfast")]
ref = shakes[(shakes.carbs == 66) & (shakes.protein == 22) & (shakes.fat == 10.5)]
def observed_effect(mask):
    a = shakes[mask].groupby("pid").dpeak.mean(); b = ref.groupby("pid").dpeak.mean()
    d = (a - b).dropna(); return float(d.mean()), float(d.sem()), int(len(d))
obs = {"carbs 66 -> 24 g": observed_effect((shakes.carbs == 24) & (shakes.protein == 22)),
       "+44 g protein": observed_effect((shakes.carbs == 66) & (shakes.protein == 66) & (shakes.fat == 10.5)),
       "+31.5 g fat": observed_effect((shakes.carbs == 66) & (shakes.protein == 22) & (shakes.fat == 42))}
X0 = TE[(TE.meal_type == "breakfast") & (TE.carbs == 66) & (TE.protein == 22) & (TE.fat == 10.5)].copy()
def pred_dpeak(model, X, feats_photo=False):
    return model.predict(X)["dpeak"]["pred"]
cf = {}
for name, chg in [("carbs 66 -> 24 g", dict(carbs=24)), ("+44 g protein", dict(protein=66)), ("+31.5 g fat", dict(fat=42))]:
    X1 = X0.copy()
    for k, v in chg.items():
        X1[k] = v
    X1["kcal"] = 4 * X1.carbs + 4 * X1.protein + 9 * X1.fat
    cf[name] = float(np.mean(pred_dpeak(m, X1) - pred_dpeak(m, X0)))
res["shake_counterfactual"] = {k: dict(model_change_mgdl=round(cf[k], 1), observed_change_mgdl=round(obs[k][0], 1),
                                       observed_se=round(obs[k][1], 1), people=obs[k][2]) for k in cf}
res["n_reference_shakes_on_test_days"] = int(len(X0))
json.dump(res, open(os.path.join(OUT, "guide_model.json"), "w"), indent=1)
pickle.dump(dict(model=m, features=FEATS), open(os.path.join(OUT, "guide_model.pkl"), "wb"))
print(json.dumps(res, indent=1))
