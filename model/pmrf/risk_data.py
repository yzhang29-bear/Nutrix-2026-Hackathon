"""Loaders for the cohorts used by the metabolic profile & risk module (CGM + lab HbA1c / outcomes).

Each loader returns a list of records: dict(cohort, pid, glucose=<regular-grid Series, mg/dL>, hba1c=<% or NaN>, ...).
Glucose is put on the sensor's native grid with pmrf.cgm_metrics.regularize (5 min, or 15 min for Shanghai).
The data folders are downloaded third-party data: they are only read, never written.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

from pmrf.cgm_metrics import regularize

SHANGHAI_DIR = "2022.11子表补充中文饮食"
MMOL_PER_MOL_TO_PCT = lambda x: x / 10.929 + 2.15          # IFCC (mmol/mol) -> NGSP (%) master equation


def _num(x):
    return pd.to_numeric(pd.Series(x).astype(str).str.strip().replace({"/": np.nan, "": np.nan, "nan": np.nan}),
                         errors="coerce").to_numpy(float)


# ---------------------------------------------------------------- CGMacros (Dexcom G6, 1-min interpolated -> 5-min bins)
def load_cgmacros(root):
    bio = pd.read_csv(os.path.join(root, "bio.csv"))
    bio.columns = [c.strip() for c in bio.columns]
    bio = bio.set_index(bio.subject.astype(int).map(lambda s: f"{s:03d}"))
    recs = []
    for f in sorted(glob.glob(os.path.join(root, "CGMacros-*", "CGMacros-*.csv"))):
        pid = os.path.basename(f)[9:12]
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d["Timestamp"] = pd.to_datetime(d["Timestamp"])
        d = d.drop_duplicates("Timestamp").sort_values("Timestamp")
        g = regularize(d.set_index("Timestamp")["Dexcom GL"], interval_min=5)
        b = bio.loc[pid] if pid in bio.index else None
        recs.append(dict(cohort="CGMacros", pid=pid, glucose=g,
                         hba1c=float(b["A1c PDL (Lab)"]) if b is not None else np.nan,
                         sex=None if b is None else ("F" if str(b["Gender"]).strip().upper().startswith("F") else "M"),
                         age=float(b["Age"]) if b is not None else np.nan, bmi=float(b["BMI"]) if b is not None else np.nan,
                         fasting_glucose=float(b["Fasting GLU - PDL (Lab)"]) if b is not None else np.nan))
    return recs


# ---------------------------------------------------------------- BIG IDEAs (Dexcom G6 export, 5-min)
def load_bigideas(root):
    demo = pd.read_csv(os.path.join(root, "Demographics.csv"), encoding="utf-8-sig")
    demo = demo.set_index(demo.ID.astype(int).map(lambda s: f"{s:03d}"))
    recs = []
    for f in sorted(glob.glob(os.path.join(root, "[0-9][0-9][0-9]", "Dexcom_*.csv"))):
        pid = os.path.basename(os.path.dirname(f))
        d = pd.read_csv(f)
        d = d[(d["Event Type"] == "EGV") & d["Timestamp (YYYY-MM-DDThh:mm:ss)"].notna()]
        t = pd.to_datetime(d["Timestamp (YYYY-MM-DDThh:mm:ss)"])
        v = pd.to_numeric(d["Glucose Value (mg/dL)"], errors="coerce").to_numpy(float)
        g = regularize(pd.Series(v, index=t.values), interval_min=5)
        r = demo.loc[pid]
        recs.append(dict(cohort="BIG IDEAs", pid=pid, glucose=g, hba1c=float(r["HbA1c"]),
                         sex="F" if str(r["Gender"]).upper().startswith("F") else "M", age=np.nan, bmi=np.nan,
                         fasting_glucose=np.nan))
    return recs


# ---------------------------------------------------------------- Shanghai T1DM / T2DM (Zhao et al. 2023, 15-min CGM)
def _shanghai_summary(path):
    s = pd.read_excel(path)
    s.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in s.columns]   # collapses the double spaces in headers
    dur = [c for c in s.columns if c.lower().startswith("duration of diabetes")][0]
    out = pd.DataFrame(dict(
        session=s["Patient Number"].astype(str).str.strip(),
        sex=np.where(s["Gender (Female=1, Male=2)"] == 1, "F", "M"),
        age=_num(s["Age (years)"]), bmi=_num(s["BMI (kg/m2)"]), dm_type=s["Type of Diabetes"].astype(str).str.strip(),
        duration=_num(s[dur]), hba1c_mmol=_num(s["HbA1c (mmol/mol)"]),
        fasting_glucose=_num(s["Fasting Plasma Glucose (mg/dl)"]),
        macro=s["Diabetic Macrovascular Complications"].astype(str).str.strip().str.lower(),
        micro=s["Diabetic Microvascular Complications"].astype(str).str.strip().str.lower(),
        agents=s["Hypoglycemic Agents"].astype(str).str.strip(),
        hypoglycemia=s["Hypoglycemia (yes/no)"].astype(str).str.strip().str.lower(),
    ))
    out["hba1c"] = MMOL_PER_MOL_TO_PCT(out.hba1c_mmol).round(2)
    out["patient"] = out.session.str.split("_").str[0]
    out["session_idx"] = out.session.str.split("_").str[1].astype(int)
    out["date"] = pd.to_datetime(out.session.str.split("_").str[2], format="%Y%m%d")
    return out


def _shanghai_cgm(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = pd.read_excel(path)
    x.columns = [str(c).strip() for c in x.columns]
    col = [c for c in x.columns if c.startswith("CGM")][0]        # "CGM (mg / dl)" or, in 2 files, "CGM"
    t = pd.to_datetime(x["Date"], errors="coerce")
    v = pd.to_numeric(x[col], errors="coerce")
    ok = t.notna() & v.notna()
    return regularize(pd.Series(v[ok].to_numpy(float), index=t[ok].values), interval_min=15)


def load_shanghai(root, one_per_patient=True):
    """One summary row per CGM session (Patient Number = <patient>_<session>_<yyyymmdd> = file stem; labs are recorded
    per session). With one_per_patient, keep for each patient the earliest-dated session that has an HbA1c, else the
    earliest-dated session (its labs are the ones recorded with that session)."""
    base = os.path.join(root, SHANGHAI_DIR)
    recs = []
    for kind in ("T1DM", "T2DM"):
        summ = _shanghai_summary(os.path.join(base, f"Shanghai_{kind}_Summary.xlsx"))
        files = {os.path.splitext(os.path.basename(f))[0]: f
                 for f in glob.glob(os.path.join(base, f"Shanghai_{kind}", "*.xls*")) if not os.path.basename(f).startswith("~$")}
        summ = summ[summ.session.isin(files)].copy()
        if one_per_patient:
            summ["has_a1c"] = summ.hba1c.notna()
            summ = (summ.sort_values(["patient", "has_a1c", "date"], ascending=[True, False, True])
                    .groupby("patient", as_index=False).head(1))
        for _, r in summ.sort_values("session").iterrows():
            micro, macro = r.micro, r.macro
            recs.append(dict(cohort=f"Shanghai {kind}", pid=r.session, patient=r.patient, glucose=_shanghai_cgm(files[r.session]),
                             hba1c=r.hba1c, sex=r.sex, age=r.age, bmi=r.bmi, fasting_glucose=r.fasting_glucose,
                             duration=r.duration, dm_type=r.dm_type, agents=r.agents,
                             micro_any=int(micro not in ("none", "nan", "")), macro_any=int(macro not in ("none", "nan", "")),
                             micro=micro, macro=macro, n_sessions=int(sum(k.startswith(r.patient + "_") for k in files))))
    return recs


# ---------------------------------------------------------------- Colas et al. 2019 (non-diabetic hypertensive, 5-min, 1-2 days)
def _colas_series(path):
    d = pd.read_csv(path)
    tod = pd.to_timedelta(d["hora"].astype(str).str.strip())
    # only the time of day is given: start a new (pseudo) day whenever the clock goes backwards
    day = (tod.diff() < pd.Timedelta(0)).cumsum()
    t = pd.Timestamp("2000-01-01") + pd.to_timedelta(day, unit="D") + tod
    v = pd.to_numeric(d["glucemia"], errors="coerce").to_numpy(float)
    return regularize(pd.Series(v, index=t.values), interval_min=5)


def load_colas(root):
    c = pd.read_csv(os.path.join(root, "clinical_data.txt"), sep=r"\s+")
    recs = []
    # readme: case files are "in the same order as in clinical_data.txt". The row labels skip "79" (labels run 1-78,
    # 80-209 for 208 rows), so the mapping is by POSITION: case k <-> k-th row. Checked: CGM mean vs fasting glucose
    # Spearman 0.27 (positional) vs -0.04 (label-based) on the rows after the skipped label.
    for k, (lab, r) in enumerate(c.iterrows(), start=1):
        f = os.path.join(root, f"case  {k}.csv")
        recs.append(dict(cohort="Colas", pid=str(k), clinical_row_label=str(lab), glucose=_colas_series(f), hba1c=float(r.HbA1c),
                         sex="F" if r.gender == 1 else "M", age=float(r.age), bmi=float(r.BMI),
                         fasting_glucose=float(r.glycaemia), followup_days=float(r["follow.up"]),
                         t2dm=int(bool(r.T2DM))))
    return recs


def load_all(project_root):
    return (load_cgmacros(os.path.join(project_root, "cgmacros", "CGMacros"))
            + load_bigideas(os.path.join(project_root, "data_ext", "bigideas"))
            + load_shanghai(os.path.join(project_root, "data_ext", "shanghai"))
            + load_colas(os.path.join(project_root, "data_ext", "colas", "S1")))
