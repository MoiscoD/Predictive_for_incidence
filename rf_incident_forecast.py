"""
Random Forest forecast WITHOUT using the dates respondents typed in.

What it answers
  A) SECTOR:     which incident types are likely in each sector over the next 3 and 6 months
  B) INDIVIDUAL: which incident types each respondent is likely to meet, and how likely

How time enters (no dates needed)
  The survey asks "How often do cyber incidents occur to you?" (Rarely / Occasionally / Frequently).
  We turn that answer into a rate per month (see RATE_PER_YEAR below), then
      expected incidents in h months = rate per month x h

Run:  python rf_forecast_no_dates.py
Needs: pip install pandas numpy scikit-learn matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict

CSV_FILE = "Worksheet.csv"
HORIZONS = [3, 6]
SEED = 42

# ASSUMPTION YOU MUST STATE IN YOUR THESIS: what each answer means in incidents per year.
# Change these numbers and every forecast rescales.  "Never" is merged into "Rarely"
# because those two respondents did report an incident.
RATE_PER_YEAR = {"Rarely": 1, "Occasionally": 3, "Frequently": 12}

# ---------------------------------------------------------------- STEP 1: load
df = pd.read_csv(CSV_FILE)
def find(text):
    return [c for c in df.columns if text.lower() in c.lower()][0]

col_sector = find("Industry/Sector"); col_had = find("Have you experienced")
col_freq   = find("How often do cyber incident occur to you")
col_type   = find("What type of cyber incident"); col_gender = find("Gender")
col_occ    = find("6. Occupation");  col_org = find("7. Organization")
col_other  = "Unnamed: 16"           # free-text "other" type

df["person_id"] = ["P" + str(i + 1).zfill(3) for i in range(len(df))]
top = ["Education", "Information Technology", "Healthcare"]
df["sector"] = df[col_sector].where(df[col_sector].isin(top), "Other sectors")
df["victim"] = (df[col_had] == "Yes").astype(int)

# -------------------------------------------- STEP 2: group the incident types
def group_type(row):
    t = (str(row[col_type]) + " " + str(row[col_other])).lower()
    if row["victim"] == 0:                                   return np.nan
    if any(k in t for k in ["malware", "ransom", "ddos", "denial", "defac"]):
        return "Malware / ransomware / DDoS"
    if any(k in t for k in ["phish", "social engineering", "credential harvest"]):
        return "Phishing / social engineering"
    if any(k in t for k in ["financial", "banking", "shopping", "scam", "identity", "bank"]):
        return "Financial fraud / scams"
    if any(k in t for k in ["account", "hijack", "password", "credential", "unauthorized", "hack", "compromise", "taken down"]):
        return "Account takeover"
    return "Other"                                            # e.g. lost device (too rare to model)
df["type_group"] = df.apply(group_type, axis=1)

freq = df[col_freq].replace({"Never": "Rarely"})
df["rate_month"] = freq.map(RATE_PER_YEAR) / 12              # NaN for non-victims

# ----------------------------------------------- STEP 3: what the model may see
# Only "who the person is" -- never anything that describes the incident itself (that would be cheating).
X = pd.get_dummies(df[["sector", col_gender, col_occ, col_org]].astype(str))

def forest_clf(): return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
def forest_reg(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1)

# Each victim is scored by a forest that never saw them ("out-of-fold"), so results are not flattered.
# ---- Model 1: how likely is a person with this profile to be hit at all?
y1 = df["victim"]
cv1 = StratifiedKFold(5, shuffle=True, random_state=SEED)
df["p_victim"] = cross_val_predict(forest_clf(), X, y1, cv=cv1, method="predict_proba")[:, 1]
acc1 = ((df["p_victim"] > 0.5).astype(int) == y1).mean()
print(f"[Model 1: hit or not]  accuracy {acc1:.2f} | always-guess-majority baseline {max(y1.mean(), 1 - y1.mean()):.2f}")

# ---- Model 2: if hit, which type?
typed = df[df["type_group"].notna() & (df["type_group"] != "Other")]
y2 = typed["type_group"]
k = max(2, min(5, y2.value_counts().min()))
oof = cross_val_predict(forest_clf(), X.loc[typed.index], y2,
                        cv=StratifiedKFold(k, shuffle=True, random_state=SEED), method="predict_proba")
m2 = forest_clf().fit(X.loc[typed.index], y2)
classes = list(m2.classes_)
P_type = pd.DataFrame(m2.predict_proba(X), index=df.index, columns=classes)   # for people not in training
P_type.loc[typed.index] = oof
acc2 = (np.array(classes)[oof.argmax(axis=1)] == y2.values).mean()
print(f"[Model 2: which type]  accuracy {acc2:.2f} | majority baseline {y2.value_counts(normalize=True).max():.2f}"
      f" | types: {dict(y2.value_counts())}")

# ---- Model 3: if hit, how often (per month)?
vic = df[df["victim"] == 1]
oof_rate = cross_val_predict(forest_reg(), X.loc[vic.index], vic["rate_month"],
                             cv=KFold(5, shuffle=True, random_state=SEED))
m3 = forest_reg().fit(X.loc[vic.index], vic["rate_month"])
df["rate_hat"] = m3.predict(X)
df.loc[vic.index, "rate_hat"] = oof_rate

# --------------------------------------------- STEP 4: combine into forecasts
# incidents per month for person i  =  P(hit) x rate if hit
df["lambda_month"] = df["p_victim"] * df["rate_hat"]

# ---- B) INDIVIDUAL
ind = df[["person_id", "sector"]].copy()
for h in HORIZONS:
    ind[f"prob_any_incident_{h}m"] = (1 - np.exp(-df["lambda_month"] * h)).round(3)
best = P_type.idxmax(axis=1)
ind["most_likely_type"] = best
for h in HORIZONS:
    lam_type = P_type.mul(df["lambda_month"], axis=0) * h
    ind[f"prob_that_type_{h}m"] = [round(1 - np.exp(-lam_type.loc[i, best[i]]), 3) for i in df.index]
for h in HORIZONS:
    for t in classes:
        ind[f"{t} | {h}m"] = (1 - np.exp(-P_type[t] * df["lambda_month"] * h)).round(3)
ind.to_csv("individual_forecast.csv", index=False)

# ---- A) SECTOR: expected incidents per 100 people
rows = []
for s, g in df.groupby("sector"):
    for h in HORIZONS:
        for t in classes:
            rows.append({"sector": s, "horizon_months": h, "incident_type": t,
                         "expected_per_100_people": round(100 * (P_type.loc[g.index, t] * g["lambda_month"] * h).mean(), 1)})
sector = pd.DataFrame(rows)
sector.to_csv("sector_forecast_by_type.csv", index=False)

print("\nSECTOR FORECAST (expected incidents per 100 people)")
print(sector.pivot_table(index=["sector", "incident_type"], columns="horizon_months",
                         values="expected_per_100_people").to_string())
print("\nINDIVIDUAL: first 8 rows")
print(ind[["person_id", "sector", "prob_any_incident_3m", "prob_any_incident_6m",
           "most_likely_type", "prob_that_type_3m", "prob_that_type_6m"]].head(8).to_string(index=False))

# ------------------------------------------------------------------ STEP 5: chart
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, h in zip(axes, HORIZONS):
    sector[sector.horizon_months == h].pivot(index="sector", columns="incident_type",
                                             values="expected_per_100_people").plot(kind="bar", ax=ax, rot=20)
    ax.set_title(f"Next {h} months"); ax.set_xlabel(""); ax.set_ylabel("Expected incidents per 100 people")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("sector_forecast_by_type.png", dpi=150)
print("\nSaved: sector_forecast_by_type.csv, individual_forecast.csv, sector_forecast_by_type.png")

# ============================================================ STEP 6: PERCENTAGE SHARES
# (a) What respondents actually reported (simple counts -> percentages; no model involved)
rep = df[df["type_group"].isin(classes)]
overall_obs = (rep["type_group"].value_counts(normalize=True) * 100).round(1)
by_sector_obs = (pd.crosstab(rep["sector"], rep["type_group"], normalize="index") * 100).round(1)
n_by_sector = rep["sector"].value_counts()

# (b) What the forecast expects: each type's share of all expected incidents
lam_type = P_type.mul(df["lambda_month"], axis=0)
overall_fc = (lam_type.sum() / lam_type.sum().sum() * 100).round(1)
by_sector_fc = lam_type.groupby(df["sector"]).sum()
by_sector_fc = (by_sector_fc.div(by_sector_fc.sum(axis=1), axis=0) * 100).round(1)

print("\nOVERALL SHARE OF ATTACK TYPES (%)")
print(pd.DataFrame({"reported_%": overall_obs, "forecast_%": overall_fc}).sort_values("reported_%", ascending=False).to_string())
print("\nREPORTED SHARE BY SECTOR (%)  [n per sector:", dict(n_by_sector), "]")
print(by_sector_obs.to_string())
print("\nFORECAST SHARE BY SECTOR (%)")
print(by_sector_fc.to_string())

by_sector_obs.assign(n=n_by_sector).to_csv("attack_share_by_sector_reported.csv")
by_sector_fc.to_csv("attack_share_by_sector_forecast.csv")
pd.DataFrame({"reported_%": overall_obs, "forecast_%": overall_fc}).to_csv("attack_share_overall.csv")

fig, ax = plt.subplots(figsize=(9, 4.8))
by_sector_obs.plot(kind="barh", stacked=True, ax=ax)
ax.set_xlabel("% of reported incidents"); ax.set_ylabel("")
ax.set_title("Attack-type share by sector (reported incidents)")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout(); plt.savefig("attack_share_by_sector.png", dpi=150)
print("Saved: attack_share_*.csv, attack_share_by_sector.png")