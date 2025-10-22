import pandas as pd
import numpy as np

#load the data you can add location to your directory here

netflix_df = pd.read_csv("/Users/pujithaattuluri/Desktop/UNCC fall 2025/Strategic Business Analytics/Datasets/netflix_customer_churn.csv")

display(netflix_df.head())

print("\n Info:")
print(netflix_df.info())


print("\n To check Missing Values if any in the data:")
print(netflix_df.isnull().sum())

netflix_df[['age','watch_hours','avg_watch_time_per_day','monthly_fee']].describe()

# Audit: shape, dtypes, nulls, duplicates, target balance
audit = {
    "shape": {"rows": netflix_df.shape[0], "cols": netflix_df.shape[1]},
    "dtypes": netflix_df.dtypes.astype(str).to_dict(),
    "null_counts": netflix_df.isnull().sum().to_dict(),
    "duplicate_rows": int(netflix_df.duplicated().sum()),
    "target_balance_churned": netflix_df["churned"].value_counts().to_dict() if "churned" in netflix_df.columns else "N/A",
}
audit

# created a new column for cust_id as the original one is big and is an UUID form and not readable

netflix_df["customer_id_clean"] = netflix_df["customer_id"].apply(lambda x: "CUST_" + x.split("-")[0].upper())



import pandas as pd
import numpy as np

raw_path = "/Users/pujithaattuluri/Desktop/UNCC fall 2025/Strategic Business Analytics/Datasets/netflix_customer_churn.csv"
out_data_path = "/Users/pujithaattuluri/Desktop/UNCC fall 2025/Strategic Business Analytics/Datasets/netflix_churn_cleaned_step2_with_custid.csv"
mapping_path = "/Users/pujithaattuluri/Desktop/UNCC fall 2025/Strategic Business Analytics/Datasets/customer_id_mapping.csv"

df = pd.read_csv(raw_path)

# --- Step 2 cleaning (same as before, but we keep original customer_id) ---
cat_cols = ['gender','subscription_type','region','device','payment_method','favorite_genre']
for c in [c for c in cat_cols if c in df.columns]:
    df[c] = (df[c].astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.title())

if "avg_watch_time_per_day" in df.columns:
    df["avg_watch_time_per_day"] = df["avg_watch_time_per_day"].clip(0, 24)
if "watch_hours" in df.columns:
    q99 = df["watch_hours"].quantile(0.99)
    df["watch_hours"] = df["watch_hours"].clip(0, q99)
if "number_of_profiles" in df.columns:
    df["number_of_profiles"] = df["number_of_profiles"].clip(1, 5).astype(int)
if "last_login_days" in df.columns:
    df["last_login_days"] = df["last_login_days"].clip(lower=0)
if "monthly_fee" in df.columns:
    df["monthly_fee"] = df["monthly_fee"].clip(lower=0)

df = df.drop_duplicates()

# shorten UUID to compact clean ID ---
def shorten_uuid_to_id(x: str) -> str:
    x = str(x)
    head = x.split("-")[0] if "-" in x else x[:8]
    head = head[:8].upper() if head else "UNK00000"
    return "CUST_" + head

base_ids = df["customer_id"].apply(shorten_uuid_to_id)

# Ensure uniqueness
dup_counts = base_ids.groupby(base_ids).cumcount()
customer_id_clean = base_ids.where(dup_counts == 0, base_ids + "-" + (dup_counts + 1).astype(str))

# Attach as a new column (keep original)
df.insert(1, "customer_id_clean", customer_id_clean)

# Save outputs
df.to_csv(out_data_path, index=False)
pd.DataFrame({
    "customer_id": df["customer_id"],
    "customer_id_clean": df["customer_id_clean"]
}).to_csv(mapping_path, index=False)

import os
import pandas as pd
import numpy as np

# Modeling & Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report, confusion_matrix
)

# Inference (p-values, CIs, marginal effects) + VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# -------------------------------
# Step 3 — Train/Test split
# -------------------------------
TARGET = "churned"
target_col = "churned"

id_cols = [c for c in ["customer_id", "customer_id_clean"] if c in df.columns]

feature_cols = [c for c in df.columns if c not in id_cols + [target_col]]
X = df[feature_cols].copy()
y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
ids = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)

# Split by dtype
cat_pred = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_pred = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.25, random_state=42, stratify=y
)


OUT_DIR = "/Users/pujithaattuluri/Desktop/UNCC fall 2025/Strategic Business Analytics/Datasets/outputs_churn"
os.makedirs(OUT_DIR, exist_ok=True)

ODDS_PATH = os.path.join(OUT_DIR, "logit_odds_ratios.csv")
VIF_PATH = os.path.join(OUT_DIR, "logit_vif.csv")
MFX_PATH = os.path.join(OUT_DIR, "logit_marginal_effects.csv")
SUMMARY_TXT = os.path.join(OUT_DIR, "logit_summary.txt")
SCORED_PATH = os.path.join(OUT_DIR, "logreg_scored_predictions.csv")


# -------------------------------
# Step 4 — sklearn pipeline (performance) & Logistic Regression
# -------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_pred),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_pred),
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred  = pipe.predict(X_test)
y_prob  = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
roc = roc_auc_score(y_test, y_prob)
cm  = confusion_matrix(y_test, y_pred)

print("=== sklearn Logistic Regression (test) ===")
print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC AUC: {roc:.3f}")
print("Confusion matrix:\n", cm)
print(classification_report(y_test, y_pred))

# Save scored predictions with IDs
scored = ids_test.copy()
scored["actual_churn"]       = y_test.values
scored["predicted_churn"]    = y_pred
scored["churn_probability"]  = y_prob
scored.to_csv(SCORED_PATH, index=False)

# -------------------------------
# Step 5 — statsmodels Logit (inference)
# -------------------------------
# One-hot encode and standardize only original numeric columns for interpretability
X_all = df[feature_cols].copy()

cat_cols = X_all.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = X_all.select_dtypes(include=[np.number, "bool"]).columns.tolist()
for c in num_cols:
    if X_all[c].isna().any():
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
        X_all[c] = X_all[c].fillna(X_all[c].median())

# Categoricals: add explicit "Missing" level
for c in cat_cols:
    X_all[c] = X_all[c].astype("category")
    if X_all[c].isna().any():
        X_all[c] = X_all[c].cat.add_categories(["Missing"]).fillna("Missing")

Xd = pd.get_dummies(X_all, columns=cat_cols, drop_first=True, dtype=float)

Xd = Xd.apply(pd.to_numeric, errors="coerce")
Xd = Xd.replace([np.inf, -np.inf], np.nan)

for c in Xd.columns:
    if Xd[c].isna().any():
        Xd[c] = Xd[c].fillna(Xd[c].median())

# align target
y_all = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_only = [c for c in num_cols if c in Xd.columns]
Xd_std = Xd.copy()
Xd_std[num_only] = scaler.fit_transform(Xd_std[num_only])

Xd_train, Xd_test, y_train, y_test = train_test_split(
    Xd_std, y_all, test_size=0.25, random_state=42, stratify=y_all
)

Xd_train_sm = sm.add_constant(Xd_train, has_constant="add").astype(float)
logit = sm.Logit(y_train, Xd_train_sm)
res   = logit.fit(disp=False)

Xd_te_sm = sm.add_constant(Xd_test, has_constant="add").astype(float)
proba_te = res.predict(Xd_te_sm)
pred_te = (proba_te >= 0.5).astype(int)
acc_sm = (pred_te == y_test).mean()
print("\n=== statsmodels Logit (test) ===")
print(f"Accuarcy: {acc_sm:.3f}")

# Save full textual summary
with open(SUMMARY_TXT, "w") as f:
    f.write(res.summary2().as_text())

# Odds ratios + 95% CI
params = res.params
conf   = res.conf_int()
conf.columns = ["2.5%", "97.5%"]

odds = np.exp(params)
or_ci_low  = np.exp(conf["2.5%"])
or_ci_high = np.exp(conf["97.5%"])

odds_table = pd.DataFrame({
    "feature": params.index,
    "coef": params.values,
    "p_value": res.pvalues.values,
    "odds_ratio": odds.values,
    "or_2.5%": or_ci_low.values,
    "or_97.5%": or_ci_high.values
}).sort_values("p_value")
odds_table.to_csv(ODDS_PATH, index=False)

# VIF (exclude constant)
vif_df = pd.DataFrame({
    "feature": Xd_train.columns,
    "VIF": [variance_inflation_factor(Xd_train.values, i) for i in range(Xd_train.shape[1])]
}).sort_values("VIF", ascending=False)
vif_df.to_csv(VIF_PATH, index=False)

# Marginal effects (dy/dx at overall mean)
mfx = res.get_margeff(at="overall").summary_frame()
mfx.to_csv(MFX_PATH)


print("\n=== statsmodels Logit (test) ===")
print(f"Accuracy: {acc_sm:.3f}")
print(f"Saved summary to: {SUMMARY_TXT}")
print(f"Saved odds ratios to: {ODDS_PATH}")
print(f"Saved VIF to: {VIF_PATH}")
print(f"Saved marginal effects to: {MFX_PATH}")
print(f"Scored predictions (with IDs): {SCORED_PATH}")

# --- FIXED: Drop `monthly_fee`, fit Logistic Regression, and compute VIF ---

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 0) Normalize column names (prevents KeyError from stray spaces/case)
df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"\s+", "_", regex=True)
      .str.lower()
)

# --- Setup & split ---
TARGET = "churned"
id_cols = [c for c in ["customer_id", "customer_id_clean"] if c in df.columns]

# All features except IDs and target
X_full = df.drop(columns=id_cols + [TARGET], errors="ignore")

# Drop the collinear numeric
X_plan = X_full.drop(columns=["monthly_fee"], errors="ignore")
y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

# Identify dtypes from X_plan (not df)
cat_cols = X_plan.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_plan.select_dtypes(include=[np.number, "bool"]).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_plan, y, test_size=0.25, random_state=42, stratify=y
)

# --- Pipeline: scale numeric + one-hot encode categoricals ---
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
], remainder="drop")

pipe = Pipeline([
    ("prep", pre),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

# Fit
pipe.fit(X_train, y_train)

# Predict & metrics
yhat  = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, yhat), 3),
      "| ROC AUC:", round(roc_auc_score(y_test, proba), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, yhat))
print(classification_report(y_test, yhat))

# --- VIF check with statsmodels design matrix (after drop) ---
# Build a numeric design matrix: OHE (drop_first) + scale numeric
Xd = pd.get_dummies(X_plan, columns=cat_cols, drop_first=True, dtype=float)

# Standardize original numeric columns only (keeps VIF interpretable)
scaler = StandardScaler()
num_only = [c for c in num_cols if c in Xd.columns]
Xd[num_only] = scaler.fit_transform(Xd[num_only])

# VIF excludes the constant term
vif_table = pd.DataFrame({
    "feature": Xd.columns,
    "VIF": [variance_inflation_factor(Xd.values, i) for i in range(Xd.shape[1])]
}).sort_values("VIF", ascending=False)

print("\nTop 15 VIFs after dropping `monthly_fee`:")
print(vif_table.head(15))


# ============================================================
# Churn Logistic Regression — Result Sets for Stakeholders
# (drops `monthly_fee` to fix collinearity with plan dummies)
# ============================================================

import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

OUT_DIR = "/Users/pujithaattuluri/Desktop/UNCC fall 2025/Strategic Business Analytics/Datasets/outputs_churn_final"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1) Load ----------
try:
    df  
except NameError:
    df = pd.read_csv("/Users/pujithaattuluri/Desktop/UNCC fall 2025/Strategic Business Analytics/Datasets/netflix_customer_churn.csv")

# Normalize column names (defensive)
df.columns = (df.columns
                .str.strip()
                .str.replace(r"\s+", "_", regex=True)
                .str.lower())

TARGET = "churned"
id_cols = [c for c in ["customer_id","customer_id_clean"] if c in df.columns]

# ---------- 2) Feature matrix (drop collinear monthly_fee) ----------
X_full = df.drop(columns=id_cols + [TARGET], errors="ignore")
if "monthly_fee" in X_full.columns:
    X_full = X_full.drop(columns=["monthly_fee"])

y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

cat_cols = X_full.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = X_full.select_dtypes(include=[np.number,"bool"]).columns.tolist()

# ---------- 3) Train/test + pipeline ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y, test_size=0.25, random_state=42, stratify=y
)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
], remainder="drop")

pipe = Pipeline([
    ("prep", pre),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

pipe.fit(X_tr, y_tr)

# Basic metrics
proba_te = pipe.predict_proba(X_te)[:, 1]
pred_te  = (proba_te >= 0.5).astype(int)
acc  = accuracy_score(y_te, pred_te)
prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred_te, average="binary")
roc  = roc_auc_score(y_te, proba_te)
cm   = confusion_matrix(y_te, pred_te)
print(f"[Test] Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  ROC AUC={roc:.3f}")
print("Confusion matrix:\n", cm)

# ---------- 4) Score the FULL dataset ----------
proba_all = pipe.predict_proba(X_full)[:, 1]

# (a) Best-F1 threshold (optional business tuning)
ths = np.linspace(0.05, 0.95, 91)
f1s = []
for t in ths:
    f1s.append(precision_recall_fscore_support(y, (proba_all >= t).astype(int), average="binary")[2])
best_idx = int(np.nanargmax(f1s))
best_t  = float(ths[best_idx])
print(f"Best-F1 threshold ~ {best_t:.2f}")

# (b) Assign risk tiers (percentile cut)
p90, p70, p50 = np.percentile(proba_all, [90, 70, 50])
def tier(p):
    if p >= p90: return "Tier A: Very High"
    if p >= p70: return "Tier B: High"
    if p >= p50: return "Tier C: Medium"
    return "Tier D: Low"

risk_tier = np.vectorize(tier)(proba_all)

# ---------- 5) RESULT SET #1: Predictions with IDs ----------
pred_df = pd.DataFrame({
    "churn_probability": proba_all,
    "predicted_churn_50": (proba_all >= 0.50).astype(int),
    "predicted_churn_bestF1": (proba_all >= best_t).astype(int),
    "risk_tier": risk_tier,
})
for c in id_cols:
    pred_df[c] = df[c].values

pred_out = os.path.join(OUT_DIR, "predictions_with_ids.csv")
pred_df.to_csv(pred_out, index=False)

# ---------- 6) RESULT SET #2: Decile lift table ----------
tmp = pd.DataFrame({"prob": proba_all, "actual": y})
tmp["decile"] = pd.qcut(tmp["prob"].rank(method="first", ascending=False),
                        10, labels=[f"D{i}" for i in range(1,11)])
overall_rate = tmp["actual"].mean()
decile_table = (tmp.groupby("decile", as_index=False)
                  .agg(n=("actual","size"),
                       churn_rate=("actual","mean"),
                       captured_churn=("actual","sum"))
                  .sort_values("decile"))  # labels already in order
decile_table["lift"] = decile_table["churn_rate"] / overall_rate
decile_out = os.path.join(OUT_DIR, "decile_lift.csv")
decile_table.to_csv(decile_out, index=False)

# ---------- 7) RESULT SET #3: Segment summary (plan/region/device if present) ----------
def segment_summary(col):
    if col not in df.columns: 
        return None
    g = (pd.DataFrame({
            col: df[col],
            "actual": y,
            "prob": proba_all
        }).groupby(col).agg(
            customers=("actual","size"),
            churn_rate=("actual","mean"),
            avg_prob=("prob","mean")
        ).reset_index())
    g["lift_vs_overall"] = g["churn_rate"] / overall_rate
    return g

segments = {}
for col in ["subscription_type","region","device","payment_method","favorite_genre"]:
    seg = segment_summary(col)
    if seg is not None:
        outp = os.path.join(OUT_DIR, f"segment_summary__{col}.csv")
        seg.to_csv(outp, index=False)
        segments[col] = outp

# ---------- 8) RESULT SET #4: Confusion-by-segment (at best-F1 threshold) ----------
def confusion_by_segment(col, threshold=best_t):
    if col not in df.columns:
        return None
    pred = (proba_all >= threshold).astype(int)
    out = []
    for k, idx in df.groupby(col).groups.items():
        yy = y.iloc[idx]
        pp = pred[idx]
        tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0,1]).ravel()
        out.append({
            col: k,
            "support": int(len(idx)),
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "f1": (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        })
    return pd.DataFrame(out).sort_values("support", ascending=False)

conf_tables = {}
for col in ["subscription_type","region","device"]:
    ctab = confusion_by_segment(col)
    if ctab is not None:
        outp = os.path.join(OUT_DIR, f"confusion_by_segment__{col}.csv")
        ctab.to_csv(outp, index=False)
        conf_tables[col] = outp

# ---------- 9) RESULT SET #5: Actionable cohorts ----------
cohorts = {}

# Dormant users (>= 21 days since last login)
if "last_login_days" in df.columns:
    dormant = df.loc[df["last_login_days"] >= 21].copy()
    dormant["prob"] = proba_all[dormant.index]
    outp = os.path.join(OUT_DIR, "cohort__dormant_21plus.csv")
    dormant.sort_values("prob", ascending=False).to_csv(outp, index=False)
    cohorts["dormant_21plus"] = outp

# Low engagement (bottom 30% watch_hours)
if "watch_hours" in df.columns:
    q30 = df["watch_hours"].quantile(0.30)
    low_eng = df.loc[df["watch_hours"] <= q30].copy()
    low_eng["prob"] = proba_all[low_eng.index]
    outp = os.path.join(OUT_DIR, "cohort__low_engagement_bottom30pct.csv")
    low_eng.sort_values("prob", ascending=False).to_csv(outp, index=False)
    cohorts["low_engagement_bottom30pct"] = outp

print("\n Saved result sets to:", os.path.abspath(OUT_DIR))
print("1) Predictions + risk tiers:", pred_out)
print("2) Decile lift:", decile_out)
print("3) Segment summaries:", segments)
print("4) Confusion-by-segment:", conf_tables)
print("5) Actionable cohorts:", cohorts)



