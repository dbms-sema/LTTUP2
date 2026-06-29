"""
End-to-End Longitudinal LTFU Prediction
=======================================
Architecture : BEHRT + Med-BERT  (Transformer encoder over visit-level
               clinical token sequences with positional/age/segment
               embeddings) + Survival Head trained with Cox partial
               likelihood.
Outputs      : (a) static risk for each patient given history,
               (b) DYNAMIC survival probability S(t* | history) where
                   t* is a user-supplied next-visit date.

Dataset      : 2016_2019_ClinicDataWide.xlsx
               One row per patient, V1..V52 visit blocks (~170 fields each).
"""

import os, warnings, joblib, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute   import SimpleImputer
from sklearn.metrics  import (roc_auc_score, accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              classification_report)

import torch
import torch.nn as nn
from   torch.utils.data import Dataset, DataLoader

from lifelines.utils import concordance_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================================================================
# 1.  LOAD WIDE DATA AND RESHAPE TO LONG (one row per visit)
# =============================================================================
DATA_PATH = "2016_2019_ClinicDataWide.xlsx"

# Identifier / patient-level cols we keep
ID_COLS = [
    "IDCNO", "Gender", "DOB", "ARTStartDate", "RegistrationDate", "Age",
    "District", "Education_level", "Occupation", "Marital_status",
    "n_visits", "first_visit", "last_visit", "followup_days",
]

# Per-visit fields we want from each V{i}_ block. Keep a tight, clinically
# meaningful subset to control width — extending this list is fine but each
# new column has to exist in V1..V52.
VISIT_FIELDS = [
    "VisitDate", "ARTAdherenceScore", "BMI", "Weight",
    "Systolic BP", "Diastolic BP", "KarnofskyScore", "CDCScore",
    "Last_CD4", "Lastest_VL", "Lastest_VL_Date",
    "WHOStage", "Regimen", "Curr_Regimen", "Funding Source",
    "VisitType", "Appointment Type",
    "Return Appointment Date", "DaysToReturn",
    "FollowUpStatus", "FollowUpStatusDate",
    "TB", "WeightLoss", "PersistentFever", "NightSweats", "Coughing",
    "OI_TB", "OI_OralCandidiasis", "OI_PCP",
    "NCD_Hypertension", "NCD_DiabetesMellitus", "NCD_MentalHealth",
    "Disclosure_status", "Disability",
    "Prophylaxis", "Anti_Hypertension", "Anti_Diabetes",
    "DSDMType",
]

MAX_VISITS = 52

print("Reading wide Excel ... (this is a 41 MB file with ~8,800 cols)")
visit_cols = [f"V{i}_{f}" for i in range(1, MAX_VISITS+1) for f in VISIT_FIELDS]
df_wide = pd.read_excel(DATA_PATH, usecols=ID_COLS + visit_cols)
print("Wide shape:", df_wide.shape)

# Reshape wide -> long
records = []
for _, row in df_wide.iterrows():
    n = int(row["n_visits"]) if not pd.isna(row["n_visits"]) else 0
    base = {c: row[c] for c in ID_COLS}
    for i in range(1, n+1):
        v = {"VisitIndex": i}
        v.update(base)
        for f in VISIT_FIELDS:
            v[f] = row.get(f"V{i}_{f}", np.nan)
        records.append(v)

df = pd.DataFrame.from_records(records)
print("Long shape:", df.shape)

# Date parsing
DATE_COLS = ["DOB", "ARTStartDate", "RegistrationDate",
             "first_visit", "last_visit",
             "VisitDate", "Lastest_VL_Date", "Return Appointment Date",
             "FollowUpStatusDate"]
for c in DATE_COLS:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# =============================================================================
# 2.  BUILD SURVIVAL TARGETS (event, time_to_event) AT THE PATIENT LEVEL
# =============================================================================
# Per patient, take the LAST non-null FollowUpStatus across visits.
def last_status(g):
    s = g.dropna(subset=["FollowUpStatus"])
    if len(s) == 0: return pd.Series({"last_status": None, "last_status_date": pd.NaT})
    s = s.sort_values("VisitIndex").iloc[-1]
    return pd.Series({"last_status": s["FollowUpStatus"],
                      "last_status_date": s["FollowUpStatusDate"]})

outcome = df.groupby("IDCNO").apply(last_status).reset_index()
df = df.merge(outcome, on="IDCNO", how="left")

# Event: LTFU (cover both "LTFU" and "LTFU90" labels)
df["event"] = df["last_status"].astype(str).str.upper().str.startswith("LTFU").astype(int)

# Time-to-event: prefer (last_status_date - RegistrationDate); fall back to followup_days
df["time_to_event"] = (df["last_status_date"] - df["RegistrationDate"]).dt.days
df["time_to_event"] = df["time_to_event"].fillna(df["followup_days"])

df = df.dropna(subset=["time_to_event", "event"])
df = df[df["time_to_event"] > 0]

print("\nLTFU event rate:", df.drop_duplicates('IDCNO')['event'].mean().round(4))
print("Patients after cleaning:", df['IDCNO'].nunique())

# =============================================================================
# 3.  FEATURE ENGINEERING (BEHRT/Med-BERT style)
# =============================================================================
df = df.sort_values(["IDCNO", "VisitIndex"])

# --- Age at visit ---
df["AgeYears"] = (df["VisitDate"] - df["DOB"]).dt.days / 365.25
df["AgeYears"] = df["AgeYears"].fillna(df["Age"])

# --- Viral suppression flag (clinically << 1000 cp/mL = suppressed) ---
df["viral_suppressed"] = (
    pd.to_numeric(df["Lastest_VL"], errors="coerce") < 1000
).astype(int)

# --- Time deltas (only PAST information w.r.t. the current visit) ---
df["prev_visit"]     = df.groupby("IDCNO")["VisitDate"].shift(1)
df["visit_gap_days"] = (df["VisitDate"] - df["prev_visit"]).dt.days
df["prev_return"]    = df.groupby("IDCNO")["Return Appointment Date"].shift(1)
df["days_late"]      = (df["VisitDate"] - df["prev_return"]).dt.days
df["DaysToReturn_prev"] = df.groupby("IDCNO")["DaysToReturn"].shift(1)

# --- DROP COLUMNS THAT LEAK THE OUTCOME ---
# Anything observed at/after the LTFU determination must not be a predictor.
LEAKAGE_COLS = ["FollowUpStatus", "FollowUpStatusDate",
                "last_status", "last_status_date"]
df_model = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

# --- Final feature set (numeric + categorical) ---
FEATURE_COLUMNS = [
    "AgeYears", "Gender", "WHOStage", "ARTAdherenceScore",
    "BMI", "Weight", "Systolic BP", "Diastolic BP",
    "KarnofskyScore", "Last_CD4", "Lastest_VL", "viral_suppressed",
    "visit_gap_days", "days_late", "DaysToReturn_prev",
    "TB", "WeightLoss", "PersistentFever", "NightSweats", "Coughing",
    "OI_TB", "OI_OralCandidiasis", "OI_PCP",
    "NCD_Hypertension", "NCD_DiabetesMellitus", "NCD_MentalHealth",
    "Regimen", "Curr_Regimen", "Funding Source", "VisitType",
    "Appointment Type", "Disclosure_status", "Disability", "DSDMType",
    "Prophylaxis", "Anti_Hypertension", "Anti_Diabetes",
    "Occupation", "Education_level", "Marital_status",
]
FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c in df_model.columns]
print(f"\nUsing {len(FEATURE_COLUMNS)} features per visit:")
print(FEATURE_COLUMNS)

# --- Encode categoricals + scale numerics ---
cat_cols = df_model[FEATURE_COLUMNS].select_dtypes(include="object").columns.tolist()
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df_model[c] = df_model[c].astype(str)
    df_model[c] = le.fit_transform(df_model[c])
    encoders[c] = le

imputer = SimpleImputer(strategy="median")
df_model[FEATURE_COLUMNS] = imputer.fit_transform(df_model[FEATURE_COLUMNS])

scaler = StandardScaler()
df_model[FEATURE_COLUMNS] = scaler.fit_transform(df_model[FEATURE_COLUMNS])

# Age bucket for BEHRT-style age embedding (0..MAX_AGE_BUCKETS-1)
MAX_AGE_BUCKETS = 12  # ~ every ~10y bucket of patient age at visit
df_model["AgeBucket"] = pd.cut(
    (df["VisitDate"] - df["DOB"]).dt.days / 365.25,
    bins=np.linspace(0, 120, MAX_AGE_BUCKETS+1),
    labels=False, include_lowest=True,
).fillna(0).astype(int)

# =============================================================================
# 4.  PATIENT-LEVEL SPLIT  (no leakage between train/val/test)
# =============================================================================
patient_df = df_model.drop_duplicates("IDCNO")[["IDCNO", "event", "time_to_event"]]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(df_model, groups=df_model["IDCNO"]))
train_df = df_model.iloc[train_idx]
test_df  = df_model.iloc[test_idx]

print(f"\nTrain patients: {train_df['IDCNO'].nunique()}, "
      f"Test patients: {test_df['IDCNO'].nunique()}")

# =============================================================================
# 5.  BUILD PER-PATIENT SEQUENCES (right-padded, attention-masked)
# =============================================================================
MAX_SEQ_LENGTH = 32   # keep last 32 visits per patient

def create_sequences(frame):
    X, A, E, T, M = [], [], [], [], []
    for pid, g in frame.groupby("IDCNO"):
        g = g.sort_values("VisitDate")
        feats = g[FEATURE_COLUMNS].values.astype(np.float32)
        ages  = g["AgeBucket"].values.astype(np.int64)
        n = len(feats)

        # truncate to last MAX_SEQ_LENGTH visits
        if n > MAX_SEQ_LENGTH:
            feats, ages = feats[-MAX_SEQ_LENGTH:], ages[-MAX_SEQ_LENGTH:]
            n = MAX_SEQ_LENGTH

        # right-pad
        pad_n = MAX_SEQ_LENGTH - n
        if pad_n > 0:
            feats = np.vstack([feats, np.zeros((pad_n, len(FEATURE_COLUMNS)),
                                               dtype=np.float32)])
            ages  = np.concatenate([ages, np.zeros(pad_n, dtype=np.int64)])
        mask = np.array([1]*n + [0]*pad_n, dtype=np.float32)

        X.append(feats); A.append(ages); M.append(mask)
        E.append(g["event"].iloc[-1])
        T.append(g["time_to_event"].iloc[-1])
    return (np.stack(X), np.stack(A), np.stack(M),
            np.array(E, dtype=np.float32), np.array(T, dtype=np.float32))

X_train, A_train, M_train, y_train, t_train = create_sequences(train_df)
X_test,  A_test,  M_test,  y_test,  t_test  = create_sequences(test_df)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

# =============================================================================
# 6.  PYTORCH DATASET / DATALOADER
# =============================================================================
class LongitudinalDataset(Dataset):
    def __init__(self, X, A, M, y, t):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.long)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
    def __len__(self):       return len(self.X)
    def __getitem__(self, i): return self.X[i], self.A[i], self.M[i], self.y[i], self.t[i]

train_loader = DataLoader(LongitudinalDataset(X_train, A_train, M_train, y_train, t_train),
                          batch_size=64, shuffle=True)
test_loader  = DataLoader(LongitudinalDataset(X_test,  A_test,  M_test,  y_test,  t_test),
                          batch_size=64, shuffle=False)

# =============================================================================
# 7.  MODEL: BEHRT + Med-BERT inspired transformer + survival head
# =============================================================================
class BEHRTMedBERTSurvival(nn.Module):
    """
    Mirrors the design ideas of BEHRT (Li et al., Sci. Rep. 2020) and
    Med-BERT (Rasmy et al., npj Digital Medicine 2021):
      * value projection of raw visit features  ->  hidden_dim
      * learned VISIT positional embedding      (Med-BERT)
      * learned AGE-bucket embedding            (BEHRT)
      * Transformer encoder with padding mask   (BEHRT)
      * attention pooling over time             (BEHRT)
      * survival head -> single risk score (Cox)
    """
    def __init__(self, input_dim, n_age_buckets=MAX_AGE_BUCKETS,
                 max_visits=MAX_SEQ_LENGTH, hidden_dim=128,
                 num_heads=4, num_layers=4, dropout=0.2):
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(max_visits, hidden_dim)
        self.age_emb = nn.Embedding(n_age_buckets, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # attention pooling
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))

        # survival head -> scalar log-risk (Cox)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, age, mask):
        # x:(B,T,F)  age:(B,T)  mask:(B,T)   1 = real visit, 0 = pad
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.value_proj(x) + self.pos_emb(pos) + self.age_emb(age)

        # TransformerEncoder expects True for positions to MASK OUT
        key_padding_mask = (mask == 0)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        # attention pooling, masking padded steps
        a = self.attn(h).squeeze(-1)               # (B,T)
        a = a.masked_fill(key_padding_mask, -1e4)
        a = torch.softmax(a, dim=1).unsqueeze(-1)  # (B,T,1)
        pooled = (h * a).sum(dim=1)                # (B,H)

        risk = self.head(pooled).squeeze(-1)       # (B,)
        return risk

# =============================================================================
# 8.  Cox partial-likelihood loss (numerically stable)
# =============================================================================
class CoxLoss(nn.Module):
    def forward(self, risk, time, event):
        order  = torch.argsort(time, descending=True)
        risk   = torch.clamp(risk[order], min=-15, max=15)
        event  = event[order]
        log_cs = torch.logcumsumexp(risk, dim=0)
        ll     = risk - log_cs
        denom  = event.sum().clamp_min(1.0)
        return -(ll * event).sum() / denom

# =============================================================================
# 9.  TRAIN
# =============================================================================
model     = BEHRTMedBERTSurvival(input_dim=len(FEATURE_COLUMNS)).to(device)
print(model)

criterion = CoxLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

EPOCHS = 25
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for X_b, A_b, M_b, y_b, t_b in train_loader:
        X_b, A_b, M_b, y_b, t_b = [v.to(device) for v in (X_b, A_b, M_b, y_b, t_b)]
        optimizer.zero_grad()
        risk = model(X_b, A_b, M_b)
        loss = criterion(risk, t_b, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  loss={total/len(train_loader):.4f}")

# =============================================================================
# 10.  EVALUATE
# =============================================================================
model.eval()
risks, evts, times = [], [], []
with torch.no_grad():
    for X_b, A_b, M_b, y_b, t_b in test_loader:
        X_b, A_b, M_b = X_b.to(device), A_b.to(device), M_b.to(device)
        r = model(X_b, A_b, M_b).cpu().numpy()
        risks.extend(r); evts.extend(y_b.numpy()); times.extend(t_b.numpy())
risks, evts, times = map(np.array, (risks, evts, times))

c_idx = concordance_index(times, -risks, evts)
print(f"\nC-index on test set: {c_idx:.4f}")

probs = 1.0 / (1.0 + np.exp(-risks))
preds = (probs >= 0.5).astype(int)
print("AUROC :", round(roc_auc_score(evts, probs), 4))
print("Acc   :", round(accuracy_score(evts, preds), 4))
print("Prec  :", round(precision_score(evts, preds, zero_division=0), 4))
print("Recall:", round(recall_score(evts, preds, zero_division=0), 4))
print("F1    :", round(f1_score(evts, preds, zero_division=0), 4))
print(classification_report(evts, preds, zero_division=0))

# =============================================================================
# 11.  ESTIMATE BASELINE CUMULATIVE HAZARD (Breslow) FOR SURVIVAL CURVES
# =============================================================================
# We need this to translate model risk -> probability of LTFU by time t.
# H0(t) = sum over event_times t_i<=t of d_i / sum_{j in risk set} exp(risk_j)
def compute_breslow_baseline(times_tr, evts_tr, risks_tr):
    order = np.argsort(times_tr)
    t_s, e_s, r_s = times_tr[order], evts_tr[order], risks_tr[order]
    exp_r = np.exp(np.clip(r_s, -15, 15))
    H0, ts = [], []
    cum = 0.0
    # risk set sum is over all subjects with time >= t_i
    # iterate from latest to earliest to maintain running sum efficiently
    rev_cumsum_expr = np.cumsum(exp_r[::-1])[::-1]
    for i in range(len(t_s)):
        if e_s[i] == 1:
            denom = rev_cumsum_expr[i]
            if denom > 0:
                cum += 1.0 / denom
            H0.append(cum); ts.append(t_s[i])
    return np.array(ts), np.array(H0)

# Compute training risks
model.eval()
tr_risks = []
tr_loader = DataLoader(LongitudinalDataset(X_train, A_train, M_train, y_train, t_train),
                       batch_size=128, shuffle=False)
with torch.no_grad():
    for X_b, A_b, M_b, _, _ in tr_loader:
        r = model(X_b.to(device), A_b.to(device), M_b.to(device)).cpu().numpy()
        tr_risks.extend(r)
tr_risks = np.array(tr_risks)

baseline_times, H0 = compute_breslow_baseline(t_train, y_train, tr_risks)

# =============================================================================
# 12.  DYNAMIC SURVIVAL PREDICTION
#      "Given a patient's history and a candidate next visit date, what is
#       the probability that they will be LTFU by that date?"
# =============================================================================
def dynamic_survival_prediction(patient_seq_X, patient_seq_A, patient_seq_M,
                                t_query_days):
    """
    patient_seq_X : (T, F) np.ndarray  – features for one patient (already
                                          scaled & padded)
    patient_seq_A : (T,)   np.ndarray  – age buckets
    patient_seq_M : (T,)   np.ndarray  – attention mask
    t_query_days  : int                – number of days from RegistrationDate
                                         to the proposed next visit
    Returns        : dict with risk_score, log-risk, S(t*), and P(LTFU by t*)
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(patient_seq_X[None], dtype=torch.float32, device=device)
        a = torch.tensor(patient_seq_A[None], dtype=torch.long,    device=device)
        m = torch.tensor(patient_seq_M[None], dtype=torch.float32, device=device)
        risk = model(x, a, m).cpu().item()

    exp_r = math.exp(max(min(risk, 15), -15))
    # locate H0 at t_query_days
    if t_query_days <= baseline_times[0]:
        H_t = 0.0
    elif t_query_days >= baseline_times[-1]:
        H_t = H0[-1]
    else:
        H_t = float(np.interp(t_query_days, baseline_times, H0))

    S_t = math.exp(-H_t * exp_r)
    return {"risk_score": risk,
            "exp_risk": exp_r,
            "baseline_hazard_at_t": H_t,
            "survival_prob_at_t": S_t,
            "ltfu_prob_at_t": 1.0 - S_t}

# Demo: pick the first test patient, ask "what's LTFU prob by day 365?"
demo = dynamic_survival_prediction(X_test[0], A_test[0], M_test[0], t_query_days=365)
print("\nDynamic-prediction demo (patient 0, horizon = 365 days):")
for k, v in demo.items():
    print(f"  {k:>25s} : {v:.4f}")

# Helper: convert a *date* to days-from-registration for a given patient
def predict_ltfu_by_next_visit_date(patient_row_idx, next_visit_date_str,
                                    reference_dataframe):
    """
    Convenience wrapper: pass the patient's test index and a calendar
    next-visit date string ('YYYY-MM-DD'). The function computes days
    from this patient's RegistrationDate to that date and calls the
    dynamic predictor.
    """
    pid       = test_df["IDCNO"].drop_duplicates().iloc[patient_row_idx]
    reg_date  = reference_dataframe.loc[reference_dataframe["IDCNO"] == pid,
                                        "RegistrationDate"].iloc[0]
    next_date = pd.to_datetime(next_visit_date_str)
    t_days    = (next_date - reg_date).days
    if t_days <= 0:
        raise ValueError("Next visit date must be after RegistrationDate.")
    return dynamic_survival_prediction(
        X_test[patient_row_idx], A_test[patient_row_idx],
        M_test[patient_row_idx], t_query_days=t_days)

# =============================================================================
# 13.  PERSIST EVERYTHING
# =============================================================================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/behrt_medbert_survival.pt")
joblib.dump(scaler,          "models/scaler.pkl")
joblib.dump(imputer,         "models/imputer.pkl")
joblib.dump(encoders,        "models/encoders.pkl")
joblib.dump(FEATURE_COLUMNS, "models/feature_columns.pkl")
np.savez("models/baseline_hazard.npz",
         baseline_times=baseline_times, H0=H0)

print("\nSaved:")
print("  models/behrt_medbert_survival.pt")
print("  models/scaler.pkl  models/imputer.pkl  models/encoders.pkl")
print("  models/baseline_hazard.npz")
print("\nDone.")
