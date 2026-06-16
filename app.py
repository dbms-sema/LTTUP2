"""
LTFU Risk Prediction App  (BEHRT + Med-BERT + Survival Head)
============================================================
Loads artifacts produced by `ltfu_behrt_medbert_survival.py`:
    models/behrt_medbert_survival.pt
    models/scaler.pkl
    models/imputer.pkl
    models/encoders.pkl
    models/baseline_hazard.npz

Run:
    streamlit run app.py
"""

import math
import os
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# =============================================================================
#  CONSTANTS  – must match the training script exactly
# =============================================================================
MAX_SEQ_LENGTH   = 32
MAX_AGE_BUCKETS  = 12

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

# Which features are categorical at the raw-input stage (everything else numeric)
CATEGORICAL_FEATURES_DEFAULT = {
    "Gender", "WHOStage", "Regimen", "Curr_Regimen", "Funding Source",
    "VisitType", "Appointment Type", "Disclosure_status", "Disability",
    "DSDMType", "Occupation", "Education_level", "Marital_status",
    "TB", "WeightLoss", "PersistentFever", "NightSweats", "Coughing",
    "OI_TB", "OI_OralCandidiasis", "OI_PCP",
    "NCD_Hypertension", "NCD_DiabetesMellitus", "NCD_MentalHealth",
    "Prophylaxis", "Anti_Hypertension", "Anti_Diabetes",
}

MODEL_DIR = "models"

# =============================================================================
#  MODEL  – identical class definition to the training script
# =============================================================================
class BEHRTMedBERTSurvival(nn.Module):
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
        self.pos_emb = nn.Embedding(max_visits,    hidden_dim)
        self.age_emb = nn.Embedding(n_age_buckets, hidden_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x, age, mask):
        B, T, _ = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h    = self.value_proj(x) + self.pos_emb(pos) + self.age_emb(age)
        kpm  = (mask == 0)
        h    = self.encoder(h, src_key_padding_mask=kpm)
        a    = self.attn(h).squeeze(-1).masked_fill(kpm, -1e4)
        a    = torch.softmax(a, dim=1).unsqueeze(-1)
        pooled = (h * a).sum(dim=1)
        return self.head(pooled).squeeze(-1)


# =============================================================================
#  CACHED LOADERS
# =============================================================================
@st.cache_resource
def load_artifacts():
    """Load model + preprocessors + baseline hazard. Cached across reruns."""
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    imputer  = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    hz       = np.load(os.path.join(MODEL_DIR, "baseline_hazard.npz"))
    baseline_times, H0 = hz["baseline_times"], hz["H0"]

    # Prefer the FEATURE_COLUMNS list persisted by the training script.
    # If it's missing (old artifacts), fall back to the hardcoded list above.
    feat_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    if os.path.exists(feat_path):
        feature_columns = joblib.load(feat_path)
    else:
        feature_columns = FEATURE_COLUMNS

    # Read the checkpoint and infer the true input_dim from the first
    # weight matrix — this is bullet-proof against any list mismatch.
    state = torch.load(os.path.join(MODEL_DIR, "behrt_medbert_survival.pt"),
                       map_location=device)
    ckpt_input_dim = state["value_proj.0.weight"].shape[1]
    if ckpt_input_dim != len(feature_columns):
        st.warning(
            f"Checkpoint expects {ckpt_input_dim} features but "
            f"feature_columns has {len(feature_columns)}. Truncating the list "
            f"to the first {ckpt_input_dim} entries — verify the training "
            f"script and app share the same FEATURE_COLUMNS.")
        feature_columns = feature_columns[:ckpt_input_dim]

    model = BEHRTMedBERTSurvival(input_dim=ckpt_input_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    return (model, scaler, imputer, encoders,
            baseline_times, H0, device, feature_columns)


# =============================================================================
#  HELPERS
# =============================================================================
def safe_label_encode(le, value):
    """LabelEncoder.transform crashes on unseen labels. Map unseen to 0."""
    value = str(value)
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0


def age_bucket(age_years: float) -> int:
    edges = np.linspace(0, 120, MAX_AGE_BUCKETS + 1)
    b = int(np.digitize(age_years, edges, right=False) - 1)
    return max(0, min(MAX_AGE_BUCKETS - 1, b))


def build_sequence(visits_df: pd.DataFrame,
                   demographics: dict,
                   encoders, imputer, scaler, feature_columns):
    """
    Take raw visits + demographics, run the SAME preprocessing as training,
    pad/truncate to MAX_SEQ_LENGTH, and return (X, A, M).
    """
    visits_df = visits_df.copy().sort_values("VisitDate").reset_index(drop=True)

    # ---- enrich with patient-level fields ----
    for k, v in demographics.items():
        visits_df[k] = v

    # ---- derived features (mirror training script) ----
    visits_df["VisitDate"] = pd.to_datetime(visits_df["VisitDate"])
    dob = pd.to_datetime(demographics.get("DOB"))
    visits_df["AgeYears"] = (visits_df["VisitDate"] - dob).dt.days / 365.25

    vl_num = pd.to_numeric(visits_df.get("Lastest_VL"), errors="coerce")
    visits_df["viral_suppressed"] = (vl_num < 1000).astype(int)

    visits_df["prev_visit"]      = visits_df["VisitDate"].shift(1)
    visits_df["visit_gap_days"]  = (visits_df["VisitDate"]
                                    - visits_df["prev_visit"]).dt.days
    if "Return Appointment Date" in visits_df.columns:
        visits_df["prev_return"] = pd.to_datetime(
            visits_df["Return Appointment Date"]).shift(1)
        visits_df["days_late"]   = (visits_df["VisitDate"]
                                    - visits_df["prev_return"]).dt.days
    else:
        visits_df["days_late"] = np.nan

    if "DaysToReturn" in visits_df.columns:
        visits_df["DaysToReturn_prev"] = pd.to_numeric(
            visits_df["DaysToReturn"], errors="coerce").shift(1)
    else:
        visits_df["DaysToReturn_prev"] = np.nan

    # ---- categorical encoding using saved LabelEncoders ----
    for col, le in encoders.items():
        if col in visits_df.columns:
            visits_df[col] = visits_df[col].astype(str)
            visits_df[col] = visits_df[col].apply(lambda x: safe_label_encode(le, x))

    # ---- ensure all expected columns exist ----
    for c in feature_columns:
        if c not in visits_df.columns:
            visits_df[c] = np.nan

    feats_raw = visits_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    feats_imp = imputer.transform(feats_raw)
    feats_scl = scaler.transform(feats_imp).astype(np.float32)

    ages = visits_df["AgeYears"].fillna(demographics.get("Age", 35.0)) \
                                .apply(age_bucket).values.astype(np.int64)

    n = len(feats_scl)
    if n > MAX_SEQ_LENGTH:
        feats_scl = feats_scl[-MAX_SEQ_LENGTH:]
        ages      = ages[-MAX_SEQ_LENGTH:]
        n = MAX_SEQ_LENGTH

    pad = MAX_SEQ_LENGTH - n
    if pad > 0:
        feats_scl = np.vstack([feats_scl,
                               np.zeros((pad, len(feature_columns)), dtype=np.float32)])
        ages      = np.concatenate([ages, np.zeros(pad, dtype=np.int64)])
    mask = np.array([1] * n + [0] * pad, dtype=np.float32)

    return feats_scl, ages, mask, n


def predict_dynamic(model, device, X, A, M,
                    baseline_times, H0, t_query_days):
    """Run inference and convert log-risk + baseline hazard -> S(t*), P(LTFU)."""
    with torch.no_grad():
        x = torch.tensor(X[None], dtype=torch.float32, device=device)
        a = torch.tensor(A[None], dtype=torch.long,    device=device)
        m = torch.tensor(M[None], dtype=torch.float32, device=device)
        risk = float(model(x, a, m).cpu().item())

    exp_r = math.exp(max(min(risk, 15), -15))
    if t_query_days <= baseline_times[0]:
        H_t = 0.0
    elif t_query_days >= baseline_times[-1]:
        H_t = float(H0[-1])
    else:
        H_t = float(np.interp(t_query_days, baseline_times, H0))
    S_t = math.exp(-H_t * exp_r)
    return {"risk_score": risk, "exp_risk": exp_r,
            "H_t": H_t, "S_t": S_t, "P_ltfu": 1.0 - S_t}


def survival_curve(model, device, X, A, M, baseline_times, H0,
                   horizon_days=1500, step=30):
    """Compute the full S(t|x) curve for plotting."""
    with torch.no_grad():
        x = torch.tensor(X[None], dtype=torch.float32, device=device)
        a = torch.tensor(A[None], dtype=torch.long,    device=device)
        m = torch.tensor(M[None], dtype=torch.float32, device=device)
        risk = float(model(x, a, m).cpu().item())
    exp_r = math.exp(max(min(risk, 15), -15))

    ts = np.arange(0, horizon_days + 1, step)
    Ht = np.interp(ts, baseline_times, H0,
                   left=0.0, right=float(H0[-1]))
    St = np.exp(-Ht * exp_r)
    return ts, St


# =============================================================================
#  STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="LTFU Risk Predictor", page_icon="🏥", layout="wide")

st.title("🏥 LTFU Risk Predictor")
st.caption("BEHRT + Med-BERT transformer with a Cox survival head — "
           "trained on 2016-2019 longitudinal clinic data.")

# Load model
try:
    (model, scaler, imputer, encoders,
     baseline_times, H0, device, feature_columns) = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Could not load model artifacts from `{MODEL_DIR}/`. "
             f"Run the training script first.\n\nMissing: {e.filename}")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar: patient demographics
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🧑‍⚕️ Patient")
    pid       = st.text_input("Patient ID", value="PT-0001")
    gender    = st.selectbox("Gender", ["Female", "Male"], index=0)
    dob       = st.date_input("Date of birth",
                              value=date(1985, 1, 1),
                              min_value=date(1920, 1, 1),
                              max_value=date.today())
    reg_date  = st.date_input("Registration date",
                              value=date(2016, 1, 1),
                              min_value=date(2000, 1, 1),
                              max_value=date.today())
    art_start = st.date_input("ART start date",
                              value=date(2016, 1, 1),
                              min_value=date(2000, 1, 1),
                              max_value=date.today())

    st.markdown("---")
    st.subheader("Social")
    education = st.selectbox("Education",
                             ["None", "Primary", "Secondary", "Tertiary", "Unknown"])
    occupation = st.selectbox("Occupation",
                              ["Farmer", "Trader", "Student", "Employed",
                               "Unemployed", "Other", "Unknown"])
    marital   = st.selectbox("Marital status",
                             ["Single", "Married", "Divorced", "Widowed", "Unknown"])

age_today = (date.today() - dob).days / 365.25
st.sidebar.metric("Age (today)", f"{age_today:.1f} yrs")

demographics = {
    "Gender": gender,
    "DOB": dob,
    "RegistrationDate": reg_date,
    "ARTStartDate": art_start,
    "Age": age_today,
    "Education_level": education,
    "Occupation": occupation,
    "Marital_status": marital,
}

# ----------------------------------------------------------------------------
# Main: visit history (editable table)
# ----------------------------------------------------------------------------
st.subheader("📋 Visit history")
st.caption("Add one row per past clinic visit. Most recent visit last. "
           "Leave a cell blank if unknown — the model imputes missing values.")

# Default 3 example rows so the user has something to edit
default_visits = pd.DataFrame({
    "VisitDate":              [reg_date,
                               reg_date + timedelta(days=90),
                               reg_date + timedelta(days=180)],
    "ARTAdherenceScore":      [95.0, 92.0, 90.0],
    "BMI":                    [22.0, 21.5, 21.0],
    "Weight":                 [60.0, 59.0, 58.5],
    "Systolic BP":            [120.0, 125.0, 128.0],
    "Diastolic BP":           [80.0, 82.0, 84.0],
    "KarnofskyScore":         [90.0, 90.0, 80.0],
    "Last_CD4":               [350.0, 400.0, 420.0],
    "Lastest_VL":             [200.0, 150.0, 100.0],
    "WHOStage":               ["2", "2", "2"],
    "Regimen":                ["TDF/3TC/DTG"] * 3,
    "Curr_Regimen":           ["TDF/3TC/DTG"] * 3,
    "Funding Source":         ["PEPFAR"] * 3,
    "VisitType":              ["Routine"] * 3,
    "Appointment Type":       ["Scheduled"] * 3,
    "Return Appointment Date":[reg_date + timedelta(days=90),
                               reg_date + timedelta(days=180),
                               reg_date + timedelta(days=270)],
    "DaysToReturn":           [90, 90, 90],
    "TB":                     ["No"] * 3,
    "WeightLoss":             ["No"] * 3,
    "PersistentFever":        ["No"] * 3,
    "NightSweats":            ["No"] * 3,
    "Coughing":               ["No"] * 3,
    "OI_TB":                  ["No"] * 3,
    "OI_OralCandidiasis":     ["No"] * 3,
    "OI_PCP":                 ["No"] * 3,
    "NCD_Hypertension":       ["No"] * 3,
    "NCD_DiabetesMellitus":   ["No"] * 3,
    "NCD_MentalHealth":       ["No"] * 3,
    "Disclosure_status":      ["Disclosed"] * 3,
    "Disability":             ["No"] * 3,
    "DSDMType":               ["FBIM"] * 3,
    "Prophylaxis":            ["Yes"] * 3,
    "Anti_Hypertension":      ["No"] * 3,
    "Anti_Diabetes":          ["No"] * 3,
})

visits_df = st.data_editor(
    default_visits, num_rows="dynamic", use_container_width=True,
    column_config={
        "VisitDate":              st.column_config.DateColumn("Visit date", required=True),
        "Return Appointment Date":st.column_config.DateColumn("Next-visit date (then)"),
        "Lastest_VL":             st.column_config.NumberColumn("Viral load (cp/mL)",
                                                                format="%.0f"),
        "Last_CD4":               st.column_config.NumberColumn("CD4 (cells/µL)",
                                                                format="%.0f"),
        "ARTAdherenceScore":      st.column_config.NumberColumn("Adherence %",
                                                                min_value=0.0, max_value=100.0),
    },
)

# ----------------------------------------------------------------------------
# Prediction controls
# ----------------------------------------------------------------------------
st.subheader("🔮 Predict LTFU at the next visit")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    last_visit = visits_df["VisitDate"].dropna().max() if not visits_df["VisitDate"].dropna().empty else reg_date
    default_next = pd.to_datetime(last_visit).date() + timedelta(days=90)
    next_visit_date = st.date_input("Proposed next visit date",
                                    value=default_next,
                                    min_value=pd.to_datetime(reg_date).date()
                                              + timedelta(days=1))
with col2:
    horizon = st.slider("Survival-curve horizon (days)", 90, 1500, 1095, 30)
with col3:
    st.write("")  # spacer
    go = st.button("▶  Run prediction", type="primary",
                   use_container_width=True)

# ----------------------------------------------------------------------------
# Run prediction
# ----------------------------------------------------------------------------
if go:
    if len(visits_df.dropna(subset=["VisitDate"])) == 0:
        st.error("Add at least one visit with a date.")
        st.stop()

    t_query = (pd.to_datetime(next_visit_date)
               - pd.to_datetime(reg_date)).days
    if t_query <= 0:
        st.error("Next visit date must be after registration date.")
        st.stop()

    X, A, M, n_real = build_sequence(visits_df.dropna(subset=["VisitDate"]),
                                     demographics, encoders, imputer, scaler,
                                     feature_columns)

    out = predict_dynamic(model, device, X, A, M,
                          baseline_times, H0, t_query)
    ts, St = survival_curve(model, device, X, A, M,
                            baseline_times, H0, horizon_days=horizon, step=15)

    # ----- top KPIs -----
    p_ltfu = out["P_ltfu"]
    risk_band = ("🟢 Low"    if p_ltfu < 0.10 else
                 "🟡 Medium" if p_ltfu < 0.25 else
                 "🔴 High")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LTFU probability by next visit", f"{p_ltfu:.1%}", delta=risk_band)
    k2.metric("Survival probability",            f"{out['S_t']:.1%}")
    k3.metric("Days from registration",          f"{t_query} d")
    k4.metric("Log-risk score",                  f"{out['risk_score']:+.3f}")

    # ----- survival curve -----
    st.subheader("📈 Predicted retention curve")
    chart_df = pd.DataFrame({"Days from registration": ts,
                             "P(retained in care)": St})
    st.line_chart(chart_df, x="Days from registration",
                  y="P(retained in care)", height=320)

    # ----- guidance -----
    with st.expander("🩺 Clinical interpretation"):
        st.markdown(f"""
- The model summarises this patient's **{n_real} visit(s)** into a single
  log-risk score of **{out['risk_score']:+.3f}** (higher = more LTFU risk).
- Multiplied by the population baseline cumulative hazard at
  day **{t_query}** (H₀ = {out['H_t']:.4f}), this yields a
  predicted **{p_ltfu:.1%}** probability of being lost-to-follow-up by
  **{next_visit_date.strftime('%d %b %Y')}**.
- Risk band: **{risk_band}**.
- Clinical action thresholds you might apply:
  - **< 10 %** : routine scheduling.
  - **10 – 25 %** : SMS reminder + peer-support check-in.
  - **> 25 %** : proactive outreach (phone/home visit) before the appointment.
""")

    with st.expander("🧪 Raw model output"):
        st.json({k: round(v, 6) for k, v in out.items()})

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption("⚠️ Research prototype, not a medical device. "
           "Decisions must remain with the clinician.")
