# LTFU Risk Predictor – Streamlit App

A clinical-decision-support prototype that wraps the BEHRT + Med-BERT + Cox
survival model from `ltfu_behrt_medbert_survival.py`. Enter a patient's
visit history and a proposed next visit date; the app returns the
**probability of being lost to follow-up (LTFU) by that date** and plots the
predicted retention curve.

---

## 1. Folder layout

```
project/
├── app.py                                  # this Streamlit app
├── requirements.txt
├── ltfu_behrt_medbert_survival.py          # training script (run once)
└── models/                                 # produced by the training script
    ├── behrt_medbert_survival.pt
    ├── scaler.pkl
    ├── imputer.pkl
    ├── encoders.pkl
    └── baseline_hazard.npz
```

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train the model (once)

Put `2016_2019_ClinicDataWide.xlsx` next to the training script, then:

```bash
python ltfu_behrt_medbert_survival.py
```

This creates the `models/` directory with the five artifacts above.

## 4. Run the app

```bash
streamlit run app.py
```

A browser tab opens at <http://localhost:8501>.

## 5. Using the app

1. **Sidebar** – enter patient demographics (gender, DOB, registration date,
   ART start date, education, occupation, marital status).
2. **Visit history** – add one row per past clinic visit in the editable
   table. Most recent visit last. Unknown fields can be left blank — the
   trained imputer fills them in.
3. **Next visit date** – the date you are considering scheduling.
4. Click **Run prediction**. The app shows:
   - LTFU probability by the next visit (with a Low/Medium/High band).
   - Predicted retention probability.
   - A survival-curve chart out to your chosen horizon.
   - Suggested clinical action tiers.

## 6. How the prediction works

The model produces a scalar log-risk `r̂` (BEHRT/Med-BERT transformer over
the visit sequence + attention pooling + Cox head). The probability of
remaining in care by day `t*` is

```
S(t* | x) = exp[ -H₀(t*) · exp(r̂) ]
P(LTFU by t*) = 1 - S(t* | x)
```

`H₀(t)` is the Breslow baseline cumulative hazard estimated from the
training set and saved in `models/baseline_hazard.npz`.

## 7. Limitations

- Research prototype; **not** a medical device.
- Trained on a single Ugandan clinic cohort (2016 – 2019); recalibrate
  before deploying elsewhere.
- Unseen categorical values are mapped to bucket `0` rather than rejected.
- The proportional-hazards assumption is implicit in the Cox head; for
  long horizons consider migrating to a discrete-time DeepHit-style head.
