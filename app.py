import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

DARK = st.session_state.dark_mode

# Palette
if DARK:
    BG = "#0e1420"
    CARD_BG = "#161e2e"
    TEXT = "#e8edf5"
    SUBTEXT = "#a9b4c4"
    BORDER = "#2a3547"
    ACCENT = "#6f9dff"
    TAB_INACTIVE_BG = "#1c253a"
    TAB_INACTIVE_TEXT = "#c7d0de"
    SIDEBAR_BG = "#111827"
else:
    BG = "#f7f9fb"
    CARD_BG = "#ffffff"
    TEXT = "#1f2933"
    SUBTEXT = "#5a6472"
    BORDER = "#e3e8ee"
    ACCENT = "#5b8def"
    TAB_INACTIVE_BG = "#eef2f8"
    TAB_INACTIVE_TEXT = "#3d4654"
    SIDEBAR_BG = "#ffffff"

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0) !important; }}
    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

    p, span, label, li, div {{ color: {TEXT}; }}

    .main-header {{
        padding: 1.6rem 2rem;
        background: linear-gradient(120deg, #5b8def 0%, #8f6fff 100%);
        border-radius: 16px;
        color: white !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(91,141,239,0.25);
    }}
    .main-header h1, .main-header p {{ color: white !important; }}
    .main-header h1 {{ margin-bottom: 0.2rem; }}
    .main-header p {{ margin: 0; opacity: 0.95; }}

    .result-card {{
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid {BORDER};
        animation: fadein 0.5s ease-in-out;
    }}
    @keyframes fadein {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .benign {{ background-color: {"#123322" if DARK else "#e7f7ee"}; border-color: {"#2f6b48" if DARK else "#a9e3c3"} !important; }}
    .malignant {{ background-color: {"#3a1a17" if DARK else "#fdeceb"}; border-color: {"#8a3a32" if DARK else "#f3b7b1"} !important; }}
    .result-card h2 {{ margin-bottom: 0.3rem; }}
    .benign h2 {{ color: {"#5fd68d" if DARK else "#1e7e45"} !important; }}
    .malignant h2 {{ color: {"#ff8a80" if DARK else "#b93227"} !important; }}
    .result-card p {{ color: {SUBTEXT} !important; }}

    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.9rem;
    }}
    div[data-testid="stMetric"] label {{ color: {SUBTEXT} !important; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {TAB_INACTIVE_BG};
        border-radius: 10px 10px 0 0;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: {TAB_INACTIVE_TEXT} !important; }}
    .stTabs [aria-selected="true"] {{ background-color: {ACCENT} !important; }}
    .stTabs [aria-selected="true"] p {{ color: white !important; }}

    [data-testid="stExpander"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    .stSlider [data-baseweb="slider"] {{ padding-top: 4px; }}

    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (scaler -> pca -> svm pipeline)
# ─────────────────────────────────────────────────────────────
ARTIFACT_DIR = "artifacts"

@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "svm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return scaler, pca, model

try:
    scaler, pca, model = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ─────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS (Wisconsin Diagnostic Breast Cancer dataset)
# ─────────────────────────────────────────────────────────────
BASE_FEATURES = [
    ("radius", "Radius", 6.0, 30.0, 14.1),
    ("texture", "Texture", 9.0, 40.0, 19.3),
    ("perimeter", "Perimeter", 40.0, 190.0, 92.0),
    ("area", "Area", 140.0, 2500.0, 655.0),
    ("smoothness", "Smoothness", 0.05, 0.17, 0.096),
    ("compactness", "Compactness", 0.01, 0.35, 0.104),
    ("concavity", "Concavity", 0.0, 0.43, 0.089),
    ("concave_points", "Concave Points", 0.0, 0.2, 0.048),
    ("symmetry", "Symmetry", 0.1, 0.31, 0.181),
    ("fractal_dimension", "Fractal Dimension", 0.05, 0.1, 0.063),
]

GROUPS = [
    ("mean", "Mean Values", 1.0),
    ("se", "Standard Error", 0.08),
    ("worst", "Worst Values", 1.25),
]

FEATURE_ORDER = [f"{key}_{grp}" for grp, _, _ in GROUPS for key, *_ in BASE_FEATURES]
BENIGN_PROFILE = {f"{key}_{grp}": round(default * scale * 0.85, 4) for grp, _, scale in GROUPS for key, _, _, _, default in BASE_FEATURES}
MALIGNANT_PROFILE = {f"{key}_{grp}": round(default * scale * 1.4, 4) for grp, _, scale in GROUPS for key, _, _, _, default in BASE_FEATURES}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 About")
    st.write(
        "This app predicts whether a breast tumor is **benign** or "
        "**malignant** using cell nuclei measurements from a digitized "
        "biopsy image, based on the Wisconsin Diagnostic Breast Cancer "
        "dataset."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline")
    st.write("Input features → **StandardScaler** → **PCA** → **SVM Classifier**")
    st.markdown("---")
    st.toggle("🌙 Dark mode", value=st.session_state.dark_mode, on_change=toggle_theme)
    st.markdown("---")
    input_mode = st.radio(
        "Input method",
        ["Manual sliders", "Load sample patient", "Upload CSV row"],
    )
    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational/demonstration purposes only and "
        "is **not** a substitute for professional medical diagnosis."
    )

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🩺 Breast Cancer Diagnosis Predictor</h1>
        <p>Predict tumor malignancy from cell nuclei measurements using a trained SVM model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_loaded:
    st.error(
        f"Could not find model files in the **`{ARTIFACT_DIR}/`** folder. "
        f"Please make sure `scaler.pkl`, `pca.pkl`, and `svm_model.pkl` are "
        f"placed inside a folder named `{ARTIFACT_DIR}` next to `app.py` "
        f"before deploying."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# INPUT COLLECTION
# ─────────────────────────────────────────────────────────────
values = {}

if input_mode == "Manual sliders":
    tabs = st.tabs([label for _, label, _ in GROUPS])
    for (grp_key, grp_label, scale_factor), tab in zip(GROUPS, tabs):
        with tab:
            cols = st.columns(2)
            for i, (key, label, lo, hi, default) in enumerate(BASE_FEATURES):
                col = cols[i % 2]
                scaled_lo = round(lo * scale_factor, 4)
                scaled_hi = round(hi * scale_factor, 4)
                scaled_default = round(default * scale_factor, 4)
                with col:
                    values[f"{key}_{grp_key}"] = st.slider(
                        f"{label} ({grp_label})",
                        min_value=float(scaled_lo),
                        max_value=float(scaled_hi),
                        value=float(scaled_default),
                        key=f"{key}_{grp_key}",
                    )

elif input_mode == "Load sample patient":
    st.info("A representative sample patient has been pre-filled below. Adjust values if needed, then predict.")
    sample_choice = st.selectbox("Choose a sample profile", ["Typical benign profile", "Typical malignant profile"])
    values = dict(MALIGNANT_PROFILE if sample_choice == "Typical malignant profile" else BENIGN_PROFILE)
    with st.expander("View pre-filled values"):
        st.json(values)

else:  # Upload CSV row
    st.write(
        "Upload a single-row CSV containing the 30 WDBC features "
        "(mean, se, and worst values), in standard column order."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        row = df_up.iloc[0]
        for f in FEATURE_ORDER:
            values[f] = row.get(f, 0.0)
        st.dataframe(df_up, use_container_width=True)
    else:
        st.warning("Please upload a CSV file to continue, or switch input methods.")

# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Analysis")

predict_disabled = len(values) < len(FEATURE_ORDER)

if st.button("🔬 Analyze Tumor", type="primary", disabled=predict_disabled, use_container_width=True):
    input_array = np.array([[values[f] for f in FEATURE_ORDER]])

    scaled = scaler.transform(input_array)
    reduced = pca.transform(scaled)
    prediction = model.predict(reduced)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(reduced)[0]

    is_malignant = str(prediction) in ("1", "M", "malignant")

    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        if is_malignant:
            st.markdown(
                """
                <div class="result-card malignant">
                    <h2>⚠️ Malignant</h2>
                    <p>The model predicts this tumor is likely <b>malignant</b>.
                    Please consult an oncologist for a confirmed diagnosis.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card benign">
                    <h2>✅ Benign</h2>
                    <p>The model predicts this tumor is likely <b>benign</b>.
                    Regular check-ups are still recommended.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if proba is not None:
            c1, c2 = st.columns(2)
            c1.metric("Benign probability", f"{proba[0]*100:.1f}%")
            c2.metric("Malignant probability", f"{proba[1]*100:.1f}%")

    with gauge_col:
        if proba is not None:
            confidence = proba[1] * 100 if is_malignant else proba[0] * 100
            gauge_color = "#ff6b5f" if is_malignant else "#3ecf8e"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "%", "font": {"color": TEXT}},
                title={"text": "Model Confidence", "font": {"color": TEXT}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": TEXT},
                    "bar": {"color": gauge_color},
                    "bgcolor": CARD_BG,
                    "borderwidth": 1,
                    "bordercolor": BORDER,
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=20, r=20, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Radar chart comparing patient's mean-value profile vs typical benign/malignant
    st.markdown("#### 📊 Profile Comparison (Mean Values)")
    radar_labels = [label for _, label, _, _, _ in BASE_FEATURES]
    patient_vals = [values[f"{key}_mean"] for key, *_ in BASE_FEATURES]
    benign_vals = [BENIGN_PROFILE[f"{key}_mean"] for key, *_ in BASE_FEATURES]
    malignant_vals = [MALIGNANT_PROFILE[f"{key}_mean"] for key, *_ in BASE_FEATURES]

    def normalize(vals):
        maxes = [hi for _, _, _, hi, _ in BASE_FEATURES]
        return [v / m for v, m in zip(vals, maxes)]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=normalize(benign_vals), theta=radar_labels, fill="toself", name="Typical Benign", line=dict(color="#3ecf8e")))
    radar_fig.add_trace(go.Scatterpolar(r=normalize(malignant_vals), theta=radar_labels, fill="toself", name="Typical Malignant", line=dict(color="#ff6b5f")))
    radar_fig.add_trace(go.Scatterpolar(r=normalize(patient_vals), theta=radar_labels, fill="toself", name="Current Input", line=dict(color=ACCENT, width=3)))
    radar_fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 1], color=SUBTEXT),
            angularaxis=dict(color=TEXT),
        ),
        showlegend=True,
        legend=dict(font=dict(color=TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    with st.expander("See processed feature vector (after scaling & PCA)"):
        st.write("Scaled input:", scaled)
        st.write("PCA-reduced input:", reduced)

    report_df = pd.DataFrame([values])
    report_df["prediction"] = "Malignant" if is_malignant else "Benign"
    st.download_button(
        "⬇️ Download this prediction as CSV",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="breast_cancer_prediction.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Built with Streamlit · SVM classifier trained on the Wisconsin Diagnostic Breast Cancer dataset")
