import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from datetime import datetime, time
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ITSM AI — ABC Tech",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0f1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1e293b;
}

/* Main background */
.main .block-container {
    background-color: #0b0f1a;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Priority badges */
.badge-p1 { background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.badge-p2 { background:#7c2d12; color:#fdba74; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.badge-p3 { background:#713f12; color:#fde68a; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.badge-p4 { background:#14532d; color:#86efac; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }
.badge-p5 { background:#1e3a5f; color:#93c5fd; padding:3px 10px; border-radius:20px; font-size:0.82rem; font-weight:600; }

/* HIGH PRIORITY alert */
.alert-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    color: #fca5a5;
    font-weight: 600;
    font-size: 1rem;
    animation: pulse 1.5s ease-in-out infinite alternate;
}
.alert-normal {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    color: #86efac;
    font-weight: 600;
    font-size: 1rem;
}
.alert-rfc {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #818cf8;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #c7d2fe;
    font-size: 0.92rem;
}

@keyframes pulse {
    from { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    to   { box-shadow: 0 0 12px 4px rgba(239,68,68,0.15); }
}

/* Section headers */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
}

/* Streamlit tweaks */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

.stButton>button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }

.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

hr { border-color: #1e293b; }

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    files = {
        'priority':     'itsm_priority_model.pkl',
        'highpriority': 'itsm_highpriority_model.pkl',
        'scaler':       'itsm_scaler.pkl',
        'rfc':          'itsm_rfc_model.pkl',
    }
    for key, fname in files.items():
        try:
            with open(fname, 'rb') as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            models[key] = None
    return models

models = load_models()

FEATURE_COLS = [
    'CI_Cat', 'CI_Subcat', 'Category', 'Closure_Code',
    'No_of_Reassignments', 'No_of_Related_Interactions',
    'Handle_Time_hrs', 'CI_Name_freq',
    'Open_Hour', 'Open_DayOfWeek', 'Open_Month', 'Open_Year',
    'Is_Weekend', 'Is_BusinessHour'
]

FEAT_T4 = [c for c in FEATURE_COLS if c != 'No_of_Related_Changes']

PRIORITY_LABELS = {1:'P1 — Critical', 2:'P2 — High', 3:'P3 — Medium', 4:'P4 — Low', 5:'P5 — Very Low'}
PRIORITY_COLORS = {1:'#ef4444', 2:'#f97316', 3:'#eab308', 4:'#22c55e', 5:'#3b82f6'}

CI_CAT_OPTIONS   = ['Hardware', 'Software', 'Network', 'Database', 'Security', 'Application', 'Infrastructure']
CI_SUBCAT_OPTIONS= ['Web Based Application', 'Desktop App', 'Server', 'Router', 'Switch', 'Storage', 'VM', 'Firewall']
CATEGORY_OPTIONS = ['incident', 'problem', 'change', 'service request']
CLOSURE_OPTIONS  = ['Resolved', 'Closed', 'Cancelled', 'Other', 'Duplicate']
ALERT_OPTIONS    = ['closed', 'open', 'none']

def encode_label(val, options):
    try:
        return options.index(val)
    except ValueError:
        return 0

def build_feature_vector(inputs):
    row = {col: 0 for col in FEATURE_COLS}
    row['CI_Cat']                    = encode_label(inputs['ci_cat'],    CI_CAT_OPTIONS)
    row['CI_Subcat']                 = encode_label(inputs['ci_subcat'], CI_SUBCAT_OPTIONS)
    row['Category']                  = encode_label(inputs['category'],  CATEGORY_OPTIONS)
    row['Closure_Code']              = encode_label(inputs['closure'],   CLOSURE_OPTIONS)
    row['No_of_Reassignments']       = inputs['reassignments']
    row['No_of_Related_Interactions']= inputs['interactions']
    row['Handle_Time_hrs']           = inputs['handle_time']
    row['CI_Name_freq']              = inputs['ci_freq']
    row['Open_Hour']                 = inputs['hour']
    row['Open_DayOfWeek']            = inputs['dow']
    row['Open_Month']                = inputs['month']
    row['Open_Year']                 = inputs['year']
    row['Is_Weekend']                = 1 if inputs['dow'] >= 5 else 0
    row['Is_BusinessHour']           = 1 if 9 <= inputs['hour'] <= 18 else 0
    return pd.DataFrame([row])

def plot_style():
    plt.rcParams.update({
        'figure.facecolor': '#111827',
        'axes.facecolor':   '#111827',
        'axes.edgecolor':   '#1e293b',
        'axes.labelcolor':  '#94a3b8',
        'xtick.color':      '#64748b',
        'ytick.color':      '#64748b',
        'text.color':       '#e2e8f0',
        'grid.color':       '#1e293b',
        'grid.alpha':       0.5,
    })

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size:1.4rem; font-weight:700; color:#38bdf8;'>🛡️ ITSM AI</div>
        <div style='font-size:0.72rem; color:#475569; letter-spacing:0.1em; text-transform:uppercase; margin-top:4px;'>ABC Tech Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model Status</div>", unsafe_allow_html=True)

    model_info = [
        ("Priority Auto-Tag",   "priority",     "XGBoost",  "0.82 Acc"),
        ("High-Priority Alert", "highpriority", "XGBoost",  "0.98 Acc"),
        ("RFC Predictor",       "rfc",          "RF",       "0.98 Acc"),
        ("Scaler",              "scaler",       "StandardScaler", "—"),
    ]
    for label, key, algo, metric in model_info:
        status = "🟢" if models.get(key) else "🔴"
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:6px 4px; border-bottom:1px solid #1e293b; font-size:0.8rem;'>
            <span>{status} {label}</span>
            <span style='color:#475569; font-family:Space Mono,monospace; font-size:0.7rem;'>{metric}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)

    perf = {
        'Task 1 — High Priority': {'Accuracy': 0.98, 'F1': 0.98},
        'Task 2 — Forecasting':   {'Accuracy': None, 'F1': 0.31},
        'Task 3 — Auto-Tag':      {'Accuracy': 0.82, 'F1': 0.71},
        'Task 4 — RFC':           {'Accuracy': 0.98, 'F1': 0.68},
    }
    for task, scores in perf.items():
        acc_str = f"{scores['Accuracy']:.0%}" if scores['Accuracy'] else "—"
        f1_str  = f"{scores['F1']:.2f}" if scores['Accuracy'] else f"R²={scores['F1']:.2f}"
        st.markdown(f"""
        <div style='padding:5px 4px; font-size:0.75rem; border-bottom:1px solid #1e293b;'>
            <div style='color:#cbd5e1;'>{task}</div>
            <div style='color:#64748b; font-family:Space Mono,monospace;'>Acc:{acc_str} · F1:{f1_str}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("PRCL-0012 | DataMites™ | ABC Tech")

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;'>
    <h1 style='font-family:Space Mono,monospace; font-size:1.6rem; font-weight:700;
               color:#f1f5f9; margin:0; letter-spacing:-0.02em;'>
        ITSM Incident Intelligence Platform
    </h1>
    <p style='color:#475569; font-size:0.85rem; margin:4px 0 0 0;'>
        Real-time ML predictions · Priority tagging · Volume forecasting · RFC detection
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predict", "📊 Analyze", "📈 Forecast", "🗂️ Dashboard", "🤖 AI Assistant"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Ticket Input — Real-time Prediction</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        st.markdown("**Ticket Details**")
        c1, c2 = st.columns(2)
        with c1:
            ci_cat    = st.selectbox("CI Category",    CI_CAT_OPTIONS)
            category  = st.selectbox("Ticket Category", CATEGORY_OPTIONS)
            closure   = st.selectbox("Closure Code",   CLOSURE_OPTIONS)
        with c2:
            ci_subcat = st.selectbox("CI Sub-Category", CI_SUBCAT_OPTIONS)
            reassign  = st.number_input("No. of Reassignments", 0, 50, 0)
            interact  = st.number_input("Related Interactions",  0, 50, 1)

        st.markdown("**Timing & CI Info**")
        c3, c4 = st.columns(2)
        with c3:
            open_time = st.time_input("Ticket Open Time", value=time(9, 0))
            open_date = st.date_input("Ticket Open Date", value=datetime.today())
        with c4:
            handle_hrs = st.number_input("Handle Time (hrs)", 0.0, 5000.0, 2.0, step=0.5)
            ci_freq    = st.number_input("CI Frequency (tickets/yr)", 0, 10000, 50)

        predict_btn = st.button("⚡  Run All Predictions")

    with col_r:
        if predict_btn:
            dow = open_date.weekday()
            inputs = dict(
                ci_cat=ci_cat, ci_subcat=ci_subcat, category=category,
                closure=closure, reassignments=reassign, interactions=interact,
                handle_time=handle_hrs, ci_freq=ci_freq,
                hour=open_time.hour, dow=dow,
                month=open_date.month, year=open_date.year
            )
            X = build_feature_vector(inputs)

            if models['scaler']:
                X_sc = models['scaler'].transform(X)
            else:
                X_sc = X.values

            # ── Task 1: High Priority ──
            st.markdown("<div class='section-title'>Task 1 — High Priority Alert</div>", unsafe_allow_html=True)
            if models['highpriority']:
                hp = models['highpriority'].predict(X_sc)[0]
                hp_prob = models['highpriority'].predict_proba(X_sc)[0][1]
                if hp == 1:
                    st.markdown(f"""<div class='alert-high'>
                        🚨 HIGH PRIORITY TICKET DETECTED<br>
                        <span style='font-size:0.82rem; font-weight:400;'>
                        Confidence: {hp_prob:.1%} — Escalate immediately to on-call engineer</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='alert-normal'>
                        ✅ Normal Priority Ticket<br>
                        <span style='font-size:0.82rem; font-weight:400;'>
                        High-Priority probability: {hp_prob:.1%}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("High Priority model not loaded.")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Task 3: Priority Auto-Tag ──
            st.markdown("<div class='section-title'>Task 3 — Priority Auto-Tag (P2–P5)</div>", unsafe_allow_html=True)
            if models['priority']:
                y_min = 2
                pred_raw = models['priority'].predict(X_sc)[0]
                priority = int(pred_raw) + y_min
                priority = max(2, min(5, priority))
                proba = models['priority'].predict_proba(X_sc)[0]

                badge_cls = f"badge-p{priority}"
                p_label   = PRIORITY_LABELS.get(priority, f"P{priority}")
                st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div style='font-size:0.78rem; color:#64748b; margin-bottom:8px;'>PREDICTED PRIORITY</div>
                    <span class='{badge_cls}' style='font-size:1.1rem; padding:6px 20px;'>{p_label}</span>
                </div>""", unsafe_allow_html=True)

                plot_style()
                fig, ax = plt.subplots(figsize=(5, 2.2))
                classes = [f"P{i}" for i in range(2, 2+len(proba))]
                colors  = [PRIORITY_COLORS.get(i+2, '#38bdf8') for i in range(len(proba))]
                bars = ax.barh(classes, proba, color=colors, height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability", fontsize=8)
                ax.set_title("Class Probabilities", fontsize=9, color='#94a3b8')
                for bar, p in zip(bars, proba):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{p:.2f}', va='center', fontsize=7, color='#94a3b8')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            else:
                st.warning("Priority model not loaded.")

            # ── Task 4: RFC ──
            st.markdown("<div class='section-title'>Task 4 — RFC Detection</div>", unsafe_allow_html=True)
            if models['rfc']:
                X_t4 = X[FEAT_T4] if all(c in X.columns for c in FEAT_T4) else X
                if models['scaler']:
                    X_t4_sc = models['scaler'].transform(X_t4)
                else:
                    X_t4_sc = X_t4.values
                rfc_pred = models['rfc'].predict(X_t4_sc)[0]
                rfc_prob = models['rfc'].predict_proba(X_t4_sc)[0][1]
                icon = "📋" if rfc_pred == 1 else "✅"
                msg  = "RFC likely — prepare change request" if rfc_pred == 1 else "No RFC expected"
                st.markdown(f"""<div class='alert-rfc'>
                    {icon} <strong>{msg}</strong><br>
                    <span style='font-size:0.8rem;'>RFC probability: {rfc_prob:.1%}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("RFC model not loaded.")

        else:
            st.markdown("""
            <div style='height:400px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center; color:#334155;
                        border:1px dashed #1e293b; border-radius:12px; text-align:center;'>
                <div style='font-size:2.5rem;'>⚡</div>
                <div style='font-family:Space Mono,monospace; font-size:0.85rem; margin-top:10px;'>
                    Fill in ticket details and click<br>Run All Predictions
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Upload ITSM Data for Analysis</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your ITSM CSV (itsm_data.csv format)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        df['Handle_Time_hrs'] = (
            df['Handle_Time_hrs'].astype(str)
            .str.replace(',', '.', regex=False)
            .apply(pd.to_numeric, errors='coerce')
        )

        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        total    = len(df)
        hp_count = df['Priority'].isin([1,2]).sum() if 'Priority' in df.columns else 0
        avg_ht   = df['Handle_Time_hrs'].median() if 'Handle_Time_hrs' in df.columns else 0
        reassign = df['No_of_Reassignments'].mean() if 'No_of_Reassignments' in df.columns else 0
        rfc_cnt  = (df['No_of_Related_Changes'] > 0).sum() if 'No_of_Related_Changes' in df.columns else 0

        for col, val, label in [
            (k1, f"{total:,}",        "Total Incidents"),
            (k2, f"{hp_count:,}",     "High Priority (P1/P2)"),
            (k3, f"{avg_ht:.1f}h",    "Median Handle Time"),
            (k4, f"{reassign:.1f}",   "Avg Reassignments"),
            (k5, f"{rfc_cnt:,}",      "RFC Tickets"),
        ]:
            col.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        plot_style()

        col_a, col_b = st.columns(2)
        with col_a:
            if 'Priority' in df.columns:
                st.markdown("<div class='section-title'>Priority Distribution</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 3))
                vc = df['Priority'].value_counts().sort_index()
                colors = [PRIORITY_COLORS.get(int(p), '#38bdf8') for p in vc.index]
                ax.bar([f"P{int(p)}" for p in vc.index], vc.values, color=colors, width=0.6)
                ax.set_ylabel("Count", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with col_b:
            if 'No_of_Reassignments' in df.columns and 'Priority' in df.columns:
                st.markdown("<div class='section-title'>Reassignments by Priority</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 3))
                df_clean = df.dropna(subset=['Priority', 'No_of_Reassignments'])
                df_clean['Priority'] = pd.to_numeric(df_clean['Priority'], errors='coerce')
                for p in sorted(df_clean['Priority'].dropna().unique()):
                    data = df_clean[df_clean['Priority'] == p]['No_of_Reassignments']
                    ax.scatter([p]*len(data), data, alpha=0.15, s=6,
                               color=PRIORITY_COLORS.get(int(p), '#38bdf8'))
                ax.set_xlabel("Priority", fontsize=8)
                ax.set_ylabel("Reassignments", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        col_c, col_d = st.columns(2)
        with col_c:
            if 'Open_Time' in df.columns:
                st.markdown("<div class='section-title'>Monthly Incident Trend</div>", unsafe_allow_html=True)
                df['Open_Time'] = pd.to_datetime(df['Open_Time'], errors='coerce')
                monthly = df.groupby(df['Open_Time'].dt.to_period('M')).size()
                monthly.index = monthly.index.to_timestamp()
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.fill_between(monthly.index, monthly.values, alpha=0.25, color='#38bdf8')
                ax.plot(monthly.index, monthly.values, color='#38bdf8', linewidth=2)
                ax.set_ylabel("Incidents", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        with col_d:
            if 'Handle_Time_hrs' in df.columns and 'Priority' in df.columns:
                st.markdown("<div class='section-title'>Handle Time Distribution (clipped 95%)</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 3))
                clip_val = df['Handle_Time_hrs'].quantile(0.95)
                df_clip = df[df['Handle_Time_hrs'] < clip_val]
                df_clip['Priority'] = pd.to_numeric(df_clip['Priority'], errors='coerce')
                for p in sorted(df_clip['Priority'].dropna().unique()):
                    vals = df_clip[df_clip['Priority'] == p]['Handle_Time_hrs'].dropna()
                    ax.hist(vals, bins=20, alpha=0.5, label=f"P{int(p)}",
                            color=PRIORITY_COLORS.get(int(p), '#38bdf8'))
                ax.set_xlabel("Handle Time (hrs)", fontsize=8)
                ax.legend(fontsize=7)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.markdown("<br><div class='section-title'>Batch Prediction on Uploaded Data</div>", unsafe_allow_html=True)
        if st.button("Run Batch Predictions on Dataset") and models['priority'] and models['scaler']:
            try:
                df_enc = df.copy()
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in ['CI_Cat', 'CI_Subcat', 'Category', 'Closure_Code', 'Alert_Status', 'Status']:
                    if col in df_enc.columns:
                        df_enc[col] = df_enc[col].fillna('Unknown')
                        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                df_enc['Open_Time'] = pd.to_datetime(df_enc['Open_Time'], errors='coerce')
                df_enc['Open_Hour']       = df_enc['Open_Time'].dt.hour.fillna(9)
                df_enc['Open_DayOfWeek']  = df_enc['Open_Time'].dt.dayofweek.fillna(0)
                df_enc['Open_Month']      = df_enc['Open_Time'].dt.month.fillna(1)
                df_enc['Open_Year']       = df_enc['Open_Time'].dt.year.fillna(2013)
                df_enc['Is_Weekend']      = (df_enc['Open_DayOfWeek'] >= 5).astype(int)
                df_enc['Is_BusinessHour'] = ((df_enc['Open_Hour'] >= 9) & (df_enc['Open_Hour'] <= 18)).astype(int)
                ci_freq = df['CI_Name'].value_counts().to_dict() if 'CI_Name' in df.columns else {}
                df_enc['CI_Name_freq'] = df['CI_Name'].map(ci_freq).fillna(0) if 'CI_Name' in df.columns else 0

                avail = [c for c in FEATURE_COLS if c in df_enc.columns]
                X_batch = df_enc[avail].fillna(0)
                X_batch_sc = models['scaler'].transform(X_batch)
                y_min = 2
                preds = models['priority'].predict(X_batch_sc) + y_min
                preds = np.clip(preds, 2, 5)
                df['Predicted_Priority'] = preds

                st.dataframe(
                    df[['Incident_ID', 'Priority', 'Predicted_Priority']].head(200)
                    if 'Incident_ID' in df.columns
                    else df[['Priority', 'Predicted_Priority']].head(200),
                    use_container_width=True
                )
                correct = (df['Priority'] == df['Predicted_Priority']).mean()
                st.success(f"Batch complete — {len(df):,} rows | Accuracy on this upload: {correct:.1%}")
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
    else:
        st.info("Upload your ITSM CSV file to start analysis.")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Task 2 — Incident Volume Forecasting</div>", unsafe_allow_html=True)

    fc_uploaded = st.file_uploader("Upload ITSM CSV for forecasting", type=["csv"], key="fc_upload")

    if fc_uploaded:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        df_fc = pd.read_csv(fc_uploaded)
        df_fc['Open_Time'] = pd.to_datetime(df_fc['Open_Time'], errors='coerce')
        df_fc['YearMonth_dt'] = df_fc['Open_Time'].dt.to_period('M').dt.to_timestamp()
        monthly = df_fc.groupby('YearMonth_dt').size().reset_index(name='Incident_Count')
        monthly = monthly.sort_values('YearMonth_dt').reset_index(drop=True)

        monthly['month_num'] = monthly['YearMonth_dt'].dt.month
        monthly['year_num']  = monthly['YearMonth_dt'].dt.year
        monthly['t']         = range(len(monthly))
        monthly['lag_1']     = monthly['Incident_Count'].shift(1)
        monthly['lag_2']     = monthly['Incident_Count'].shift(2)
        monthly['rolling_3'] = monthly['Incident_Count'].rolling(3).mean()
        monthly['Quarter']   = monthly['YearMonth_dt'].dt.quarter
        monthly_clean = monthly.dropna().reset_index(drop=True)

        X_ts = monthly_clean[['month_num','year_num','t','lag_1','lag_2','rolling_3']]
        y_ts = monthly_clean['Incident_Count']
        split_idx = max(1, len(X_ts) - 6)
        X_tr, X_te = X_ts.iloc[:split_idx], X_ts.iloc[split_idx:]
        y_tr, y_te = y_ts.iloc[:split_idx], y_ts.iloc[split_idx:]

        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_tr, y_tr)
        y_pred = rf_reg.predict(X_te)

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2   = r2_score(y_te, y_pred)

        m1, m2, m3 = st.columns(3)
        for col, val, label in [
            (m1, f"{mae:.0f}", "MAE (incidents/mo)"),
            (m2, f"{rmse:.0f}", "RMSE"),
            (m3, f"{r2:.2f}", "R² Score"),
        ]:
            col.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        plot_style()

        st.markdown("<div class='section-title'>Actual vs Predicted — Last 6 Months</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(monthly_clean['YearMonth_dt'].iloc[split_idx:], y_te.values,
                label='Actual', marker='o', color='#38bdf8', linewidth=2)
        ax.plot(monthly_clean['YearMonth_dt'].iloc[split_idx:], y_pred,
                label='Predicted', marker='s', linestyle='--', color='#f97316', linewidth=2)
        ax.fill_between(monthly_clean['YearMonth_dt'].iloc[split_idx:],
                        y_te.values, y_pred, alpha=0.1, color='#ef4444')
        ax.legend(fontsize=9)
        ax.set_ylabel("Incidents", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        col_q, col_a = st.columns(2)
        forecast_df = monthly_clean.iloc[split_idx:].copy()
        forecast_df['Predicted_Count'] = y_pred.astype(int)

        with col_q:
            st.markdown("<div class='section-title'>Quarterly Forecast</div>", unsafe_allow_html=True)
            q_df = forecast_df.groupby(['year_num','Quarter'])['Predicted_Count'].sum().reset_index()
            q_df.columns = ['Year','Quarter','Predicted Incidents']
            st.dataframe(q_df, use_container_width=True, hide_index=True)

        with col_a:
            st.markdown("<div class='section-title'>Annual Forecast</div>", unsafe_allow_html=True)
            a_df = forecast_df.groupby('year_num')['Predicted_Count'].sum().reset_index()
            a_df.columns = ['Year','Predicted Incidents']
            st.dataframe(a_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Full Monthly Trend (All Data)</div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(12, 3.5))
        ax2.fill_between(monthly['YearMonth_dt'], monthly['Incident_Count'], alpha=0.15, color='#38bdf8')
        ax2.plot(monthly['YearMonth_dt'], monthly['Incident_Count'], color='#38bdf8', linewidth=1.5)
        ax2.set_ylabel("Incidents", fontsize=9)
        ax2.set_title("Historical Incident Volume", fontsize=10, color='#94a3b8')
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown("<br><div class='section-title'>Future Incident Volume Projection</div>", unsafe_allow_html=True)
        n_future = st.slider("Months to project forward", min_value=3, max_value=12, value=6, step=3)

        last_date   = monthly_clean['YearMonth_dt'].iloc[-1]
        history     = list(monthly_clean['Incident_Count'].values)
        future_dates  = []
        future_preds  = []
        t_counter = int(monthly_clean['t'].iloc[-1]) + 1

        for i in range(1, n_future + 1):
            future_date = last_date + pd.DateOffset(months=i)
            lag_1     = history[-1]
            lag_2     = history[-2] if len(history) >= 2 else history[-1]
            rolling_3 = np.mean(history[-3:]) if len(history) >= 3 else np.mean(history)
            X_fut = pd.DataFrame([{
                'month_num': future_date.month,
                'year_num':  future_date.year,
                't':         t_counter,
                'lag_1':     lag_1,
                'lag_2':     lag_2,
                'rolling_3': rolling_3,
            }])
            pred = int(rf_reg.predict(X_fut)[0])
            future_dates.append(future_date)
            future_preds.append(pred)
            history.append(pred)
            t_counter += 1

        future_df = pd.DataFrame({
            'Month':               [d.strftime('%b %Y') for d in future_dates],
            'Projected Incidents': future_preds,
            'Quarter':             [d.quarter for d in future_dates],
            'Year':                [d.year for d in future_dates],
        })

        plot_style()
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.plot(monthly_clean['YearMonth_dt'], monthly_clean['Incident_Count'],
                 color='#38bdf8', linewidth=2, label='Historical', marker='o', markersize=3)
        ax3.fill_between(monthly_clean['YearMonth_dt'], monthly_clean['Incident_Count'],
                         alpha=0.1, color='#38bdf8')
        future_ts = pd.to_datetime([d.strftime('%Y-%m-%d') for d in future_dates])
        ax3.plot(future_ts, future_preds,
                 color='#f97316', linewidth=2, linestyle='--', label='Projected', marker='s', markersize=5)
        ax3.fill_between(future_ts, future_preds, alpha=0.15, color='#f97316')
        ax3.axvline(x=monthly_clean['YearMonth_dt'].iloc[-1], color='#475569',
                    linestyle=':', linewidth=1.5, label='Forecast start')
        ax3.set_ylabel("Incidents", fontsize=9)
        ax3.legend(fontsize=9)
        ax3.set_title(f"Historical + {n_future}-Month Forward Projection", fontsize=10, color='#94a3b8')
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        col_fq, col_fa = st.columns(2)
        with col_fq:
            st.markdown("<div class='section-title'>Projected — By Quarter</div>", unsafe_allow_html=True)
            fq = future_df.groupby(['Year','Quarter'])['Projected Incidents'].sum().reset_index()
            fq['Quarter'] = 'Q' + fq['Quarter'].astype(str)
            st.dataframe(fq, use_container_width=True, hide_index=True)
        with col_fa:
            st.markdown("<div class='section-title'>Projected — By Year</div>", unsafe_allow_html=True)
            fa = future_df.groupby('Year')['Projected Incidents'].sum().reset_index()
            st.dataframe(fa, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Month-by-Month Projection</div>", unsafe_allow_html=True)
        st.dataframe(future_df[['Month','Projected Incidents','Quarter','Year']],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Upload your ITSM CSV to run the forecasting model.")

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Project Summary — PRCL-0012</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div style='font-family:Space Mono,monospace; font-size:0.78rem; color:#38bdf8; margin-bottom:12px;'>
            BUSINESS CONTEXT
        </div>
        <p style='color:#94a3b8; font-size:0.88rem; line-height:1.7; margin:0;'>
            ABC Tech receives <strong style='color:#e2e8f0;'>22–25k IT incidents/year</strong> managed under the ITIL framework.
            Despite mature processes, a recent customer survey rated incident management as poor.
            This platform applies ML to predict priority, detect high-priority tickets before SLA breach,
            forecast incident volume for resource planning, and detect RFC-triggering tickets automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>All Tasks — Best Model Summary</div>", unsafe_allow_html=True)
    summary = pd.DataFrame([
        {"Task": "Task 1 — High Priority Detection", "Best Model": "XGBoost Baseline", "Accuracy": "0.98", "Primary Metric": "F1 Weighted: 0.98", "HP Recall": "0.58"},
        {"Task": "Task 2 — Volume Forecasting",      "Best Model": "RF Regressor",     "Accuracy": "—",    "Primary Metric": "R²: 0.31, MAE: 206", "HP Recall": "—"},
        {"Task": "Task 3 — Priority Auto-Tag",       "Best Model": "XGBoost Baseline", "Accuracy": "0.82", "Primary Metric": "F1 Macro: 0.71",     "HP Recall": "—"},
        {"Task": "Task 4 — RFC Prediction",          "Best Model": "Random Forest",    "Accuracy": "0.98", "Primary Metric": "F1 Macro: 0.68",     "HP Recall": "—"},
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    plot_style()
    st.markdown("<br><div class='section-title'>Task 3 — All Models Comparison (Accuracy & F1 Macro)</div>", unsafe_allow_html=True)

    models_list = ['LR (Base)', 'DT (Base)', 'RF (Base)', 'XGB (Base)★', 'DT (Tuned)', 'RF (Tuned)', 'XGB (Tuned)']
    acc_vals    = [0.67, 0.78, 0.80, 0.82, 0.66, 0.80, 0.80]
    f1_vals     = [0.56, 0.66, 0.68, 0.71, 0.61, 0.68, 0.69]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, title, color in [
        (axes[0], acc_vals, 'Accuracy', '#38bdf8'),
        (axes[1], f1_vals,  'F1 Macro', '#f97316'),
    ]:
        bar_colors = [color if '★' not in m else '#22c55e' for m in models_list]
        bars = ax.barh(models_list, vals, color=bar_colors, height=0.55)
        ax.set_xlim(0, 1.1)
        ax.set_title(title, fontsize=10, color='#94a3b8')
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=8, color='#94a3b8')
        patch = mpatches.Patch(color='#22c55e', label='Best Model')
        ax.legend(handles=[patch], fontsize=8, loc='lower right')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br><div class='section-title'>IT Manager Recommendations</div>", unsafe_allow_html=True)
    recs = [
        ("🚨", "P1/P2 Prevention",  "Deploy binary classifier at ticket creation. Auto-escalate and notify on-call engineers before SLA breach. Model catches 6 in 10 high-priority tickets."),
        ("🏷️", "Auto-Tag Priority",  "Route new tickets through XGBoost (0.82 acc) to auto-assign priority, eliminating manual tagging delays and reassignment cycles."),
        ("📅", "Resource Planning",  "Monthly volume forecast (MAE: 206 tickets) supports quarterly staffing decisions. Use for Q-on-Q headcount and infrastructure capacity planning."),
        ("🔧", "RFC Early Warning",  "RFC predictor (0.98 acc) flags tickets likely to trigger change requests — allows change managers to prepare ITIL RFC workflow proactively."),
        ("🔍", "CI Monitoring",      "CI_Name_freq is the top SHAP feature. High-frequency failing CIs should be flagged for proactive problem management and root-cause analysis."),
    ]
    for icon, title, desc in recs:
        st.markdown(f"""
        <div class='card' style='display:flex; gap:1rem; align-items:flex-start; margin-bottom:0.6rem;'>
            <div style='font-size:1.4rem; flex-shrink:0;'>{icon}</div>
            <div>
                <div style='font-weight:600; color:#e2e8f0; margin-bottom:4px;'>{title}</div>
                <div style='color:#64748b; font-size:0.83rem; line-height:1.6;'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("PRCL-0012 · ITSM ML Prediction System · DataMites™ · ABC Tech · Models: XGBoost + Random Forest + StandardScaler")

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — AI ASSISTANT (Gemini)
# ═══════════════════════════════════════════════════════════════════
with tab5:

    st.markdown("<div class='section-title'>AI-Powered Incident Assistant — Gemini 1.5 Flash</div>", unsafe_allow_html=True)

    def call_llm(prompt, system_prompt, max_tokens=400):
        try:
            import google.generativeai as genai
            api_key = st.secrets.get("GEMINI_API_KEY", None)
            if not api_key:
                return "Error: GEMINI_API_KEY not found in Streamlit secrets."
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    api_key_present = bool(st.secrets.get("GEMINI_API_KEY", None))

    if not api_key_present:
        st.markdown("""
        <div style='background:#1a1f2e; border:1px solid #f97316; border-radius:10px;
                    padding:1.2rem 1.5rem; color:#fdba74; font-size:0.88rem; line-height:1.8;'>
            <strong>⚙️ Setup Required — Gemini API Key</strong><br><br>
            To enable the AI Assistant:<br><br>
            <span style='font-family:Space Mono,monospace; font-size:0.82rem; color:#94a3b8;'>
            1. Go to aistudio.google.com → sign in with Google<br>
            2. Click Get API Key → Create API key<br>
            3. Go to Streamlit Cloud → your app → Settings → Secrets<br>
            4. Add: GEMINI_API_KEY = "AIzaSy_xxxx..."<br>
            5. Save and reboot the app
            </span>
        </div>
        """, unsafe_allow_html=True)

    else:

        # ── Section A — Ticket Summarizer ──────────────────────────
        st.markdown("""
        <div style='margin-bottom:1rem;'>
            <div style='font-family:Space Mono,monospace; font-size:1rem; font-weight:700;
                        color:#f1f5f9; margin-bottom:4px;'>🔍 Section A — Ticket Summarizer</div>
            <div style='color:#475569; font-size:0.82rem;'>
                Paste a raw incident description. The AI extracts a structured 3-line summary instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        raw_ticket = st.text_area(
            "Paste raw incident description here",
            height=140,
            placeholder="e.g. User reported that they cannot login to the SAP portal since 9am. "
                         "Multiple users in the Finance department are affected. The issue started "
                         "after last night's patch deployment. No error message shown, page just hangs.",
            key="raw_ticket_input"
        )

        summarize_btn = st.button("🔍 Summarize Ticket", key="summarize_btn")

        if summarize_btn:
            if not raw_ticket.strip():
                st.warning("Please paste an incident description first.")
            else:
                with st.spinner("Analysing incident..."):
                    system_sum = """You are an expert IT Service Management analyst.
When given a raw incident description, respond ONLY with exactly 3 lines in this format:
ISSUE: <one sentence describing what the problem is>
IMPACT: <one sentence describing who is affected and business impact>
URGENCY: <one sentence on urgency signal and recommended priority (P1/P2/P3/P4)>
No extra text, no preamble, no bullet symbols."""
                    summary = call_llm(raw_ticket, system_sum, max_tokens=200)
                    st.code(summary)

                import re
                lines = summary.strip().split("\n")
                lines_upper = [l.upper().strip().lstrip("*# ") for l in lines]
                raw_lines   = [l.strip().lstrip("*# ") for l in lines]
                issue_line   = next((raw_lines[i] for i, l in enumerate(lines_upper) if "ISSUE:" in l),   "—")
                impact_line  = next((raw_lines[i] for i, l in enumerate(lines_upper) if "IMPACT:" in l),  "—")
                urgency_line = next((raw_lines[i] for i, l in enumerate(lines_upper) if "URGENCY:" in l), "—")
                issue_line   = re.sub(r'(?i)^\*{0,2}issue\*{0,2}:\s*',   '', issue_line).strip()   or "—"
                impact_line  = re.sub(r'(?i)^\*{0,2}impact\*{0,2}:\s*',  '', impact_line).strip()  or "—"
                urgency_line = re.sub(r'(?i)^\*{0,2}urgency\*{0,2}:\s*', '', urgency_line).strip() or "—"

                st.markdown(f"""
                <div class='card' style='margin-top:0.8rem;'>
                    <div class='section-title' style='margin-bottom:12px;'>AI Summary</div>
                    <div style='display:flex; flex-direction:column; gap:10px;'>
                        <div style='background:#0f1928; border-left:3px solid #38bdf8; padding:10px 14px; border-radius:6px;'>
                            <span style='font-size:0.7rem; color:#38bdf8; text-transform:uppercase; letter-spacing:0.1em;'>Issue</span><br>
                            <span style='color:#e2e8f0; font-size:0.88rem;'>{issue_line}</span>
                        </div>
                        <div style='background:#0f1928; border-left:3px solid #f97316; padding:10px 14px; border-radius:6px;'>
                            <span style='font-size:0.7rem; color:#f97316; text-transform:uppercase; letter-spacing:0.1em;'>Impact</span><br>
                            <span style='color:#e2e8f0; font-size:0.88rem;'>{impact_line}</span>
                        </div>
                        <div style='background:#0f1928; border-left:3px solid #a855f7; padding:10px 14px; border-radius:6px;'>
                            <span style='font-size:0.7rem; color:#a855f7; text-transform:uppercase; letter-spacing:0.1em;'>Urgency Signal</span><br>
                            <span style='color:#e2e8f0; font-size:0.88rem;'>{urgency_line}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='margin:2rem 0; border-color:#1e293b;'>", unsafe_allow_html=True)

        # ── Section B — Auto-Response Suggester ────────────────────
        st.markdown("""
        <div style='margin-bottom:1rem;'>
            <div style='font-family:Space Mono,monospace; font-size:1rem; font-weight:700;
                        color:#f1f5f9; margin-bottom:4px;'>✉️ Section B — Auto-Response Suggester</div>
            <div style='color:#475569; font-size:0.82rem;'>
                Fill in the ticket context below. The AI generates a ready-to-send first-response
                email for your support engineer to copy instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            resp_issue    = st.text_input("Issue (one line)", placeholder="e.g. SAP portal login failure", key="resp_issue")
            resp_user     = st.text_input("Affected user / team", placeholder="e.g. Finance department users", key="resp_user")
            resp_priority = st.selectbox("Predicted Priority", ["P1 — Critical", "P2 — High", "P3 — Medium", "P4 — Low", "P5 — Very Low"], key="resp_priority")
        with col_b2:
            resp_engineer = st.text_input("Assigned engineer name", placeholder="e.g. Ravi Kumar", key="resp_engineer")
            resp_eta      = st.text_input("Estimated resolution time", placeholder="e.g. 2 hours / by 3:00 PM", key="resp_eta")
            resp_action   = st.text_input("First action being taken", placeholder="e.g. Checking patch rollback logs", key="resp_action")

        generate_btn = st.button("✉️ Generate First Response", key="generate_btn")

        if generate_btn:
            if not resp_issue.strip():
                st.warning("Please fill in the issue description at minimum.")
            else:
                with st.spinner("Drafting response..."):
                    system_resp = """You are an expert IT support communication specialist.
Write a professional, concise first-response email to the affected user(s).
Tone: calm, reassuring, action-oriented. Length: 4-6 sentences maximum.
Do NOT use bullet points. Do NOT add a subject line. Start directly with 'Dear [User/Team],'."""
                    user_prompt = f"""Write a first-response email for this IT incident:
Issue: {resp_issue}
Affected: {resp_user if resp_user else 'user'}
Priority: {resp_priority}
Assigned Engineer: {resp_engineer if resp_engineer else 'our support team'}
ETA: {resp_eta if resp_eta else 'being assessed'}
First action: {resp_action if resp_action else 'investigation in progress'}"""
                    response_text = call_llm(user_prompt, system_resp, max_tokens=300)

                st.markdown(f"""
                <div class='card' style='margin-top:0.8rem;'>
                    <div class='section-title' style='margin-bottom:12px;'>Generated First Response</div>
                    <div style='background:#0f1928; border:1px solid #1e3a5f; border-radius:8px;
                                padding:1.2rem 1.4rem; color:#cbd5e1; font-size:0.88rem;
                                line-height:1.8; white-space:pre-wrap; font-family:DM Sans,sans-serif;'>
{response_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style='font-size:0.75rem; color:#334155; margin-top:0.5rem;'>
                    💡 Review before sending. Edit as needed for your specific context.
                </div>
                """, unsafe_allow_html=True)
