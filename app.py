"""
Employee Attrition Prediction — Visual ML Dashboard
RISE Internship · Machine Learning & AI | Tamizhan Skills
Project 4: Industry-Oriented Machine Learning System for Employee Attrition Prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, pickle, io

# Pure NumPy ML — no sklearn/scipy needed (avoids BLAS DLL issues)
from ml_models import (
    train_test_split, cross_val_score, StratifiedKFold,
    StandardScaler, LabelEncoder,
    LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier,
    accuracy_score, recall_score, precision_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Employee Attrition Prediction | RISE",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
PAGES = [
    "🏠  Home",
    "📥  Load HR Data",
    "🔍  EDA & Employee Trends",
    "⚙️  Preprocess",
    "🤖  Train Models",
    "📊  Evaluate",
    "🎯  Predict Employee",
    "⚠️  At-Risk Employees",
]

PURPLE = "#7c3aed"
GREEN  = "#22c55e"
RED    = "#ef4444"
BLUE   = "#3b82f6"
GOLD   = "#f59e0b"
ORANGE = "#f97316"

DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
JOB_ROLES   = ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative",
                "Manager", "Sales Representative", "Research Director", "Human Resources"]

# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
_defs = dict(
    df_raw=None, df_proc=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    scaler=None, encoders=None, feature_cols=None,
    results=None, best_name=None,
    data_ready=False, preprocessed=False, trained=False,
)
for k, v in _defs.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="metric-container"]{background:#1e293b;border:1px solid #334155;
  border-radius:10px;padding:14px 18px;}
.sec-head{font-size:22px;font-weight:800;color:#c4b5fd;
  border-bottom:2px solid #4c1d95;padding-bottom:6px;margin-bottom:8px;}
.pill-ok{background:#166534;color:#bbf7d0;padding:3px 12px;
  border-radius:20px;font-size:12px;font-weight:700;}
.pill-no{background:#7f1d1d;color:#fecaca;padding:3px 12px;
  border-radius:20px;font-size:12px;font-weight:700;}
.pipe-wrap{display:flex;align-items:center;flex-wrap:wrap;margin:14px 0;}
.pipe-box{background:#2e1065;border:1px solid #7c3aed;border-radius:8px;
  padding:8px 14px;text-align:center;font-size:11px;color:#ddd6fe;font-weight:700;}
.pipe-arr{color:#7c3aed;font-size:20px;padding:0 6px;}
.insight-box{background:#1e1b4b;border:1px solid #4338ca;border-radius:8px;
  padding:14px 18px;margin:10px 0;font-size:14px;color:#c7d2fe;}
.warn-box{background:#2d1b00;border:1px solid #d97706;border-radius:8px;
  padding:14px 18px;margin:10px 0;font-size:14px;color:#fde68a;}
.risk-high{background:#450a0a;border:1px solid #dc2626;border-radius:8px;
  padding:12px 16px;margin:6px 0;font-size:14px;}
.risk-med{background:#2d1b00;border:1px solid #d97706;border-radius:8px;
  padding:12px 16px;margin:6px 0;font-size:14px;}
.step-box{background:#1e293b;border-left:4px solid #7c3aed;border-radius:6px;
  padding:12px 16px;margin:8px 0;font-size:14px;}
.brand{text-align:center;padding:8px 0 16px;border-bottom:1px solid #334155;margin-bottom:12px;}
.brand h2{color:#c4b5fd;font-size:17px;margin:0;}
.brand p{color:#94a3b8;font-size:11px;margin:4px 0 0;}
code{background:#334155;color:#c4b5fd;padding:2px 7px;border-radius:4px;font-size:13px;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def generate_hr_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic IBM-style HR employee dataset with realistic attrition patterns."""
    rng = np.random.default_rng(seed)

    dept        = rng.choice(DEPARTMENTS, n, p=[0.13, 0.57, 0.30])
    job_role    = rng.choice(JOB_ROLES,   n)
    gender      = rng.choice(["Male", "Female"], n, p=[0.60, 0.40])
    marital     = rng.choice(["Single", "Married", "Divorced"], n, p=[0.32, 0.46, 0.22])
    overtime    = rng.choice(["Yes", "No"], n, p=[0.28, 0.72])
    edu_field   = rng.choice(["Life Sciences", "Medical", "Marketing",
                               "Technical Degree", "Human Resources", "Other"],
                              n, p=[0.41, 0.27, 0.15, 0.09, 0.05, 0.03])
    education   = rng.integers(1, 6, n)       # 1=Below College .. 5=Doctor
    env_sat     = rng.integers(1, 5, n)        # 1=Low .. 4=Very High
    job_sat     = rng.integers(1, 5, n)
    job_inv     = rng.integers(1, 5, n)
    job_lvl     = rng.integers(1, 6, n)
    rel_sat     = rng.integers(1, 5, n)
    stock       = rng.integers(0, 4, n)
    wlb         = rng.integers(1, 5, n)        # Work-Life Balance
    perf_rating = rng.integers(3, 5, n)        # 3=Excellent 4=Outstanding

    age               = rng.integers(18, 61, n)
    monthly_income    = np.round(rng.uniform(1000, 20000, n), 0).astype(int)
    num_companies     = rng.integers(0, 10, n)
    pct_salary_hike   = rng.integers(11, 26, n)
    total_working_yrs = rng.integers(0, 41, n)
    training_times    = rng.integers(0, 7, n)
    yrs_at_company    = rng.integers(0, 41, n)
    yrs_curr_role     = np.clip(rng.integers(0, 19, n), 0, yrs_at_company)
    yrs_last_promo    = np.clip(rng.integers(0, 16, n), 0, yrs_at_company)
    yrs_curr_mgr      = np.clip(rng.integers(0, 18, n), 0, yrs_at_company)

    # Attrition probability — based on IBM HR Analytics real patterns
    p = np.full(n, 0.04)
    p += (overtime   == "Yes")               * 0.20
    p += (job_sat    <= 2)                   * 0.14
    p += (env_sat    <= 2)                   * 0.09
    p += (wlb        == 1)                   * 0.12
    p += (yrs_at_company < 3)               * 0.14
    p += (monthly_income < 3000)             * 0.10
    p += (marital    == "Single")            * 0.08
    p += (dept       == "Sales")             * 0.05
    p += (job_inv    <= 2)                   * 0.07
    p += (stock      == 0)                   * 0.05
    p += (num_companies > 5)                 * 0.05
    p += (yrs_last_promo > 6)               * 0.06
    p -= (yrs_at_company > 15)              * 0.08
    p -= (job_lvl    >= 3)                   * 0.05
    p -= (job_sat    == 4)                   * 0.05
    p  = np.clip(p, 0.01, 0.95)
    attrition = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "EmployeeID":             [f"EMP-{i+1:04d}" for i in range(n)],
        "Age":                    age,
        "Department":             dept,
        "EducationField":         edu_field,
        "Education":              education,
        "Gender":                 gender,
        "JobRole":                job_role,
        "JobLevel":               job_lvl,
        "MaritalStatus":          marital,
        "OverTime":               overtime,
        "EnvironmentSatisfaction":env_sat,
        "JobInvolvement":         job_inv,
        "JobSatisfaction":        job_sat,
        "RelationshipSatisfaction":rel_sat,
        "WorkLifeBalance":        wlb,
        "PerformanceRating":      perf_rating,
        "StockOptionLevel":       stock,
        "MonthlyIncome":          monthly_income,
        "NumCompaniesWorked":     num_companies,
        "PercentSalaryHike":      pct_salary_hike,
        "TotalWorkingYears":      total_working_yrs,
        "TrainingTimesLastYear":  training_times,
        "YearsAtCompany":         yrs_at_company,
        "YearsInCurrentRole":     yrs_curr_role,
        "YearsSinceLastPromotion":yrs_last_promo,
        "YearsWithCurrManager":   yrs_curr_mgr,
        "Attrition":              attrition,
    })


def preprocess_df(df: pd.DataFrame):
    data = df.drop(columns=["EmployeeID", "Attrition"], errors="ignore").copy()
    y    = df["Attrition"].values
    cat_cols = data.select_dtypes("object").columns.tolist()
    num_cols = data.select_dtypes(exclude="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data.values, y, scaler, encoders, data.columns.tolist()


def fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return buf.getvalue()


def pill(ok):
    c, t = ("pill-ok","Ready") if ok else ("pill-no","Not ready")
    return f'<span class="{c}">{t}</span>'


def dark_ax(ax):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_visible(False)
    return ax


def dark_fig(nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, **kw)
    fig.patch.set_facecolor("#1e293b")
    flat = np.array(axes).flat if hasattr(axes, "__iter__") else [axes]
    for ax in flat:
        dark_ax(ax)
    return fig, axes


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div class="brand">
      <h2>👥 Attrition Prediction</h2>
      <p>RISE Internship · Tamizhan Skills</p>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Pipeline Status**")
    st.markdown(
        f"Data &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{pill(st.session_state.data_ready)}<br>"
        f"Preprocessed {pill(st.session_state.preprocessed)}<br>"
        f"Trained &nbsp;&nbsp;&nbsp;&nbsp;{pill(st.session_state.trained)}",
        unsafe_allow_html=True,
    )

    if st.session_state.data_ready and st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.markdown("---")
        attr_rate = df["Attrition"].mean() * 100
        st.metric("Total Employees", f"{len(df):,}")
        st.metric("Attrition Rate",  f"{attr_rate:.1f}%")

    if st.session_state.trained and st.session_state.best_name:
        st.markdown("---")
        best = st.session_state.results[st.session_state.best_name]
        st.metric("Best Model",    st.session_state.best_name)
        st.metric("Test Recall",   f"{best['recall']*100:.1f}%")
        st.metric("Test Accuracy", f"{best['accuracy']*100:.1f}%")

    st.markdown("---")
    st.caption("Uttarakhand Government institute of Polytechnic Nainital · IT Dept")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1: HOME
# ════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown('<div class="sec-head">👥 Employee Attrition Prediction Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown("**Project 4 — RISE Internship · Machine Learning & AI | Tamizhan Skills**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project Type",    "Binary Classification")
    c2.metric("Target Variable", "Attrition (Yes/No)")
    c3.metric("Key Metrics",     "Accuracy + Recall")
    c4.metric("HR Features",     "25 Features")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Problem Statement")
        st.markdown("""<div class="insight-box">
        Organizations face high costs due to employee attrition and struggle to identify
        factors that cause employees to leave. Manual HR analysis is ineffective for large
        workforces. Industry uses machine learning to <b>predict attrition</b> and
        improve <b>retention strategies</b>.
        </div>""", unsafe_allow_html=True)
        st.subheader("🎯 Objective")
        st.info("Build a machine learning model that predicts **employee attrition** "
                "based on HR and performance data.")

    with col2:
        st.subheader("📋 Project Requirements")
        for r in ["✅ HR dataset ingestion",
                  "✅ Data cleaning and preprocessing",
                  "✅ Feature engineering for attrition factors",
                  "✅ Exploratory data analysis",
                  "✅ Machine learning model training",
                  "✅ Model evaluation using accuracy and recall",
                  "✅ Identification of key attrition drivers",
                  "✅ Visualization of employee trends",
                  "✅ Model documentation"]:
            st.markdown(r)

    st.markdown("---")
    st.subheader("🔗 ML Pipeline")
    st.markdown("""<div class="pipe-wrap">
      <div class="pipe-box">📥<br>Load HR<br>Data</div><div class="pipe-arr">›</div>
      <div class="pipe-box">🔍<br>EDA &<br>Trends</div><div class="pipe-arr">›</div>
      <div class="pipe-box">⚙️<br>Feature<br>Engineering</div><div class="pipe-arr">›</div>
      <div class="pipe-box">🤖<br>Train<br>Models</div><div class="pipe-arr">›</div>
      <div class="pipe-box">📊<br>Evaluate<br>Recall+Acc</div><div class="pipe-arr">›</div>
      <div class="pipe-box">🔑<br>Attrition<br>Drivers</div><div class="pipe-arr">›</div>
      <div class="pipe-box">⚠️<br>At-Risk<br>Employees</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛠️ Tools")
        for t, u in [("Python","Core language"),("NumPy","Numerical computing"),
                     ("Pandas","Data manipulation"),("Matplotlib / Seaborn","Visualisation"),
                     ("Scikit-learn","ML models & evaluation"),("Streamlit","This dashboard")]:
            st.markdown(f"- **{t}** — {u}")
    with col2:
        st.subheader("🏆 Expected Outcomes")
        for o in ["Hands-on experience with HR analytics use cases",
                  "Understanding of workforce-related ML problems",
                  "Industry-ready ML portfolio project",
                  "Identification of key attrition drivers for HR strategy",
                  "Job relevance: ML Engineer, HR Analyst, Data Scientist"]:
            st.markdown(f"✅ {o}")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2: LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
def page_load():
    st.markdown('<div class="sec-head">📥 Load HR Dataset</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎲 Generate Synthetic HR Data", "📂 Upload CSV"])

    with tab1:
        st.markdown("""<div class="insight-box">
        Generates a <b>realistic IBM-style HR employee dataset</b> — no download required.<br>
        Includes satisfaction scores, job details, compensation, tenure, and overtime —
        all key attrition drivers based on real HR research.<br>
        Attrition generated using business-realistic probability rules (~17% rate).
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            n_emp = st.slider("Number of employees", 200, 5000, 1000, 100)
            seed  = st.number_input("Random seed", 0, 9999, 42)
        with col2:
            st.markdown("**Key HR Features Included:**")
            st.markdown("- Age, Gender, MaritalStatus, Department\n"
                        "- JobRole, JobLevel, Education, EducationField\n"
                        "- MonthlyIncome, PercentSalaryHike, StockOptionLevel\n"
                        "- OverTime, WorkLifeBalance, JobSatisfaction\n"
                        "- EnvironmentSatisfaction, JobInvolvement\n"
                        "- YearsAtCompany, YearsSinceLastPromotion\n"
                        "- **Attrition** (target: 0=Stays, 1=Leaves)")

        if st.button("🚀 Generate HR Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating employee records..."):
                df = generate_hr_data(n_emp, seed)
            st.session_state.df_raw       = df
            st.session_state.data_ready   = True
            st.session_state.preprocessed = False
            st.session_state.trained      = False
            attr_pct = df["Attrition"].mean() * 100
            a, b, c, d = st.columns(4)
            a.metric("Total Employees", f"{len(df):,}")
            b.metric("Left (Attrition)", f"{df['Attrition'].sum():,}", f"{attr_pct:.1f}%")
            c.metric("Stayed",  f"{(df['Attrition']==0).sum():,}", f"{100-attr_pct:.1f}%")
            d.metric("HR Features", f"{df.shape[1]-2}")
            st.success("✅ HR dataset ready! Proceed to EDA & Employee Trends →")

    with tab2:
        uploaded = st.file_uploader("Upload HR CSV (must include 'Attrition' column, 0=Stays 1=Leaves)",
                                     type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if "Attrition" not in df.columns:
                    # Try to encode Yes/No
                    if df.select_dtypes("object").apply(lambda c: c.str.lower().isin(["yes","no"]).any()).any():
                        for col in df.select_dtypes("object").columns:
                            if df[col].str.lower().isin(["yes","no"]).all():
                                df[col] = (df[col].str.lower() == "yes").astype(int)
                if "Attrition" not in df.columns:
                    st.error("CSV must contain 'Attrition' column.")
                else:
                    st.session_state.df_raw       = df
                    st.session_state.data_ready   = True
                    st.session_state.preprocessed = False
                    st.session_state.trained      = False
                    st.success(f"✅ Uploaded: {df.shape[0]} rows × {df.shape[1]} cols")
                    st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.data_ready:
        st.markdown("---")
        st.subheader("📋 Dataset Preview")
        df = st.session_state.df_raw
        st.dataframe(df.head(10), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data Types**")
            st.dataframe(df.dtypes.rename("dtype").reset_index(),
                         use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Basic Statistics (Numeric)**")
            st.dataframe(df.select_dtypes("number").describe().round(1),
                         use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3: EDA
# ════════════════════════════════════════════════════════════════════════════
def page_eda():
    st.markdown('<div class="sec-head">🔍 EDA & Employee Trends</div>', unsafe_allow_html=True)
    if not st.session_state.data_ready:
        st.warning("⚠️ Load HR data first."); return

    df = st.session_state.df_raw
    attr_rate = df["Attrition"].mean() * 100

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Attrition Overview",
        "🏢 Department & Role Analysis",
        "😊 Satisfaction & Work-Life",
        "💰 Compensation Analysis",
        "📅 Tenure & Experience",
    ])

    # ── TAB 1 ──────────────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Attrition Rate", f"{attr_rate:.1f}%")
        c2.metric("Employees Left",  f"{df['Attrition'].sum():,}")
        c3.metric("Employees Stayed",f"{(df['Attrition']==0).sum():,}")
        c4.metric("Industry Avg Attrition", "~15–18%")

        fig, axes = dark_fig(1, 2, figsize=(12, 4))
        ax1, ax2 = axes

        sizes  = [df["Attrition"].sum(), (df["Attrition"]==0).sum()]
        ax1.pie(sizes, labels=["Left", "Stayed"], colors=[RED, GREEN],
                autopct="%1.1f%%", startangle=90,
                textprops={"color":"#e2e8f0","fontsize":12,"fontweight":"bold"},
                wedgeprops={"edgecolor":"#0f172a","linewidth":2})
        ax1.set_title("Attrition Distribution", color="#e2e8f0", fontsize=13, fontweight="bold")

        ot = df.groupby("OverTime")["Attrition"].mean() * 100
        bars = ax2.bar(ot.index, ot.values,
                       color=[RED if v > 20 else GREEN for v in ot.values],
                       edgecolor="#0f172a", width=0.4, alpha=0.9)
        for bar, v in zip(bars, ot.values):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold", color="#e2e8f0")
        ax2.set_title("Attrition Rate: OverTime vs No OverTime",
                      color="#e2e8f0", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Attrition Rate (%)", color="#94a3b8")
        ax2.set_ylim(0, max(ot.values)*1.3)
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download Chart", fig_bytes(fig), "attrition_overview.png", "image/png")
        plt.close(fig)
        st.markdown("""<div class="insight-box">
        💡 <b>Key Insight:</b> Employees working overtime have a dramatically higher attrition rate.
        OverTime is consistently the #1 attrition predictor in HR datasets.
        </div>""", unsafe_allow_html=True)

    # ── TAB 2 ──────────────────────────────────────────────────────────────
    with tab2:
        fig, axes = dark_fig(1, 2, figsize=(14, 5))
        ax1, ax2 = axes

        dept_attr = df.groupby("Department")["Attrition"].mean() * 100
        clrs = [RED if v > 20 else GOLD if v > 14 else GREEN for v in dept_attr.values]
        b1 = ax1.bar(dept_attr.index, dept_attr.values, color=clrs,
                     edgecolor="#0f172a", width=0.5, alpha=0.9)
        for bar, v in zip(b1, dept_attr.values):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color="#e2e8f0")
        ax1.set_title("Attrition Rate by Department", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Attrition Rate (%)", color="#94a3b8")
        ax1.set_xticklabels(dept_attr.index, rotation=20, ha="right", fontsize=9)
        ax1.set_ylim(0, max(dept_attr.values)*1.3)

        role_attr = df.groupby("JobRole")["Attrition"].mean().sort_values(ascending=True) * 100
        clrs2 = [RED if v > 25 else GOLD if v > 15 else GREEN for v in role_attr.values]
        ax2.barh(role_attr.index, role_attr.values, color=clrs2, edgecolor="#0f172a", alpha=0.9)
        ax2.set_title("Attrition Rate by Job Role", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Attrition Rate (%)", color="#94a3b8")
        ax2.tick_params(axis="y", labelsize=9)

        fig.suptitle("Department & Role Analysis", color="#e2e8f0", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download Chart", fig_bytes(fig), "dept_role_analysis.png", "image/png")
        plt.close(fig)

        fig2, axes2 = dark_fig(1, 2, figsize=(12, 4))
        ms = df.groupby("MaritalStatus")["Attrition"].mean() * 100
        gender_a = df.groupby("Gender")["Attrition"].mean() * 100
        axes2[0].bar(ms.index, ms.values, color=[RED, GOLD, GREEN][:len(ms)],
                     edgecolor="#0f172a", width=0.4, alpha=0.9)
        axes2[0].set_title("Attrition by Marital Status", color="#e2e8f0", fontsize=11, fontweight="bold")
        axes2[0].set_ylabel("Attrition Rate (%)", color="#94a3b8")
        for bar, v in zip(axes2[0].patches, ms.values):
            axes2[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                          f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color="#e2e8f0")
        axes2[0].set_ylim(0, max(ms.values)*1.3)

        axes2[1].bar(gender_a.index, gender_a.values, color=[BLUE, PURPLE],
                     edgecolor="#0f172a", width=0.4, alpha=0.9)
        axes2[1].set_title("Attrition by Gender", color="#e2e8f0", fontsize=11, fontweight="bold")
        axes2[1].set_ylabel("Attrition Rate (%)", color="#94a3b8")
        for bar, v in zip(axes2[1].patches, gender_a.values):
            axes2[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                          f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color="#e2e8f0")
        axes2[1].set_ylim(0, max(gender_a.values)*1.3)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── TAB 3 ──────────────────────────────────────────────────────────────
    with tab3:
        sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction",
                    "WorkLifeBalance", "JobInvolvement", "RelationshipSatisfaction"]
        sat_lbls = ["Job Satisfaction", "Environment Satisfaction",
                    "Work-Life Balance", "Job Involvement", "Relationship Satisfaction"]
        sat_cols = [c for c in sat_cols if c in df.columns]
        sat_lbls = sat_lbls[:len(sat_cols)]

        fig, axes = dark_fig(1, len(sat_cols), figsize=(4*len(sat_cols), 4))
        if len(sat_cols) == 1: axes = [axes]
        for ax, col, lbl in zip(axes, sat_cols, sat_lbls):
            attr_by = df.groupby(col)["Attrition"].mean() * 100
            clrs = [RED if v > 20 else GOLD if v > 12 else GREEN for v in attr_by.values]
            ax.bar(attr_by.index, attr_by.values, color=clrs, edgecolor="#0f172a", width=0.6, alpha=0.9)
            ax.set_title(lbl, color="#e2e8f0", fontsize=10, fontweight="bold")
            ax.set_xlabel("Score (1=Low, 4=High)", color="#94a3b8", fontsize=8)
            ax.set_ylabel("Attrition %", color="#94a3b8")
            ax.set_ylim(0, max(attr_by.values)*1.3)
            for bar, v in zip(ax.patches, attr_by.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#e2e8f0")

        fig.suptitle("Attrition Rate by Satisfaction Scores",
                     color="#e2e8f0", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download Chart", fig_bytes(fig), "satisfaction_analysis.png", "image/png")
        plt.close(fig)
        st.markdown("""<div class="insight-box">
        💡 <b>Insight:</b> Employees with low Job Satisfaction (1) and low Environment
        Satisfaction (1) show the highest attrition rates — these are critical HR intervention points.
        </div>""", unsafe_allow_html=True)

    # ── TAB 4 ──────────────────────────────────────────────────────────────
    with tab4:
        fig, axes = dark_fig(1, 2, figsize=(13, 4))
        ax1, ax2 = axes

        ax1.hist(df[df["Attrition"]==0]["MonthlyIncome"], bins=30, alpha=0.7,
                 color=GREEN, label="Stayed", edgecolor="#0f172a")
        ax1.hist(df[df["Attrition"]==1]["MonthlyIncome"], bins=30, alpha=0.7,
                 color=RED,   label="Left",   edgecolor="#0f172a")
        ax1.set_title("Monthly Income Distribution", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Monthly Income ($)", color="#94a3b8")
        ax1.set_ylabel("Count", color="#94a3b8")
        ax1.legend(fontsize=10)

        income_bands = pd.cut(df["MonthlyIncome"],
                               bins=[0,3000,6000,10000,20000],
                               labels=["<3K","3K-6K","6K-10K",">10K"])
        band_attr = df.groupby(income_bands, observed=True)["Attrition"].mean() * 100
        clrs = [RED if v > 20 else GOLD if v > 12 else GREEN for v in band_attr.values]
        ax2.bar(band_attr.index.astype(str), band_attr.values, color=clrs,
                edgecolor="#0f172a", width=0.5, alpha=0.9)
        for bar, v in zip(ax2.patches, band_attr.values):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color="#e2e8f0")
        ax2.set_title("Attrition Rate by Income Band", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Attrition Rate (%)", color="#94a3b8")
        ax2.set_ylim(0, max(band_attr.values)*1.3)

        fig.suptitle("Compensation Analysis", color="#e2e8f0", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download Chart", fig_bytes(fig), "compensation_analysis.png", "image/png")
        plt.close(fig)
        st.info("💡 Employees earning < $3,000/month have significantly higher attrition — "
                "salary review for low-band employees should be a priority.")

    # ── TAB 5 ──────────────────────────────────────────────────────────────
    with tab5:
        fig, axes = dark_fig(1, 2, figsize=(13, 4))
        ax1, ax2 = axes

        ax1.hist(df[df["Attrition"]==0]["YearsAtCompany"], bins=25, alpha=0.7,
                 color=GREEN, label="Stayed", edgecolor="#0f172a")
        ax1.hist(df[df["Attrition"]==1]["YearsAtCompany"], bins=25, alpha=0.7,
                 color=RED,   label="Left",   edgecolor="#0f172a")
        ax1.set_title("Years at Company", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Years", color="#94a3b8")
        ax1.set_ylabel("Count", color="#94a3b8")
        ax1.legend(fontsize=10)

        ax2.hist(df[df["Attrition"]==0]["Age"], bins=20, alpha=0.7,
                 color=GREEN, label="Stayed", edgecolor="#0f172a")
        ax2.hist(df[df["Attrition"]==1]["Age"], bins=20, alpha=0.7,
                 color=RED,   label="Left",   edgecolor="#0f172a")
        ax2.set_title("Age Distribution", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Age (years)", color="#94a3b8")
        ax2.set_ylabel("Count", color="#94a3b8")
        ax2.legend(fontsize=10)

        fig.suptitle("Tenure & Age Trends", color="#e2e8f0", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download Chart", fig_bytes(fig), "tenure_age_trends.png", "image/png")
        plt.close(fig)
        st.info("💡 New employees (< 3 years) and younger employees show highest attrition — "
                "early-career onboarding and mentoring programs are critical.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4: PREPROCESS
# ════════════════════════════════════════════════════════════════════════════
def page_preprocess():
    st.markdown('<div class="sec-head">⚙️ Feature Engineering & Preprocessing</div>',
                unsafe_allow_html=True)
    if not st.session_state.data_ready:
        st.warning("⚠️ Load HR data first."); return

    df = st.session_state.df_raw
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Preprocessing Steps")
        st.markdown("""
**Step 1 — Drop Employee ID**  
`EmployeeID` is an identifier, not a predictor — removed.

**Step 2 — Label Encoding**  
Converts text to numbers:
- `Department` (HR→0, R&D→1, Sales→2)
- `OverTime` (No→0, Yes→1)
- `MaritalStatus`, `Gender`, `JobRole`, `EducationField`

**Step 3 — StandardScaler**  
Normalises all numeric features (Age, Income, Tenure...)
to mean=0, std=1 — essential for SVM and KNN.

**Step 4 — Stratified Train/Test Split**  
Maintains the ~17% attrition ratio in both train and test sets.
""")

    with col2:
        st.subheader("⚙️ Configure Split")
        test_sz = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
        rs      = st.number_input("Random state", 0, 99, 42)
        cat_cols = df.select_dtypes("object").drop(
            columns=["EmployeeID"] if "EmployeeID" in df.columns else [], errors="ignore"
        ).columns.tolist()
        num_cols = df.select_dtypes(include=np.number).drop(
            columns=["Attrition"], errors="ignore").columns.tolist()
        st.markdown(f"**Categorical columns to encode ({len(cat_cols)}):**")
        st.code(", ".join(cat_cols))
        st.markdown(f"**Numerical columns to scale ({len(num_cols)}):**")
        st.code(str(len(num_cols)) + " columns")

    if st.button("✅ Apply Feature Engineering & Split", type="primary", use_container_width=True):
        with st.spinner("Preprocessing..."):
            X, y, scaler, encoders, feat_names = preprocess_df(df)
            X_train, X_test, y_train, y_test   = train_test_split(
                X, y, test_size=test_sz, random_state=rs, stratify=y)
        st.session_state.X_train      = X_train
        st.session_state.X_test       = X_test
        st.session_state.y_train      = y_train
        st.session_state.y_test       = y_test
        st.session_state.scaler       = scaler
        st.session_state.encoders     = encoders
        st.session_state.feature_cols = feat_names
        st.session_state.preprocessed = True
        st.session_state.trained      = False
        a, b, c, d = st.columns(4)
        a.metric("Train samples",  f"{len(X_train):,}")
        b.metric("Test samples",   f"{len(X_test):,}")
        c.metric("Train attrition", f"{y_train.mean()*100:.1f}%")
        d.metric("Features",       f"{X_train.shape[1]}")
        st.success("✅ Preprocessing done! Proceed to Train Models →")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 5: TRAIN
# ════════════════════════════════════════════════════════════════════════════
def page_train():
    st.markdown('<div class="sec-head">🤖 Train Models</div>', unsafe_allow_html=True)
    if not st.session_state.preprocessed:
        st.warning("⚠️ Preprocess data first."); return

    st.subheader("⚙️ Model Selection & Tuning")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        use_lr = st.checkbox("Logistic Regression", True)
        lr_c   = st.slider("LR — C", 0.01, 10.0, 1.0, disabled=not use_lr)
    with col2:
        use_rf = st.checkbox("Random Forest", True)
        rf_n   = st.slider("RF — Trees", 50, 300, 100, 50, disabled=not use_rf)
    with col3:
        use_svm = st.checkbox("SVM (RBF)", True)
        svm_c   = st.slider("SVM — C", 0.1, 10.0, 1.0, disabled=not use_svm)
    with col4:
        use_knn = st.checkbox("KNN", True)
        knn_k   = st.slider("KNN — k", 1, 21, 5, 2, disabled=not use_knn)

    cv_k = st.slider("Cross-validation folds", 2, 10, 5)

    st.markdown("""<div class="insight-box">
    💡 <b>Why Recall is the primary metric:</b> Missing an employee who is about to leave
    (false negative) means losing training investment, institutional knowledge, and paying
    recruitment costs. It is more expensive to miss an at-risk employee than to
    offer unnecessary retention incentives (false positive).
    </div>""", unsafe_allow_html=True)

    if not (use_lr or use_rf or use_svm or use_knn):
        st.error("Select at least one model."); return

    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        mdls = {}
        if use_lr:  mdls["Logistic Regression"] = LogisticRegression(
                        C=lr_c, class_weight="balanced", random_state=42, max_iter=500)
        if use_rf:  mdls["Random Forest"]       = RandomForestClassifier(
                        n_estimators=rf_n, class_weight="balanced", random_state=42, n_jobs=-1)
        if use_svm: mdls["SVM (RBF)"]           = SVC(
                        kernel="rbf", C=svm_c, class_weight="balanced",
                        probability=True, random_state=42)
        if use_knn: mdls["KNN"]                 = KNeighborsClassifier(n_neighbors=knn_k)

        X_train = st.session_state.X_train; X_test = st.session_state.X_test
        y_train = st.session_state.y_train; y_test = st.session_state.y_test
        results, rows = {}, []
        skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
        bar = st.progress(0, text="Training...")

        for i, (name, model) in enumerate(mdls.items()):
            bar.progress(i/len(mdls), text=f"Training {name}...")
            cv = cross_val_score(model, X_train, y_train, cv=skf, scoring="recall")
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else None
            results[name] = {"model":model,"cv_recall":cv.mean(),"cv_std":cv.std(),
                             "accuracy":accuracy_score(y_test,y_pred),
                             "recall":recall_score(y_test,y_pred),
                             "precision":precision_score(y_test,y_pred),
                             "f1":f1_score(y_test,y_pred),
                             "roc_auc":roc_auc_score(y_test,y_proba) if y_proba is not None else None,
                             "y_pred":y_pred,"y_proba":y_proba}
            rows.append({"Model":name,
                         "CV Recall":f"{cv.mean()*100:.2f}%","CV Std":f"±{cv.std()*100:.2f}%",
                         "Accuracy":f"{results[name]['accuracy']*100:.2f}%",
                         "Recall":f"{results[name]['recall']*100:.2f}%",
                         "Precision":f"{results[name]['precision']*100:.2f}%",
                         "F1":f"{results[name]['f1']*100:.2f}%",
                         "ROC-AUC":f"{results[name]['roc_auc']*100:.2f}%" if results[name]['roc_auc'] else "N/A"})

        bar.progress(1.0, text="Done!")
        best_name = max(results, key=lambda k: results[k]["recall"])
        st.session_state.results   = results
        st.session_state.best_name = best_name
        st.session_state.trained   = True
        st.success(f"✅ Complete! Best Recall: **{best_name}** "
                   f"({results[best_name]['recall']*100:.1f}%)")
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        st.balloons()


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 6: EVALUATE
# ════════════════════════════════════════════════════════════════════════════
def page_evaluate():
    st.markdown('<div class="sec-head">📊 Model Evaluation</div>', unsafe_allow_html=True)
    if not st.session_state.trained:
        st.warning("⚠️ Train models first."); return

    results   = st.session_state.results
    best_name = st.session_state.best_name
    y_test    = st.session_state.y_test
    feat_cols = st.session_state.feature_cols
    best      = results[best_name]

    cols = st.columns(len(results))
    for cs, (name, res) in zip(cols, results.items()):
        cs.metric(name, f"Recall {res['recall']*100:.1f}%",
                  f"Acc {res['accuracy']*100:.1f}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️ Confusion Matrices",
        "📈 Metrics Comparison",
        "📉 ROC Curves",
        "🔑 Attrition Drivers",
        "💾 Download Model",
    ])

    with tab1:
        n = len(results)
        fig, axes = dark_fig(1, n, figsize=(6*n, 4))
        if n == 1: axes = [axes]
        for ax, (name, res) in zip(axes, results.items()):
            cm = confusion_matrix(y_test, res["y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", ax=ax,
                        xticklabels=["Stayed","Left"],yticklabels=["Stayed","Left"],
                        linewidths=0.5, cbar=False, annot_kws={"size":14,"weight":"bold"})
            star = " ★" if name==best_name else ""
            ax.set_title(f"{name}{star}\nRecall: {res['recall']*100:.1f}%  Acc: {res['accuracy']*100:.1f}%",
                         fontweight="bold",fontsize=11,
                         color="#c4b5fd" if name==best_name else "#94a3b8")
            ax.set_xlabel("Predicted",color="#94a3b8"); ax.set_ylabel("Actual",color="#94a3b8")
            ax.tick_params(colors="#94a3b8",rotation=0)
        fig.suptitle("Confusion Matrices  (Left = Attrition=1)",
                     color="#e2e8f0",fontsize=13,fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download PNG", fig_bytes(fig), "confusion_matrices.png", "image/png")
        plt.close(fig)

    with tab2:
        names = list(results.keys())
        mets  = ["accuracy","recall","precision","f1"]
        mlbls = ["Accuracy","Recall","Precision","F1 Score"]
        mclrs = [BLUE,RED,GOLD,PURPLE]
        bi    = names.index(best_name)
        fig, axes = dark_fig(1, 4, figsize=(16, 4))
        for ax, met, lbl, col in zip(axes, mets, mlbls, mclrs):
            vals = [results[n][met]*100 for n in names]
            bars = ax.bar(names, vals, color=col, edgecolor="#0f172a", width=0.5, alpha=0.88)
            bars[bi].set_edgecolor(GOLD); bars[bi].set_linewidth(3)
            ax.set_title(lbl, color="#e2e8f0", fontweight="bold", fontsize=11)
            ax.set_ylabel("%", color="#94a3b8")
            ax.set_ylim(max(0,min(vals)-15), 105)
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#e2e8f0")
        fig.suptitle("Model Performance Comparison", color="#e2e8f0", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download PNG", fig_bytes(fig), "metrics_comparison.png", "image/png")
        plt.close(fig)

    with tab3:
        fig, ax = dark_fig(figsize=(8, 5))
        roc_c = [BLUE,GREEN,RED,PURPLE,GOLD]
        for (name, res), col in zip(results.items(), roc_c):
            if res["y_proba"] is not None:
                fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
                ax.plot(fpr, tpr, color=col, linewidth=2.5 if name==best_name else 1.5,
                        label=f"{name}  (AUC={res['roc_auc']:.3f})")
        ax.plot([0,1],[0,1],"k--",linewidth=1,alpha=0.4,label="Random Guess")
        ax.set_xlabel("False Positive Rate",color="#94a3b8")
        ax.set_ylabel("True Positive Rate (Recall)",color="#94a3b8")
        ax.set_title("ROC Curves — All Models",color="#e2e8f0",fontsize=13,fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download PNG", fig_bytes(fig), "roc_curves.png", "image/png")
        plt.close(fig)

    with tab4:
        st.subheader("🔑 Key Attrition Drivers (Feature Importance)")
        if "Random Forest" in results:
            rf_model = results["Random Forest"]["model"]
            fi = pd.DataFrame({"Feature":feat_cols,
                               "Importance":rf_model.feature_importances_})
            fi = fi.sort_values("Importance", ascending=True).tail(15)
            fig, ax = dark_fig(figsize=(9, 6))
            clrs = [RED if v > fi["Importance"].quantile(0.75) else
                    GOLD if v > fi["Importance"].quantile(0.5) else BLUE
                    for v in fi["Importance"]]
            ax.barh(fi["Feature"], fi["Importance"], color=clrs, edgecolor="#0f172a", alpha=0.9)
            ax.set_title("Top 15 Attrition Drivers — Random Forest",
                         color="#e2e8f0", fontsize=12, fontweight="bold")
            ax.set_xlabel("Feature Importance", color="#94a3b8")
            legend = [mpatches.Patch(color=RED,label="High importance"),
                      mpatches.Patch(color=GOLD,label="Medium importance"),
                      mpatches.Patch(color=BLUE,label="Lower importance")]
            ax.legend(handles=legend, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            st.download_button("⬇️ Download PNG", fig_bytes(fig),
                               "attrition_drivers.png", "image/png")
            plt.close(fig)
            top5 = fi.sort_values("Importance",ascending=False).head(5)
            st.markdown("**Top 5 Attrition Drivers:**")
            for _, row in top5.iterrows():
                st.markdown(f"- **{row['Feature']}** — importance: `{row['Importance']:.4f}`")
        else:
            st.info("Train with Random Forest selected to see feature importance.")

    with tab5:
        buf = io.BytesIO()
        pickle.dump({"model":best["model"],"scaler":st.session_state.scaler,
                     "encoders":st.session_state.encoders,"features":feat_cols}, buf)
        buf.seek(0)
        st.markdown(f"### 💾 Best Model: `{best_name}`")
        st.download_button("⬇️ Download attrition_model.pkl",
                           buf.getvalue(),"attrition_model.pkl",
                           "application/octet-stream",type="primary")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 7: PREDICT EMPLOYEE
# ════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="sec-head">🎯 Predict Employee Attrition</div>', unsafe_allow_html=True)
    if not st.session_state.trained:
        st.warning("⚠️ Train models first."); return

    best_model = st.session_state.results[st.session_state.best_name]["model"]
    scaler     = st.session_state.scaler
    encoders   = st.session_state.encoders
    feat_cols  = st.session_state.feature_cols
    st.markdown(f"Using: **{st.session_state.best_name}** (best Recall)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Personal Info**")
        age      = st.slider("Age", 18, 60, 30)
        gender   = st.selectbox("Gender",       ["Male","Female"])
        marital  = st.selectbox("Marital Status",["Single","Married","Divorced"])
        edu      = st.slider("Education Level (1=Below College, 5=Doctor)", 1, 5, 3)
        edu_fld  = st.selectbox("Education Field",
                                 ["Life Sciences","Medical","Marketing",
                                  "Technical Degree","Human Resources","Other"])

    with col2:
        st.markdown("**🏢 Job Details**")
        dept     = st.selectbox("Department", DEPARTMENTS)
        job_role = st.selectbox("Job Role", JOB_ROLES)
        job_lvl  = st.slider("Job Level (1–5)", 1, 5, 2)
        overtime = st.selectbox("Over Time", ["No","Yes"])
        stock    = st.slider("Stock Option Level (0–3)", 0, 3, 0)
        pct_hike = st.slider("Percent Salary Hike", 11, 25, 14)

    with col3:
        st.markdown("**😊 Satisfaction & Tenure**")
        job_sat  = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 2)
        env_sat  = st.slider("Environment Satisfaction (1–4)", 1, 4, 2)
        wlb      = st.slider("Work-Life Balance (1–4)", 1, 4, 2)
        job_inv  = st.slider("Job Involvement (1–4)", 1, 4, 2)
        rel_sat  = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)
        yrs_co   = st.slider("Years at Company", 0, 40, 2)
        monthly  = st.slider("Monthly Income ($)", 1000, 20000, 3500, 100)
        num_co   = st.slider("Num Companies Worked", 0, 9, 2)
        yrs_promo= st.slider("Years Since Last Promotion", 0, 15, 3)

    if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "Age": age, "Department": dept, "EducationField": edu_fld,
            "Education": edu, "Gender": gender, "JobRole": job_role,
            "JobLevel": job_lvl, "MaritalStatus": marital, "OverTime": overtime,
            "EnvironmentSatisfaction": env_sat, "JobInvolvement": job_inv,
            "JobSatisfaction": job_sat, "RelationshipSatisfaction": rel_sat,
            "WorkLifeBalance": wlb, "PerformanceRating": 3,
            "StockOptionLevel": stock, "MonthlyIncome": monthly,
            "NumCompaniesWorked": num_co, "PercentSalaryHike": pct_hike,
            "TotalWorkingYears": yrs_co + num_co * 2,
            "TrainingTimesLastYear": 2,
            "YearsAtCompany": yrs_co,
            "YearsInCurrentRole": min(yrs_co, 5),
            "YearsSinceLastPromotion": yrs_promo,
            "YearsWithCurrManager": min(yrs_co, 4),
        }])
        try:
            num_c = row.select_dtypes(include=np.number).columns.tolist()
            for col in row.select_dtypes("object").columns:
                if col in encoders:
                    row[col] = encoders[col].transform(row[col])
            row[num_c] = scaler.transform(row[num_c])
            row = row[feat_cols]
            pred  = best_model.predict(row)[0]
            proba = best_model.predict_proba(row)[0] if hasattr(best_model,"predict_proba") else None
            risk  = proba[1] if proba is not None else float(pred)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            risk_lbl = "HIGH RISK" if risk > 0.6 else "MEDIUM RISK" if risk > 0.35 else "LOW RISK"
            c1.metric("Prediction", "⚠️ WILL LEAVE" if pred else "✅ WILL STAY", risk_lbl)
            c2.metric("Attrition Probability", f"{risk*100:.1f}%")
            c3.metric("Retention Probability", f"{(1-risk)*100:.1f}%")

            if proba is not None:
                fig, ax = dark_fig(figsize=(7, 2.5))
                ax.barh(["Will Stay","Will Leave"], proba,
                        color=[GREEN,RED], alpha=0.85, edgecolor="#0f172a")
                for bar, p in zip(ax.patches, proba):
                    ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                            f"{p*100:.1f}%", va="center", color="#e2e8f0",
                            fontsize=12, fontweight="bold")
                ax.set_xlim(0, 1.25)
                ax.set_title("Prediction Confidence", color="#e2e8f0", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

            if pred == 1:
                box = "risk-high" if risk > 0.65 else "risk-med"
                st.markdown(f"""<div class="{box}">
                <b>⚠️ {risk_lbl} — Recommended HR Actions:</b><br>
                • Schedule a <b>1:1 retention conversation</b> with manager within 1 week<br>
                • Review <b>compensation</b> — consider salary adjustment or bonus<br>
                • Offer <b>flexible work arrangements</b> to improve work-life balance<br>
                • Provide a <b>career growth roadmap</b> and promotion timeline<br>
                • Enrol in <b>learning & development program</b> for upskilling<br>
                • Consider <b>stock option or benefits upgrade</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ Employee is likely to stay. Focus on continued engagement "
                           "and recognition to maintain satisfaction.")
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 8: AT-RISK EMPLOYEES
# ════════════════════════════════════════════════════════════════════════════
def page_atrisk():
    st.markdown('<div class="sec-head">⚠️ At-Risk Employee Identification</div>',
                unsafe_allow_html=True)
    if not st.session_state.trained:
        st.warning("⚠️ Train models first."); return

    df        = st.session_state.df_raw
    best_name = st.session_state.best_name
    model     = st.session_state.results[best_name]["model"]
    scaler    = st.session_state.scaler
    encoders  = st.session_state.encoders
    feat_cols = st.session_state.feature_cols

    st.markdown(f"Model: **{best_name}** | Scoring all **{len(df):,}** employees")
    threshold = st.slider("At-risk threshold (attrition probability ≥)", 0.35, 0.85, 0.55, 0.05)

    if st.button("🔍 Identify At-Risk Employees", type="primary"):
        with st.spinner("Scoring all employees..."):
            data = df.drop(columns=["EmployeeID","Attrition"], errors="ignore").copy()
            num_c = data.select_dtypes(include=np.number).columns.tolist()
            for col in data.select_dtypes("object").columns:
                if col in encoders:
                    data[col] = encoders[col].transform(data[col])
            data[num_c] = scaler.transform(data[num_c])
            data   = data[feat_cols]
            proba  = model.predict_proba(data)[:,1] if hasattr(model,"predict_proba") \
                     else model.predict(data).astype(float)

        df_scored = df.copy()
        df_scored["Attrition_Probability_%"] = np.round(proba * 100, 1)
        df_scored["Risk_Level"] = pd.cut(proba, bins=[0,0.35,0.55,1.0],
                                          labels=["🟢 Low","🟡 Medium","🔴 High"])

        at_risk = df_scored[proba >= threshold].sort_values(
                    "Attrition_Probability_%", ascending=False)

        a, b, c, d = st.columns(4)
        a.metric("At-Risk Employees", f"{len(at_risk):,}", f"{len(at_risk)/len(df)*100:.1f}% of total")
        b.metric("Avg Risk Probability", f"{at_risk['Attrition_Probability_%'].mean():.1f}%")
        c.metric("Actual Attrition in Risk Group",
                 f"{at_risk['Attrition'].sum():,}" if "Attrition" in at_risk else "N/A")
        d.metric("Threshold Used", f"{threshold:.0%}")

        fig, axes = dark_fig(1, 2, figsize=(13, 4))
        axes[0].hist(proba, bins=40, color=PURPLE, alpha=0.75, edgecolor="#0f172a")
        axes[0].axvline(threshold, color=RED, linewidth=2, linestyle="--",
                        label=f"Threshold ({threshold:.0%})")
        axes[0].fill_between([threshold,1], 0, axes[0].get_ylim()[1] or 50,
                              alpha=0.12, color=RED)
        axes[0].set_title("Attrition Probability Distribution",
                           color="#e2e8f0", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Attrition Probability", color="#94a3b8")
        axes[0].set_ylabel("Number of Employees", color="#94a3b8")
        axes[0].legend(fontsize=10)

        risk_counts = df_scored["Risk_Level"].value_counts().reindex(["🔴 High","🟡 Medium","🟢 Low"])
        risk_counts = risk_counts.fillna(0)
        axes[1].pie(risk_counts.values, labels=risk_counts.index,
                    colors=[RED, GOLD, GREEN],
                    autopct="%1.1f%%", startangle=90,
                    textprops={"color":"#e2e8f0","fontsize":11,"fontweight":"bold"},
                    wedgeprops={"edgecolor":"#0f172a","linewidth":2})
        axes[1].set_title("Employee Risk Level Distribution",
                           color="#e2e8f0", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇️ Download Chart", fig_bytes(fig),
                           "risk_distribution.png", "image/png")
        plt.close(fig)

        st.markdown("### 📋 At-Risk Employee List")
        show_cols = ["EmployeeID","Department","JobRole","OverTime","JobSatisfaction",
                     "YearsAtCompany","MonthlyIncome","Attrition_Probability_%","Risk_Level","Attrition"]
        show_cols = [c for c in show_cols if c in at_risk.columns]
        st.dataframe(at_risk[show_cols].reset_index(drop=True), use_container_width=True)

        csv = at_risk[show_cols].to_csv(index=False).encode()
        st.download_button("⬇️ Download At-Risk List (CSV)", csv,
                           "at_risk_employees.csv","text/csv", type="primary")

        st.markdown("""<div class="risk-high">
        <b>🎯 Priority HR Actions for At-Risk Employees:</b><br>
        1. <b>Top 10% highest-risk</b> → Immediate manager conversation + compensation review<br>
        2. <b>OverTime workers</b> → Workload audit + flexible hours or comp time<br>
        3. <b>Low job satisfaction</b> → Career counselling + role redesign discussion<br>
        4. <b>Low income band</b> → Market salary benchmarking + adjustment<br>
        5. <b>No stock options</b> → Introduce ESOP or retention bonus
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════
if   page == PAGES[0]: page_home()
elif page == PAGES[1]: page_load()
elif page == PAGES[2]: page_eda()
elif page == PAGES[3]: page_preprocess()
elif page == PAGES[4]: page_train()
elif page == PAGES[5]: page_evaluate()
elif page == PAGES[6]: page_predict()
elif page == PAGES[7]: page_atrisk()
