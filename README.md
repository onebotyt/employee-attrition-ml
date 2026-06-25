# 👥 Employee Attrition Prediction — ML Dashboard
### RISE Internship · Machine Learning & AI | Tamizhan Skills
**Project 4: Industry-Oriented Machine Learning System for Employee Attrition Prediction**

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35+-red?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/NumPy-ML_Models-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HR_Analytics-Project_4-purple?style=for-the-badge"/>
</p>

---

## 🔗 Live Demo
> 👉 **[Open App on Streamlit Cloud](https://employee-attrition-ml-hswubfms4tdnfxhz7wbmn9.streamlit.app/)**  
> 👉 **[View on GitHub](https://github.com/onebotyt/employee-attrition-ml.git)**

---

## 📌 Problem Statement
Organizations face high costs due to employee attrition and struggle to identify factors
that cause employees to leave. Manual HR analysis is ineffective for large workforces.
Industry uses machine learning to **predict attrition** and improve **retention strategies**.

## 🎯 Objective
Build a machine learning model that predicts employee attrition based on HR and
performance data — with focus on **Accuracy** and **Recall** metrics.

---

## ✅ Project Requirements Coverage

| Requirement | Status |
|-------------|--------|
| HR dataset ingestion | ✅ Kaggle Employee Attrition dataset (25 features) |
| Data cleaning and preprocessing | ✅ Label encoding + StandardScaler + SMOTE |
| Feature engineering for attrition factors | ✅ 25 HR features engineered |
| Exploratory data analysis | ✅ 5 EDA tabs with charts |
| Machine learning model training | ✅ LR, RF, SVM, KNN with 5-fold CV (Pure NumPy) |
| Model evaluation using accuracy and recall | ✅ Primary metric: Recall |
| Identification of key attrition drivers | ✅ Feature importance chart & SHAP-style explanation |
| Visualization of employee trends | ✅ Department, tenure, income, satisfaction |
| Model documentation | ✅ README + Interactive Streamlit Dashboard |

---

## 📊 Model Results

| Model | Test Accuracy | Test Recall | ROC-AUC |
|-------|--------------|-------------|---------|
| **Logistic Regression** | ~69% | **~80%** | **~0.86** |
| Random Forest | ~76% | ~74% | ~0.87 |
| SVM (Linear) | ~72% | ~76% | ~0.85 |
| KNN | ~68% | ~70% | ~0.79 |

> **Primary metric: Recall** — missing an at-risk employee costs more than a false alarm.
> 
> *Note: To avoid C++ BLAS/DLL deployment issues on free cloud tiers, all ML algorithms (Logistic Regression, Random Forest, SVM, KNN) have been custom-implemented from scratch in pure NumPy!*

---

## 🔑 Top Attrition Drivers (from Random Forest)
1. **OverTime** — #1 driver; overtime workers leave at 3× the rate
2. **MonthlyIncome** — Low-income employees have much higher attrition
3. **YearsAtCompany** — New employees (< 3 yrs) are highest risk
4. **JobSatisfaction** — Score 1 employees leave far more frequently
5. **Age** — Younger employees have higher mobility

---

## 🗂️ Project Structure

```
employee_attrition_project/
├── app.py                              ← Streamlit dashboard (9 pages)
├── ml_models.py                        ← Pure NumPy Machine Learning models
├── employee_attrition_dataset.csv      ← Kaggle dataset
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
└── .streamlit/
    └── config.toml                     ← Purple HR theme
```

---

## 💻 How to Run Locally

**1. Install Python 3.10+** → [python.org/downloads](https://python.org/downloads)  
Tick **"Add to PATH"** during install.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the dashboard**
```bash
streamlit run app.py
```

**4. Open browser** → `http://localhost:8501`

---

## 📊 Dashboard Pages (8 Pages/Tabs)

| Page | What it does |
|------|-------------|
| 🏠 Home | Problem statement, objective, all requirements checklist |
| 📥 Load HR Data | Load Kaggle dataset or upload custom CSV |
| 🔍 EDA & Employee Trends | 5 tabs: attrition overview, dept/role, satisfaction, compensation, tenure |
| ⚙️ Preprocess | Label encoding, StandardScaler, stratified split, SMOTE |
| 🤖 Train Models | LR, RF, SVM, KNN with tunable hyperparameters + CV |
| 📊 Evaluate | Confusion matrices, metrics, ROC curves, **feature importance**, explainability |
| 🎯 Predict Employee | Input employee details → attrition risk + confidence |
| ⚠️ At-Risk Employees | Score all employees, download at-risk CSV for HR team |

---

## 👤 Author

**Girdhar Dhami**  
Diploma in Information Technology (V Semester)  
Uttarakhand Government institute of Polytechnic Nainital

------------
RISE Internship 2026 — Tamizhan Skills  
📧 Rise@tamizhanskills.com | 🌐 www.tamizhanskills.com

---

## 📄 License
MIT License — free to use and modify for educational purposes.
