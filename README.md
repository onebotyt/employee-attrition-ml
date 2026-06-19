# 👥 Employee Attrition Prediction — ML Dashboard
### RISE Internship · Machine Learning & AI | Tamizhan Skills
**Project 4: Industry-Oriented Machine Learning System for Employee Attrition Prediction**

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35+-red?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=for-the-badge"/>
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
| HR dataset ingestion | ✅ Synthetic IBM-style dataset (1,000 employees) |
| Data cleaning and preprocessing | ✅ Label encoding + StandardScaler |
| Feature engineering for attrition factors | ✅ 25 HR features engineered |
| Exploratory data analysis | ✅ 5 EDA tabs with charts |
| Machine learning model training | ✅ LR, RF, SVM, KNN with 5-fold CV |
| Model evaluation using accuracy and recall | ✅ Primary metric: Recall |
| Identification of key attrition drivers | ✅ Feature importance chart |
| Visualization of employee trends | ✅ Department, tenure, income, satisfaction |
| Model documentation | ✅ README + Jupyter Notebook |

---

## 📊 Model Results

| Model | Test Accuracy | Test Recall | ROC-AUC |
|-------|--------------|-------------|---------|
| **Logistic Regression** | ~69% | **~80%** | **~0.86** |
| Random Forest | ~76% | ~74% | ~0.87 |
| SVM (RBF) | ~72% | ~76% | ~0.85 |
| KNN | ~68% | ~70% | ~0.79 |

> **Primary metric: Recall** — missing an at-risk employee costs more than a false alarm.

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
├── requirements.txt                    ← Python dependencies
├── employee_attrition_prediction.ipynb ← Jupyter Notebook (step-by-step)
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

### Run the Jupyter Notebook
```bash
pip install jupyter
jupyter notebook employee_attrition_prediction.ipynb
```

---


## 📊 Dashboard Pages (9 Pages)

| Page | What it does |
|------|-------------|
| 🏠 Home | Problem statement, objective, all 9 requirements checklist |
| 📥 Load HR Data | Generate synthetic IBM HR dataset or upload CSV |
| 🔍 EDA & Employee Trends | 5 tabs: attrition overview, dept/role, satisfaction, compensation, tenure |
| ⚙️ Preprocess | Label encoding, StandardScaler, stratified split |
| 🤖 Train Models | LR, RF, SVM, KNN with tunable hyperparameters + 5-fold CV |
| 📊 Evaluate | Confusion matrices, metrics, ROC curves, **feature importance (attrition drivers)** |
| 🎯 Predict Employee | Input employee details → attrition risk + HR action recommendations |
| ⚠️ At-Risk Employees | Score all employees, download at-risk CSV for HR team |




## 🌱 Use Real IBM HR Analytics Data

```python
# Download from Kaggle: kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Encode Attrition column (Yes/No → 1/0)
df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)

# Remove constant columns (EmployeeCount, Over18, StandardHours)
df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], errors='ignore')
```

---

## 👤 Author

**Girdhar Dhami**  
Diploma in Information Technology (V Semester)  
Uttarakhand Government institute of Polytechnic Nainital
RISE Internship 2026 — Tamizhan Skills  
📧 Rise@tamizhanskills.com | 🌐 www.tamizhanskills.com

---

## 📄 License
MIT License — free to use and modify for educational purposes.
