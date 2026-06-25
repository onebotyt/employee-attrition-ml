# 🚀 GitHub + Streamlit Cloud Deployment Guide
## Employee Attrition Prediction | RISE Internship

---

## 📁 Files You Need on GitHub (exactly these 5)

```
your-repo/
├── app.py                              ← Main Streamlit dashboard
├── ml_models.py                        ← Pure NumPy ML models
├── requirements.txt                    ← Pinned dependencies
├── README.md                           ← Project documentation
└── .streamlit/
    └── config.toml                     ← Streamlit production settings
```

**Do NOT push:**  `saved_model/`, `*.pkl`, `*.png`, `.env`, `secrets.toml`

---

## Step 1 — Create GitHub Repository

1. Go to **github.com** → Sign in → Click **"+"** → **"New repository"**
2. Repository name: `employee-attrition-ml`
3. Set to **Public** (required for free Streamlit Cloud)
4. Click **"Create repository"**

---

## Step 2 — Push Your Code

Open **Command Prompt** in your project folder and run these commands **one by one**:

```bash
git init
git add app.py ml_models.py requirements.txt README.md
git add .streamlit/config.toml
git add .gitignore
git commit -m "Initial commit: Employee Attrition ML Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/employee-attrition-ml.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

**Verify on GitHub:**  
Go to `github.com/YOUR_USERNAME/employee-attrition-ml`  
You should see all 5 files listed ✅

---

## Step 3 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in"** → **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub
4. Click **"New app"**
5. Fill in:
   - **Repository:** `YOUR_USERNAME/employee-attrition-ml`
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click **"Deploy!"**

Your app will build in ~2 minutes. You will get a free link:
```
https://your-username-employee-attrition-ml-app-xxxx.streamlit.app
```

---

## Step 4 — Share Your Link

Put the live link in:
- Your **README.md** (replace `YOUR_STREAMLIT_LINK_HERE`)
- Your **internship presentation** last slide
- Your **GitHub profile** bio

---

## 🔒 Security Checklist Before Pushing

Run through this list before `git push`:

- [ ] No passwords or API keys in any `.py` file
- [ ] No `.env` file pushed (check with `git status`)
- [ ] No `saved_model/` folder pushed
- [ ] No `*.pkl` files pushed
- [ ] `.gitignore` file is present in repo
- [ ] `.streamlit/secrets.toml` is NOT pushed
- [ ] `requirements.txt` has pinned versions (no `>=`)
- [ ] `enableXsrfProtection = true` in `config.toml`
- [ ] `gatherUsageStats = false` in `config.toml`

---

## 🔄 How to Update Your App After Changes

```bash
git add .
git commit -m "Fix: improved model recall"
git push
```
Streamlit Cloud **auto-redeploys** within 1-2 minutes. No manual step needed.

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| App shows "Module not found" | Check `requirements.txt` has the missing package |
| App crashes on startup | Check Streamlit Cloud logs → "Manage app" → "Logs" |
| "Branch not found" error | Make sure you pushed to `main` not `master` |
| White screen | Clear browser cache or open in incognito |
| "File not found: ml_models.py" | Make sure `ml_models.py` is committed to GitHub |

---

## 📞 Tamizhan Skills Contact
Rise@tamizhanskills.com | www.tamizhanskills.com | +91 6383418100
