# PhishGuard
# 🛡️ PhishGuard – AI-Powered Phishing Detection System

PhishGuard is a real-time phishing detection system that helps users identify **phishing websites** using **Machine Learning** and a clean web interface.

---

## 🚀 Features
- Detect phishing websites using URL-based ML models
- Real-time results with confidence score
- Simple web UI (React frontend + Flask backend)
- (Optional) Screenshot-based detection

---

## 🛠️ Tech Stack
- **Frontend:** React.js
- **Backend:** Flask (Python)
- **ML Models:** Random Forest, XGBoost
- **Database (Optional):** SQLite / PostgreSQL

---

## 📂 Project Structure

PhishGuard/
│── frontend/        # React app (UI)
│── backend/         # Flask API
│── ml_model/        # Jupyter notebooks, model.pkl
│── docs/            # screenshots, PPT, demo resources
│── README.md
│── requirements.txt # Python dependencies


---

## ⚡ Setup Instructions
### 1. Clone Repo
```bash
git clone https://github.com/<your-username>/PhishGuard.git
cd PhishGuard


2. Backend Setup

cd backend
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt

3. Frontend Setup

cd frontend
npm install
npm start

🎥 Demo (to be added)

A 2–5 minute video showing how PhishGuard detects phishing websites.

---


Team

Haritha Netha – Developer

## 4️⃣ requirements.txt (start basic)
Inside `backend/requirements.txt`:

flask
flask-cors
scikit-learn
xgboost
pandas
numpy
joblib



---

## 5️⃣ Git Init & Push
Run:

```bash
git init
git add .
git commit -m "Day 1 setup: project structure + README + requirements"
git branch -M main
git remote add origin https://github.com/<your-username>/PhishGuard.git
git push -u origin main
