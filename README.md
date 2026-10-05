# Student Mental Health Score Prediction API 🧠📊

A production-ready **FastAPI** backend that serves machine learning predictions for the **Student Social Media and Mental Health Impact** dataset using a trained scikit-learn `RandomForestRegressor` pipeline (`Mental_Health_Model.pkl`).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Server
```bash
uvicorn main:app --reload
```
or run directly via Python:
```bash
python main.py
```

The server starts at `http://127.0.0.1:8000`.

### 3. Open the Frontend

Open the app at `http://127.0.0.1:8000/` while the FastAPI server is running. Sign in with the demo account:

- **Email:** `demo@example.com`
- **Password:** `demo123`

You can also click **Use demo account** on the sign-in page to fill in these values and open the predictor.

The login is a client-side demo gate only; it does not authenticate real email accounts. The predictor page sends requests to `/predict` on the same server.

For a separate production frontend, set `CORS_ALLOWED_ORIGINS` to a comma-separated list of explicit frontend origins before starting the API. Local development using a static server on port 5500 is allowed by default.

---

## 📖 API Documentation & Swagger UI

Once the server is running, visit:
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root status and endpoints information |
| `GET` | `/health` | Healthcheck and model load verification |
| `GET` | `/model-info` | Feature list, supported categorical values & model specs |
| `POST` | `/predict` | Single student mental health score prediction |
| `POST` | `/predict/batch` | Batch student predictions |

---

## 💡 Example Request & Response

### Single Prediction (`POST /predict`)

#### Request Body (JSON):
```json
{
  "Age": 21,
  "Gender": "Male",
  "Country": "Canada",
  "Academic_Level": "Undergraduate",
  "Most_Used_Platform": "Instagram",
  "Purpose_Of_Use": "Entertainment",
  "Avg_Daily_Usage_Hours": 4.6,
  "Daily_Unlocks": 166,
  "Study_Hours": 4.0,
  "Physical_Activity_Hours": 1.8,
  "Sleep_Hours_Per_Night": 6.7,
  "Stress_Level": "Medium"
}
```

#### Response Body (JSON):
```json
{
  "predicted_mental_health_score": 7.0,
  "risk_category": "Good / Moderate",
  "status": "success"
}
```
