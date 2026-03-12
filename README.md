# 🔧 Predictive Maintenance System

An AI-powered predictive maintenance platform for industrial machinery built with Flask and XGBoost. Predicts equipment failures and estimates remaining useful life to enable proactive maintenance scheduling.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

This system uses machine learning models trained on industrial sensor data to predict whether a machine will fail within the next 7 days, its Remaining Useful Life (RUL) in days, and a failure probability score with risk classification. It also includes an AI-powered maintenance assistant chatbot (Google Gemini) for expert recommendations.

The live demo showcases a full end-to-end predictive maintenance workflow for industrial machinery. Users can input real-time sensor readings — including temperature, vibration, power consumption, oil level, coolant level, and error codes — directly through the web dashboard. Upon submission, the system instantly runs the data through a trained XGBoost pipeline and returns a failure probability score, a risk level classification (Low to Critical), and an estimated Remaining Useful Life in days. The demo also highlights the AI Maintenance Assistant, where users can ask natural language questions about machine health and receive expert-level maintenance recommendations powered by Google Gemini 2.5 Flash. The prediction history page displays all past evaluations stored in the database, allowing users to track machine degradation over time. Additionally, the feature importance analysis page visualizes which sensor parameters — such as the Temperature-Vibration interaction or Oil-Coolant differential — contribute most to the failure prediction, providing explainability for the model's decisions. The entire system is accessible via both the web interface and a RESTful JSON API, making it suitable for integration with automation tools such as n8n.

---

## ✨ Features

- **Machine Failure Prediction** – XGBoost Classifier predicts 7-day failure risk with probability score
- **Remaining Useful Life (RUL)** – XGBoost Regressor estimates days until maintenance is needed
- **Risk Classification** – Low / Medium / High / Critical risk levels
- **AI Maintenance Assistant** – Google Gemini 2.5 Flash powered chatbot for maintenance advice
- **Google Sheets Integration** – Real-time external maintenance log reading for context-aware chatbot
- **Prediction History** – All predictions stored in PostgreSQL database with pagination
- **Feature Importance Analysis** – Visualizes which sensor parameters most influence failure risk
- **Dual API Support** – Endpoints with and without Operational Hours for flexible integration
- **Batch Prediction** – Predict multiple machines in a single API call
- **n8n Automation Integration** – Ready-to-use endpoints and documentation for n8n workflows
- **REST API** – JSON-based API for easy integration with external systems
- **Web Dashboard** – Bootstrap-based responsive UI with Turkish/English support
- **Health Check** – `/api_status` endpoint for server and model monitoring

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.0.3, Python 3.11+ |
| ML Models | XGBoost, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Database | PostgreSQL (SQLite fallback) |
| ORM | SQLAlchemy |
| AI Chatbot | Google Gemini 2.5 Flash |
| Web Server | Gunicorn |
| Frontend | Bootstrap, Font Awesome, Jinja2 |

---

## 📡 API Endpoints

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Main prediction endpoint (supports batch) |
| `POST` | `/predict_without_hours` | Prediction without Operational Hours (fixed 25,000 hrs) |
| `GET` | `/api_status` | Health check – server and model status |

### Web Interface

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main dashboard |
| `GET` | `/history` | Prediction history with pagination |
| `GET` | `/machine/<id>` | Detailed machine view |
| `GET` | `/docs` | API documentation |
| `GET` | `/feature_analysis` | Feature importance analysis |
| `GET` | `/maintenance_assistant` | AI chatbot interface |
| `GET` | `/n8n_integration` | n8n integration guide |

### AI Assistant

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/maintenance_chat` | Gemini AI chatbot endpoint |

---

## 📥 Input Parameters

```json
{
  "Temperature": 75.5,
  "Vibration": 0.45,
  "Power_Consumption": 320.0,
  "Operational_Hours": 15000,
  "Error_Codes": 3,
  "Oil_Level": 85.0,
  "Coolant_Level": 90.0,
  "Last_Maintenance_Days": 45,
  "Machine_Type": "CNC_Lathe"
}
```

## 📤 Output Example

```json
{
  "failure_prediction": 1,
  "failure_probability": 0.87,
  "remaining_useful_life_days": 12,
  "risk_level": "Critical",
  "will_fail_in_7_days": true,
  "machine_id": "MACHINE_001"
}
```

---

## ⚙️ Feature Engineering

The ML pipeline applies the following transformations:

- **Temperature × Vibration** interaction feature
- **Power per Operational Hour** ratio
- **Log transformation** of Error Codes
- **Oil–Coolant differential**
- **Maintenance-to-failure ratio**
- Highly correlated feature removal
- StandardScaler normalization

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL (or SQLite for development)

### Installation

```bash
git clone https://github.com/RzaKUTLU/predictive-maintenance-system.git
cd predictive-maintenance-system
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file or set the following variables:

```
GEMINI_API_KEY=your_google_gemini_api_key
SESSION_SECRET=your_secret_key
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

### Run

```bash
gunicorn main:app --bind 0.0.0.0:5000
```

Then visit `http://localhost:5000`

---

## 🗄️ Database Schema

### Machine Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| machine_id | String | Unique machine identifier |
| machine_type | String | Type of machine |
| created_at | DateTime | Record creation time |

### Prediction Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| machine_id | FK | Reference to machine |
| failure_probability | Float | Predicted failure probability |
| remaining_useful_life | Integer | Estimated days until failure |
| risk_level | String | Low/Medium/High/Critical |
| created_at | DateTime | Prediction timestamp |

---

## 📊 ML Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| `xgb_clf.pkl` | XGBoost Classifier | Binary failure prediction |
| `xgb_reg.pkl` | XGBoost Regressor | RUL estimation |
| `preprocessor.pkl` | ColumnTransformer | Categorical encoding |
| `scaler.pkl` | StandardScaler | Feature normalization |
| `feature_selector.pkl` | Custom selector | Correlation-based feature removal |

---

## 🔄 n8n Integration

This system provides full support for [n8n](https://n8n.io) automation workflows, enabling you to automate machine monitoring, failure alerts, and maintenance scheduling without writing additional code.

### Endpoints

| Endpoint | Use Case |
|----------|----------|
| `POST /predict` | Full prediction **with** Operational Hours |
| `POST /predict_without_hours` | Prediction **without** Operational Hours (prevents data leakage) |

---

### 🔁 Real-World Workflow Example

The workflow below demonstrates a complete automated maintenance pipeline:

```
Manual Trigger
      ↓
HTTP Request  →  POST /predict (sends machine sensor data)
      ↓
Set – Timestamp  →  Adds current timestamp to response
      ↓
IF  →  Checks failure_prediction result
   ↙ true                      ↘ false
True_Set                     False_Set
(failure detected)           (machine healthy)
      ↘                          ↙
           Merge (combine results)
                  ↓
      Append row in sheet  →  Logs result to Google Sheets
                  ↓
             HTML  →  Converts data to HTML table
                  ↓
      Send a message  →  Sends Gmail alert
```

**Workflow Steps:**
1. **Manual / Schedule Trigger** – Starts the workflow on demand or on a schedule
2. **HTTP Request** – Sends machine sensor data to `POST /predict`
3. **Set – Timestamp** – Adds a timestamp field to the prediction response
4. **IF** – Branches based on `failure_prediction` value (1 = failure detected)
5. **True_Set / False_Set** – Labels the result as "FAILURE" or "HEALTHY"
6. **Merge** – Combines both branches back into a single flow
7. **Append row in sheet** – Logs all results to a Google Sheets document
8. **HTML** – Formats the data into an HTML table for the email body
9. **Send a message** – Sends a Gmail notification with the maintenance report

---

### HTTP Request Node Configuration

**Method:** `POST`  
**URL:** `https://your-app-url/predict`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "Temperature": {{ $json.temperature }},
  "Vibration": {{ $json.vibration }},
  "Power_Consumption": {{ $json.power }},
  "Operational_Hours": {{ $json.hours }},
  "Error_Codes": {{ $json.errors }},
  "Oil_Level": {{ $json.oil }},
  "Coolant_Level": {{ $json.coolant }},
  "Last_Maintenance_Days": {{ $json.maintenance_days }},
  "Machine_Type": "{{ $json.machine_type }}"
}
```

### Response Fields for IF Node Conditions

| Field | Type | Example Condition |
|-------|------|-------------------|
| `failure_prediction` | 0 or 1 | `{{ $json.failure_prediction === 1 }}` |
| `failure_probability` | 0.0–1.0 | `{{ $json.failure_probability > 0.8 }}` |
| `risk_level` | String | `{{ $json.risk_level === "Critical" }}` |
| `remaining_useful_life_days` | Integer | `{{ $json.remaining_useful_life_days < 7 }}` |
| `will_fail_in_7_days` | Boolean | `{{ $json.will_fail_in_7_days === true }}` |

### Batch Prediction (Multiple Machines)

Send an array to process multiple machines in one call:

```json
[
  { "Temperature": 75, "Vibration": 0.45, "Machine_Type": "CNC_Lathe", ... },
  { "Temperature": 82, "Vibration": 0.61, "Machine_Type": "Hydraulic_Press", ... }
]
```

> 📘 For the full integration guide, visit `/n8n_integration` in the web interface.

---

## 👤 Author

**Rza KUTLU**  
GitHub: [@RzaKUTLU](https://github.com/RzaKUTLU)

---

*Built for academic thesis research and industrial IoT applications.*
