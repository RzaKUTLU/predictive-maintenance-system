# 🔧 Predictive Maintenance System

An AI-powered predictive maintenance platform for industrial machinery built with Flask and XGBoost. Predicts equipment failures and estimates remaining useful life to enable proactive maintenance scheduling.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

This system uses machine learning models trained on industrial sensor data to predict:
- Whether a machine will fail within the next **7 days**
- The machine's **Remaining Useful Life (RUL)** in days
- A **failure probability score** with risk classification

It also includes an AI-powered maintenance assistant chatbot (Google Gemini) for expert recommendations.

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
