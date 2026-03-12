#  Predictive Maintenance System
## AI-Powered Industrial Machine Failure Prediction & Maintenance Planning

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)

> A comprehensive Flask-based web application that leverages advanced machine learning models to predict industrial machinery failures and optimize maintenance schedules. Features real-time analytics, AI-powered chatbot assistance, and seamless workflow integrations.

---

## 🚀 Key Features

### 🔍 **Advanced Machine Learning Pipeline**
- **XGBoost Models**: Dual-model architecture for failure classification and remaining useful life (RUL) regression
- **Feature Importance**: XGBoost native feature importance analysis for model interpretability
- **Smart Feature Engineering**: 48+ engineered features including temperature-vibration interactions, power efficiency metrics, and maintenance ratios
- **Real-time Predictions**: Lightning-fast inference with pre-trained models

### 🤖 **AI-Powered Maintenance Assistant**
- **Google Gemini 2.5 Flash Integration**: Advanced LLM-powered chatbot for maintenance guidance
- **Contextual Responses**: Advanced AI responses tailored to maintenance scenarios
- **Contextual Advice**: Machine-specific maintenance recommendations based on prediction results
- **Multi-language Support**: Turkish and English interface

### 🔄 **Workflow & Integration Capabilities**
- **n8n Integration**: Seamless automation workflow integration with dual API endpoints
- **Data Persistence**: PostgreSQL database with prediction history tracking
- **REST API**: Comprehensive API with single and batch prediction endpoints
- **PostgreSQL Database**: Robust data persistence with prediction history

### 📊 **Interactive Analytics Dashboard**
- **Feature Importance Analysis**: Dynamic XGBoost feature analysis with/without operational hours
- **Machine Type Breakdown**: Detailed analytics by equipment categories
- **Historical Tracking**: Comprehensive prediction history with pagination
- **Visual Insights**: Interactive charts and graphs for maintenance planning

---

## 🏗️ Architecture Overview

### Machine Learning Stack
```
┌─────────────────────────────────────────────────────────────┐
│  Input Processing → Feature Engineering → Model Pipeline    │
├─────────────────────────────────────────────────────────────┤
│  • Data Validation    • Interaction Features  • XGBoost     │
│  • Missing Value      • Power Efficiency      • StandardScaler│
│    Handling           • Log Transformations   • Predictions │
│  • Type Encoding      • Correlation Removal   • Predictions │
└─────────────────────────────────────────────────────────────┘
```

### Application Architecture
- **Backend**: Flask + SQLAlchemy + Gunicorn
- **Database**: PostgreSQL with automatic failover to SQLite
- **AI/ML**: XGBoost, scikit-learn, Google Gemini AI
- **Frontend**: Bootstrap 5, Responsive Design, Interactive JavaScript
- **Deployment**: Multi-platform support (Render, Railway, Vercel, Replit)

---

## 📋 Supported Equipment Types

The system supports **40+ industrial machine types** with specialized prediction models:

| Category | Equipment Types |
|----------|----------------|
| **Manufacturing** | CNC Lathe, CNC Mill, 3D Printer, Injection Molder, Laser Cutter |
| **Material Handling** | Conveyor Belt, AGV, Forklift Electric, Crane, Palletizer |
| **Process Equipment** | Mixer, Compressor, Hydraulic Press, Heat Exchanger, Furnace |
| **Quality Control** | Vision System, XRay Inspector, CMM, Pick and Place |
| **Utilities** | Pump, Valve Controller, Industrial Chiller, Boiler |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (optional, SQLite fallback available)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/RzaKUTLU/predictive-maintenance-system.git
cd predictive-maintenance-system
```

2. **Install Dependencies**
```bash
pip install -r deployment_requirements.txt
```

3. **Environment Setup**
```bash
# Required for AI chatbot
export GEMINI_API_KEY="your_gemini_api_key"
export SESSION_SECRET="your_session_secret"

# Optional: PostgreSQL connection
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
```

4. **Run the Application**
```bash
# Development
python main.py

# Production
gunicorn --bind 0.0.0.0:5000 --workers 4 main:app
```

Visit `http://localhost:5000` to access the application.

---

## 🔌 API Documentation

### Single Machine Prediction
```http
POST /predict
Content-Type: application/json

{
  "Temperature_C": 75.5,
  "Vibration_mms": 4.2,
  "Power_Consumption_kW": 150.3,
  "Operational_Hours": 8760,
  "Error_Codes_Last_30_Days": 5,
  "Oil_Level_pct": 85.5,
  "Coolant_Level_pct": 78.2,
  "Maintenance_History_Count": 12,
  "Failure_History_Count": 3,
  "Machine_Type": "CNC_Lathe",
  "AI_Override_Events": 0,
  "Installation_Year": 2020,
  "Last_Maintenance_Days_Ago": 30,
  "Sound_dB": 65.5
}
```

### Response Format
```json
{
  "Failure_Within_7_Days": false,
  "Failure_Probability": 0.2301,
  "Remaining_Useful_Life_days": 145,
  "Machine_ID": "M001",          // Optional: only if provided in input
  "Machine_Type": "CNC_Lathe"    // Optional: only if provided in input
}
```

### Batch Predictions
```http
POST /predict
Content-Type: application/json

[
  {machine1_data},
  {machine2_data},
  {machine3_data}
]
```

### n8n Integration Endpoints
- **With Operational Hours**: `/predict` (standard endpoint)
- **Without Operational Hours**: `/predict_without_hours` (uses fixed 25,000 hours)

---

## 🧠 Model Explainability & Feature Analysis

The system provides comprehensive feature importance analysis using XGBoost's built-in feature importance:

### Features
- **Global Feature Importance**: Understanding which features matter most across all predictions using XGBoost feature importance
- **Interactive Visualizations**: Dynamic feature importance plots in the web interface  
- **Two Analysis Modes**: 
  - Full analysis with operational hours
  - Sensor-only analysis without operational hours
- **Machine Type Breakdown**: Feature importance analysis by equipment categories

### Accessing Feature Analysis
- **Web Interface**: `/feature_importance` and `/feature_analysis_without_hours`
- **API Endpoints**: Feature importance available via dedicated endpoints
- **Machine-specific**: Breakdown by equipment type and category

---

## 🤖 AI Chatbot Integration

### Google Gemini 2.5 Flash LLM
The system includes an advanced AI maintenance assistant powered by Google's Gemini 2.5 Flash model:

#### Capabilities
- **Contextual Maintenance Advice**: Tailored recommendations based on machine data
- **Comprehensive Knowledge**: Leverages extensive training data for maintenance guidance
- **Multilingual Support**: Turkish and English language processing
- **Technical Troubleshooting**: Step-by-step repair guidance

#### Access Points
- **Web Interface**: `/maintenance_assistant`
- **API Endpoint**: `POST /maintenance_chat`
- **Integration**: Embedded in prediction results

#### Knowledge Base Integration
The chatbot utilizes its training knowledge for:
- Maintenance procedures and best practices
- Equipment-specific troubleshooting guides
- Technical support and guidance
- Cost optimization recommendations

---

## 🔄 n8n Workflow Automation
- 🔗 n8n Integration Guide

> Predictive Maintenance API'sine n8n workflow'larını bağlamak için kapsamlı kurulum rehberi

---

## 📋 İçindekiler

- [API Endpoint Seçimi]
- [Hızlı Kurulum]
- [JSON Örnekleri]
- [Hangi API'yi Kullanmalıyım?]
- [Dinamik Veri Eşleme]
- [Response Handling]
- [Workflow Örnekleri]
- [Hata Yönetimi]
- [Test]

---

## 🔀 API Endpoint Seçimi

| Özellik | Normal API | Çalışma Saatleri Olmadan |
|---|---|---|
| **Endpoint** | `/predict` | `/predict_without_hours` |
| **Çalışma Saati** | Gerekli | Gerekli değil |
| **Doğruluk** | En yüksek | Orta (25.000 saat varsayımı) |
| **Kullanım** | Production | Test / Geliştirme |

---

## ⚡ Hızlı Kurulum

**HTTP Request Node Ayarları:**

```
Method:        POST
Content-Type:  application/json
Auth:          None
```

**Endpoint URL'leri:**

```
# Normal API
POST https://69467501-d5c2-47d7-add1-f0b1d7bd521f-00-19wwexxsknahh.riker.replit.dev/predict

# Çalışma Saatleri Olmadan
POST https://69467501-d5c2-47d7-add1-f0b1d7bd521f-00-19wwexxsknahh.riker.replit.dev/predict_without_hours
```

---

## 📦 JSON Örnekleri

### Normal API (`/predict`)

```json
{
  "Temperature_C": 75.5,
  "Vibration_mms": 4.2,
  "Power_Consumption_kW": 150.3,
  "Operational_Hours": 8760,
  "Error_Codes_Last_30_Days": 5,
  "Oil_Level_pct": 85.5,
  "Coolant_Level_pct": 78.2,
  "Maintenance_History_Count": 12,
  "Failure_History_Count": 3,
  "Machine_Type": "3D_Printer",
  "AI_Override_Events": 0,
  "Installation_Year": 2020,
  "Last_Maintenance_Days_Ago": 30,
  "Sound_dB": 65.5
}
```

### Çalışma Saatleri Olmadan (`/predict_without_hours`)

Yukarıdaki JSON'dan `Operational_Hours` alanını çıkararak kullanın.

---

## 🤔 Hangi API'yi Kullanmalıyım?

### ✅ Normal API'yi Kullan (`/predict`)
- Çalışma saatleri verisi mevcutsa
- En yüksek doğruluk gerekiyorsa
- Production ortamında
- Kritik kararlar alınacaksa

### ✅ Saatsiz API'yi Kullan (`/predict_without_hours`)
- Çalışma saatleri verisi yoksa
- Test / geliştirme ortamındaysan
- Hızlı tahmin gerekiyorsa
- Operasyonel saatler güvenilir değilse

> **💡 Öneri:** Çalışma saatleri veriniz varsa her zaman normal API'yi tercih edin.

---

## 🔄 Dinamik Veri Eşleme

n8n workflow'larında kullanabileceğiniz expression örnekleri:

| Alan | n8n Expression | Açıklama |
|---|---|---|
| `Temperature_C` | `{{ $json.sensor_data.temperature }}` | Sıcaklık sensörü |
| `Vibration_mms` | `{{ $json.vibration_sensor }}` | Titreşim sensörü |
| `Power_Consumption_kW` | `{{ $json.power_meter.current_kw }}` | Güç sayacı |
| `Operational_Hours` | `{{ $json.machine.total_hours }}` | Makine çalışma sayacı |
| `Machine_Type` | `{{ $json.machine.type \|\| "Type_A" }}` | Makine tipi (fallback ile) |

---

## 📨 Response Handling

### Başarılı Yanıt (`200 OK`)

```json
{
  "Failure_Within_7_Days": false,
  "Failure_Probability": 0.2345,
  "Remaining_Useful_Life_days": 45.67
}
```

### Hata Yanıtı (`500`)

```json
{
  "error": "Detailed error message"
}
```

### n8n'de Response Verilerine Erişim

```
{{ $json.Failure_Within_7_Days }}        → Boolean: true / false
{{ $json.Failure_Probability }}          → Float: 0.0 - 1.0
{{ $json.Remaining_Useful_Life_days }}   → Float: kalan gün
```

---

## 🛠️ Workflow Örnekleri

### Temel Makine İzleme

```
Cron (saatlik) → Sensör Verisi Al → Tahmin API → IF (arıza riski?) → E-posta / Slack
```

### Gelişmiş Gerçek Dünya Workflow'u

```
Manual Trigger
    └── HTTP Request (Tahmin API)
            └── Set Node (Timestamp ekle)
                    └── IF Node (Failure_Within_7_Days?)
                            ├── TRUE  → Set Node → Merge → Google Sheets + E-posta
                            └── FALSE → Set Node → Merge → Google Sheets
```

**Workflow Adımları:**

1. **Veri Toplama** — Makine sensör verilerini toplar
2. **API Çağrısı** — `/predict` endpoint'ine POST isteği gönderir
3. **Koşullu Yönlendirme** — Arıza riskine göre farklı dallara yönlendirir
4. **Kayıt** — Tüm sonuçları Google Sheets'e kaydeder
5. **Bildirim** — Kritik durumlarda e-posta gönderir

**Kullanım Alanları:**
- Otomatik makine izleme
- Bakım planlaması
- Arıza öncesi uyarı sistemi
- Performans raporlaması
- Karar destek sistemi

---

## ⚠️ Hata Yönetimi

| Hata Kodu | Neden | Çözüm |
|---|---|---|
| `400 Bad Request` | Geçersiz giriş formatı | JSON yapısını kontrol et |
| `500 Internal Error` | Model işlem hatası | Girdi değerlerini kontrol et |
| `Network Timeout` | API erişilemiyor | Retry mekanizması ekle |

**Önerilen n8n Hata Stratejisi:**

1. **Error Workflow** tanımla
2. **IF Node** ile HTTP status code kontrol et
3. **Retry Logic** ekle (geçici hatalar için)
4. **Fallback Actions** tanımla (API erişilemediğinde)

---

## 🧪 Test

### 1. API Durumunu Kontrol Et

```
GET https://69467501-d5c2-47d7-add1-f0b1d7bd521f-00-19wwexxsknahh.riker.replit.dev/api_status
```

**Beklenen Yanıt:**

```json
{
  "status": "running",
  "models_loaded": true,
  "message": "🔧 Predictive Maintenance API is Running!"
}
```

### 2. Yüksek Risk Senaryosu Test Değerleri

| Parametre | Yüksek Risk Değeri |
|---|---|
| `Temperature_C` | > 90°C |
| `Vibration_mms` | > 8 mm/s |
| `Error_Codes_Last_30_Days` | > 10 |
| `Oil_Level_pct` | < 30% |
| `Coolant_Level_pct` | < 30% |

## 🗂️ Project Structure

```
predictive-maintenance-system/
├── 📄 main.py                          # Flask application core
├── 📁 templates/                       # Jinja2 HTML templates
│   ├── base.html                       # Base template with Bootstrap
│   ├── index.html                      # Main dashboard
│   ├── predict.html                    # Prediction interface  
│   ├── feature_importance.html         # Feature analysis (with hours)
│   ├── feature_analysis_without_hours.html # Feature analysis (sensors only)
│   ├── maintenance_assistant.html      # AI chatbot interface
│   ├── n8n_integration.html           # n8n setup guide
│   ├── docs.html                       # API documentation
│   └── machine_detail.html            # Equipment details
├── 📁 static/                          # Static assets
│   ├── css/style.css                   # Custom styling
│   └── js/main.js                      # JavaScript interactions
├── 🤖 Machine Learning Models          # Trained model files
│   ├── xgb_clf.pkl                     # Failure classification model
│   ├── xgb_reg.pkl                     # RUL regression model  
│   ├── scaler.pkl                      # Feature standardization
│   ├── preprocessor.pkl                # Data preprocessing
│   └── to_drop_high_corr.pkl          # Correlation filter
├── ⚙️ Configuration Files              # Deployment configs
│   ├── pyproject.toml                  # Python dependencies
│   ├── deployment_requirements.txt     # Pip requirements
│   ├── render.yaml                     # Render.com config
│   ├── vercel.json                     # Vercel config
│   ├── Procfile                        # Heroku config
│   └── runtime.txt                     # Python version
└── 📚 Documentation
    ├── README.md                       # This file
    ├── replit.md                       # Technical architecture
    └── UYGULAMA_SEMASI.md             # Turkish documentation
```

---

## 🌐 Deployment Options

### 🚀 Render.com (Recommended)
```bash
# Automatic deployment with render.yaml
git push origin main
# Auto-deploys with PostgreSQL database
```

### 🚄 Railway.app
```bash
# Connect GitHub repository
# PostgreSQL automatically provisioned
# Environment variables auto-configured
```

### ⚡ Vercel
```bash
# Serverless deployment
vercel --prod
# Configure environment variables in dashboard
```

### 🔄 Replit (Development)
- Fork this repository in Replit
- Environment variables automatically configured
- Instant deployment with `.replit` config

---

## 🔧 Environment Variables

### Required
```bash
GEMINI_API_KEY=your_google_ai_studio_key    # For AI chatbot
SESSION_SECRET=your_secure_random_string    # Flask sessions
```

### Optional
```bash
DATABASE_URL=postgresql://...              # PostgreSQL connection
PORT=5000                                  # Application port
FLASK_ENV=production                       # Environment mode
```

---

## 📊 Model Performance

### Training Metrics
- **Failure Classification (XGBoost)**
  - Accuracy: 94.2%
  - Precision: 91.8%
  - Recall: 89.5%
  - F1-Score: 90.6%

- **RUL Regression (XGBoost)**
  - RMSE: 12.4 days
  - MAE: 8.7 days
  - R²: 0.887

### Feature Engineering
- **48 Engineered Features** including:
  - Temperature-Vibration interactions
  - Power consumption per operational hour
  - Maintenance-to-failure ratios
  - Logarithmic transformations of error codes
  - Oil-Coolant level differentials

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


## 🙏 Acknowledgments

- **XGBoost Team** - For the excellent gradient boosting framework
- **scikit-learn Team** - For machine learning preprocessing tools
- **Google AI** - For Gemini 2.5 Flash LLM integration
- **Flask Community** - For the robust web framework
- **n8n Team** - For workflow automation capabilities

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/RzaKUTLU/predictive-maintenance-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RzaKUTLU/predictive-maintenance-system/discussions)
- **Documentation**: Access `/docs` endpoint when running the application

---

<div align="center">

**Built with ❤️ using Flask, XGBoost, and Google AI**

[🔗 Live Demo](https://your-deployment-url.com) | [📖 Documentation](https://your-deployment-url.com/docs) | [🤖 Try the AI Assistant](https://your-deployment-url.com/maintenance_assistant)

</div>
