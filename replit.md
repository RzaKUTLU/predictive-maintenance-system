# Predictive Maintenance System

## Overview

This is a Flask-based web application that provides predictive maintenance capabilities for industrial machinery using machine learning models. The system predicts equipment failures and estimates remaining useful life to enable proactive maintenance scheduling.

## System Architecture

### Backend Architecture
- **Framework**: Flask (Python 3.11+)
- **Database**: PostgreSQL (with SQLite fallback for development)
- **ORM**: SQLAlchemy with declarative base
- **Web Server**: Gunicorn for production deployment
- **ML Stack**: XGBoost, scikit-learn, pandas, numpy

### Frontend Architecture
- **Template Engine**: Jinja2 with Flask
- **CSS Framework**: Bootstrap (dark theme)
- **Icons**: Font Awesome
- **JavaScript**: Vanilla JS for form handling and UI interactions

### Database Schema
- **Machine Table**: Stores machine information (ID, type, creation date)
- **Prediction Table**: Stores prediction results with foreign key to machines
- **Relationships**: One-to-many relationship between machines and predictions

## Key Components

### Core Models
1. **Machine Learning Models**
   - XGBoost Classifier for failure prediction
   - XGBoost Regressor for remaining useful life estimation
   - StandardScaler for feature normalization
   - Numeric imputer for handling missing values
   - Preprocessor for categorical encoding

2. **Feature Engineering Pipeline**
   - Temperature-Vibration interaction features
   - Power consumption per operational hour
   - Logarithmic transformation of error codes
   - Oil-Coolant level differential
   - Maintenance-to-failure ratio calculation

3. **Database Models**
   - Machine model with unique machine_id constraint
   - Prediction model with foreign key relationships
   - Automatic timestamp tracking

### API Endpoints
- **POST /predict**: Main prediction endpoint accepting machine parameters (includes operational hours)
- **POST /predict_without_hours**: Alternative prediction endpoint excluding operational hours (uses fixed 25,000 hours)
- **GET /**: Web interface for manual predictions
- **GET /history**: Prediction history with pagination
- **GET /docs**: API documentation
- **GET /n8n_integration**: n8n integration guide with dual API support
- **GET /feature_analysis**: Feature importance analysis with operational hours
- **GET /feature_analysis_without_hours**: Feature importance analysis without operational hours
- **GET /maintenance_assistant**: AI-powered maintenance planning chatbot
- **POST /maintenance_chat**: Chatbot endpoint for maintenance advice using Gemini AI

### Web Interface
- Multi-language support (Turkish/English)
- Responsive design with Bootstrap
- Real-time form validation
- Interactive prediction results display
- Historical data visualization
- AI-powered maintenance assistant chatbot
- Google Sheets data integration for maintenance planning

## Data Flow

1. **Input Processing**
   - Accept JSON data via REST API or web form
   - Validate required parameters
   - Create pandas DataFrame from input

2. **Feature Engineering**
   - Apply mathematical transformations
   - Create interaction features
   - Handle categorical variables

3. **Model Pipeline**
   - Impute missing values
   - Apply feature scaling
   - Remove highly correlated features
   - Generate predictions using trained models

4. **Output Generation**
   - Failure probability calculation
   - Remaining useful life estimation
   - Risk classification (7-day failure window)
   - Database storage of results

## External Dependencies

### Python Packages
- **Flask**: Web framework and routing
- **SQLAlchemy**: Database ORM
- **XGBoost**: Gradient boosting ML models
- **scikit-learn**: ML preprocessing and utilities
- **pandas/numpy**: Data manipulation
- **psycopg2-binary**: PostgreSQL adapter
- **gunicorn**: WSGI HTTP server
- **google-genai**: Google Gemini AI integration
- **requests**: HTTP client for Google Sheets integration

### Frontend Dependencies
- **Bootstrap**: CSS framework via CDN
- **Font Awesome**: Icon library via CDN
- **Custom CSS/JS**: Local static files

### Infrastructure
- **PostgreSQL**: Primary database (auto-configured via DATABASE_URL)
- **SQLite**: Development fallback database
- **Replit**: Hosting platform with auto-scaling

## Deployment Strategy

### Production Configuration
- **Server**: Gunicorn with auto-scaling deployment target
- **Database**: PostgreSQL with connection pooling
- **Environment**: Nix package manager for consistent dependencies
- **Port Configuration**: Internal port 5000, external port 80

### Development Setup
- **Local Database**: SQLite for development
- **Hot Reload**: Gunicorn with --reload flag
- **Debug Mode**: Enabled logging for troubleshooting

### Environment Variables
- **DATABASE_URL**: PostgreSQL connection string (production)
- **SESSION_SECRET**: Flask session security key

## Changelog
- June 27, 2025. Initial setup
- July 8, 2025. Complete Turkish translation implemented
- July 8, 2025. Deployment options researched for continuous hosting
- July 8, 2025. N8N integration guide updated with dual API support for operational hours
- July 9, 2025. AI-powered maintenance assistant chatbot added with Google Sheets integration

## User Preferences

Preferred communication style: Simple, everyday language.
Hosting preference: Seeking free hosting alternatives to Replit for 24/7 availability.
Priority: 24/7 uptime requirement for production use.
Custom URL preference: Wants Turkish-friendly URL like "makinearızatahmini" for better accessibility.