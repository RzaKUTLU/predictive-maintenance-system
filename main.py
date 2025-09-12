# main.py

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import requests
from google import genai
from google.genai import types

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Database configuration
database_url = os.environ.get("DATABASE_URL")
if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
else:
    # Fallback to SQLite for development
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"
    
db.init_app(app)

# Database Models
class Machine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.String(100), unique=True, nullable=False)
    machine_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='machine', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    
    # Input parameters
    temperature_c = db.Column(db.Float, nullable=False)
    vibration_mms = db.Column(db.Float, nullable=False)
    power_consumption_kw = db.Column(db.Float, nullable=False)
    operational_hours = db.Column(db.Float, nullable=False)
    error_codes_last_30_days = db.Column(db.Integer, nullable=False)
    oil_level_pct = db.Column(db.Float, nullable=False)
    coolant_level_pct = db.Column(db.Float, nullable=False)
    maintenance_history_count = db.Column(db.Integer, nullable=False)
    failure_history_count = db.Column(db.Integer, nullable=False)
    machine_type = db.Column(db.String(50), nullable=False)
    
    # Prediction results
    failure_within_7_days = db.Column(db.Boolean, nullable=False)
    failure_probability = db.Column(db.Float, nullable=False)
    remaining_useful_life_days = db.Column(db.Float, nullable=False)
    
    # Metadata
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))

# 📦 Model ve preprocessing nesnelerini yükle
try:
    clf_model = joblib.load('xgb_clf.pkl')
    reg_model = joblib.load('xgb_reg.pkl')
    scaler = joblib.load('scaler.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    to_drop_high_corr = joblib.load('to_drop_high_corr.pkl')
    
    # numeric_imputer.pkl dosyası yoksa None olarak ayarla
    try:
        imputer = joblib.load('numeric_imputer.pkl')
    except FileNotFoundError:
        imputer = None
        logging.warning("numeric_imputer.pkl not found, will skip imputation")
    
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    # Create dummy objects for development
    clf_model = None
    reg_model = None
    scaler = None
    imputer = None
    preprocessor = None
    to_drop_high_corr = []

# Create database tables
with app.app_context():
    db.create_all()

# 🔧 Özellik mühendisliği fonksiyonu
def perform_feature_engineering(df):
    try:
        if all(col in df.columns for col in ['Temperature_C', 'Vibration_mms']):
            df['Temp_Vib_Interaction'] = df['Temperature_C'] * df['Vibration_mms']

        if all(col in df.columns for col in ['Power_Consumption_kW', 'Operational_Hours']):
            df['Power_per_Hour'] = df['Power_Consumption_kW'] / (df['Operational_Hours'] + 1)

        if 'Error_Codes_Last_30_Days' in df.columns:
            df['Log_Error_Codes'] = np.log1p(df['Error_Codes_Last_30_Days'])

        if all(col in df.columns for col in ['Oil_Level_pct', 'Coolant_Level_pct']):
            df['Oil_Coolant_Diff'] = df['Oil_Level_pct'] - df['Coolant_Level_pct']

        if all(col in df.columns for col in ['Maintenance_History_Count', 'Failure_History_Count']):
            df['Maintenance_to_Failure_Ratio'] = df['Maintenance_History_Count'] / (df['Failure_History_Count'] + 1)
    except Exception as e:
        logging.error(f"Feature engineering error: {e}")
        
    return df

# 🧠 Categorical encoding fonksiyonu
def handle_categorical_encoding(df):
    if 'Machine_Type' in df.columns and preprocessor is not None:
        try:
            cat_df = df[['Machine_Type']]
            rest_df = df.drop(columns=['Machine_Type'])
            encoded_array = preprocessor.transform(cat_df)
            encoded_cols = preprocessor.get_feature_names_out(['Machine_Type'])
            encoded_df = pd.DataFrame(encoded_array.toarray(), columns=encoded_cols, index=df.index)
            
            # Remove cat__ prefix from column names
            encoded_df.columns = [col.replace('cat__', '') for col in encoded_df.columns]
            
            result_df = pd.concat([rest_df, encoded_df], axis=1)
            return result_df
        except Exception as e:
            logging.error(f"Categorical encoding error: {e}")
            return df
    return df

def ensure_all_features(df):
    """Ensure all required features are present for the model"""
    # Features in the exact order the model expects
    required_features = [
        'Installation_Year', 'Operational_Hours', 'Temperature_C', 'Vibration_mms', 'Sound_dB',
        'Oil_Level_pct', 'Coolant_Level_pct', 'Power_Consumption_kW', 'Last_Maintenance_Days_Ago',
        'Maintenance_History_Count', 'Failure_History_Count', 'Error_Codes_Last_30_Days',
        'AI_Override_Events', 'Power_per_Hour', 'Maintenance_to_Failure_Ratio',
        'Machine_Type_3D_Printer', 'Machine_Type_AGV', 'Machine_Type_Automated_Screwdriver',
        'Machine_Type_Boiler', 'Machine_Type_CMM', 'Machine_Type_CNC_Lathe',
        'Machine_Type_CNC_Mill', 'Machine_Type_Carton_Former', 'Machine_Type_Compressor',
        'Machine_Type_Conveyor_Belt', 'Machine_Type_Crane', 'Machine_Type_Dryer',
        'Machine_Type_Forklift_Electric', 'Machine_Type_Furnace', 'Machine_Type_Grinder',
        'Machine_Type_Heat_Exchanger', 'Machine_Type_Hydraulic_Press', 'Machine_Type_Industrial_Chiller',
        'Machine_Type_Injection_Molder', 'Machine_Type_Labeler', 'Machine_Type_Laser_Cutter',
        'Machine_Type_Mixer', 'Machine_Type_Palletizer', 'Machine_Type_Pick_and_Place',
        'Machine_Type_Press_Brake', 'Machine_Type_Pump', 'Machine_Type_Robot_Arm',
        'Machine_Type_Shrink_Wrapper', 'Machine_Type_Shuttle_System', 'Machine_Type_Vacuum_Packer',
        'Machine_Type_Valve_Controller', 'Machine_Type_Vision_System', 'Machine_Type_XRay_Inspector'
    ]
    
    # Add missing features with default values
    for feature in required_features:
        if feature not in df.columns:
            if feature == 'Sound_dB':
                df[feature] = 50.0  # Default sound level
            else:
                df[feature] = 0
    
    # Ensure features are in the correct order
    df = df[required_features]
    return df

def make_prediction(input_data):
    """Helper function to make predictions"""
    if clf_model is None or reg_model is None:
        # Model dosyaları mevcut değilse demo tahmin oluştur
        return generate_demo_prediction(input_data)
    
    # 📥 Convert to DataFrame
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else pd.DataFrame(input_data)

    # 🧠 Özellik mühendisliği + ön işlem
    df = perform_feature_engineering(df)
    if 'Machine_ID' in df.columns:
        df.drop(columns=['Machine_ID'], inplace=True)
    df.drop(columns=[col for col in to_drop_high_corr if col in df.columns], inplace=True)
    
    # Çalışma saati normal kullanım

    # 🧼 Eksik değer doldurma
    numeric_cols = df.select_dtypes(include=np.number).columns
    if imputer is not None:
        df[numeric_cols] = imputer.transform(df[numeric_cols])

    # 🔡 Kategorik kodlama
    df = handle_categorical_encoding(df)
    
    # Ensure all required features are present
    df = ensure_all_features(df)

    # ⚖️ Ölçekleme - Critical for model performance
    if scaler is not None:
        try:
            df_scaled = scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
            logging.info("Features scaled successfully")
        except Exception as e:
            logging.error(f"Scaling error: {e}")
    else:
        logging.warning("Scaler not available, using raw features")

    # 📊 Tahmin
    classification_pred = clf_model.predict(df)[0]
    classification_proba = clf_model.predict_proba(df)[0][1]
    regression_pred = reg_model.predict(df)[0]

    # Prepare result with original Machine_ID and Machine_Type
    result = {
        "Failure_Within_7_Days": bool(classification_pred),
        "Failure_Probability": round(float(classification_proba), 4),
        "Remaining_Useful_Life_days": round(float(regression_pred), 2)
    }
    
    # Add Machine_ID and Machine_Type if present in input
    if 'Machine_ID' in input_data:
        result['Machine_ID'] = input_data['Machine_ID']
    if 'Machine_Type' in input_data:
        result['Machine_Type'] = input_data['Machine_Type']
    
    return result

def get_feature_importance():
    """Get feature importance from the trained models"""
    if clf_model is None or reg_model is None:
        return None
    
    try:
        # Get feature names from a sample prediction
        sample_data = {
            'Machine_Type': '3D_Printer',
            'Temperature_C': 40.0,
            'Vibration_mms': 2.0,
            'Power_Consumption_kW': 80.0,
            'Operational_Hours': 1000,
            'Error_Codes_Last_30_Days': 0,
            'Oil_Level_pct': 90.0,
            'Coolant_Level_pct': 85.0,
            'Maintenance_History_Count': 10,
            'Failure_History_Count': 0,
            'AI_Override_Events': 0,
            'Installation_Year': 2020,
            'Last_Maintenance_Days_Ago': 30,
            'Sound_dB': 60.0
        }
        
        # Process sample to get feature names
        df = pd.DataFrame([sample_data])
        df = perform_feature_engineering(df)
        if 'Machine_ID' in df.columns:
            df.drop(columns=['Machine_ID'], inplace=True)
        df.drop(columns=[col for col in to_drop_high_corr if col in df.columns], inplace=True)
        
        # Çalışma saati normal kullanım
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        if imputer is not None:
            df[numeric_cols] = imputer.transform(df[numeric_cols])
        
        df = handle_categorical_encoding(df)
        df = ensure_all_features(df)
        
        # Get feature importance
        clf_importance = clf_model.feature_importances_
        reg_importance = reg_model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': df.columns,
            'classification_importance': clf_importance,
            'regression_importance': reg_importance,
            'average_importance': (clf_importance + reg_importance) / 2
        }).sort_values('average_importance', ascending=False)
        
        # Group machine type features together
        machine_type_features = [col for col in importance_df['feature'] if col.startswith('Machine_Type_')]
        other_features = [col for col in importance_df['feature'] if not col.startswith('Machine_Type_')]
        
        # Calculate total machine type importance
        machine_type_importance = importance_df[importance_df['feature'].isin(machine_type_features)]
        total_machine_type_clf = machine_type_importance['classification_importance'].sum()
        total_machine_type_reg = machine_type_importance['regression_importance'].sum()
        total_machine_type_avg = (total_machine_type_clf + total_machine_type_reg) / 2
        
        # Get non-machine-type features
        other_importance = importance_df[importance_df['feature'].isin(other_features)]
        
        # Create final result
        result_list = []
        
        # Add machine type as single grouped feature
        if len(machine_type_features) > 0:
            result_list.append({
                'feature': 'Machine_Type (Toplam Etkisi)',
                'classification_importance': float(total_machine_type_clf),
                'regression_importance': float(total_machine_type_reg),
                'average_importance': float(total_machine_type_avg),
                'description': 'Makine tipinin toplam etkisi (tüm makine tipleri)'
            })
        
        # Add other features
        for _, row in other_importance.iterrows():
            feature_name = row['feature']
            # Better Turkish names for features
            if feature_name == 'Operational_Hours':
                display_name = 'Çalışma Saati'
                description = 'Makinenin toplam çalışma saati'
            elif feature_name == 'Temperature_C':
                display_name = 'Sıcaklık'
                description = 'Makine çalışma sıcaklığı (°C)'
            elif feature_name == 'Vibration_mms':
                display_name = 'Titreşim'
                description = 'Makine titreşim seviyesi (mm/s)'
            elif feature_name == 'Power_Consumption_kW':
                display_name = 'Güç Tüketimi'
                description = 'Anlık güç tüketimi (kW)'
            elif feature_name == 'Oil_Level_pct':
                display_name = 'Yağ Seviyesi'
                description = 'Yağ seviyesi yüzdesi (%)'
            elif feature_name == 'Coolant_Level_pct':
                display_name = 'Soğutucu Seviyesi'
                description = 'Soğutucu seviyesi yüzdesi (%)'
            elif feature_name == 'Error_Codes_Last_30_Days':
                display_name = 'Hata Kodu Sayısı'
                description = 'Son 30 gündeki hata kodu sayısı'
            elif feature_name == 'Maintenance_History_Count':
                display_name = 'Bakım Geçmişi'
                description = 'Toplam bakım sayısı'
            elif feature_name == 'Failure_History_Count':
                display_name = 'Arıza Geçmişi'
                description = 'Toplam arıza sayısı'
            elif feature_name == 'Last_Maintenance_Days_Ago':
                display_name = 'Son Bakımdan Bu Yana'
                description = 'Son bakımdan geçen gün sayısı'
            elif feature_name == 'Sound_dB':
                display_name = 'Ses Seviyesi'
                description = 'Makine ses seviyesi (dB)'
            elif feature_name == 'Power_per_Hour':
                display_name = 'Saatlik Güç Tüketimi'
                description = 'Saat başına güç tüketimi (kW/h)'
            elif feature_name == 'Maintenance_to_Failure_Ratio':
                display_name = 'Bakım/Arıza Oranı'
                description = 'Bakım sayısının arıza sayısına oranı'
            elif feature_name == 'Installation_Year':
                display_name = 'Kurulum Yılı'
                description = 'Makinenin kurulum yılı'
            elif feature_name == 'AI_Override_Events':
                display_name = 'AI Müdahale Sayısı'
                description = 'AI müdahale olay sayısı'
            else:
                display_name = feature_name
                description = f'{feature_name} özelliğinin etkisi'
            
            result_list.append({
                'feature': display_name,
                'classification_importance': float(row['classification_importance']),
                'regression_importance': float(row['regression_importance']),
                'average_importance': float(row['average_importance']),
                'description': description
            })
        
        # Sort by average importance and return top 15
        result_list.sort(key=lambda x: x['average_importance'], reverse=True)
        return result_list[:15]
        
    except Exception as e:
        logging.error(f"Feature importance error: {e}")
        return None

def generate_demo_prediction(input_data):
    """Demo tahmin oluşturucu - gerçek modeller yüklenene kadar"""
    
    # Giriş parametrelerine dayalı basit risk hesaplama
    temp = input_data.get('Temperature_C', 0)
    vibration = input_data.get('Vibration_mms', 0)
    power = input_data.get('Power_Consumption_kW', 0)
    hours = input_data.get('Operational_Hours', 0)
    errors = input_data.get('Error_Codes_Last_30_Days', 0)
    oil_level = input_data.get('Oil_Level_pct', 100)
    coolant_level = input_data.get('Coolant_Level_pct', 100)
    ai_overrides = input_data.get('AI_Override_Events', 0)
    install_year = input_data.get('Installation_Year', 2020)
    last_maintenance = input_data.get('Last_Maintenance_Days_Ago', 30)
    
    # Risk skoru hesapla (0-1 arası)
    risk_score = 0.0
    
    # Sıcaklık riski (optimal: 20-80°C)
    if temp > 90 or temp < 10:
        risk_score += 0.3
    elif temp > 80 or temp < 20:
        risk_score += 0.1
    
    # Titreşim riski (yüksek titreşim kötü)
    if vibration > 8:
        risk_score += 0.4
    elif vibration > 5:
        risk_score += 0.2
    
    # Hata kodu riski
    if errors > 10:
        risk_score += 0.3
    elif errors > 5:
        risk_score += 0.1
    
    # Yağ seviyesi riski
    if oil_level < 30:
        risk_score += 0.3
    elif oil_level < 50:
        risk_score += 0.1
    
    # Soğutucu seviyesi riski
    if coolant_level < 30:
        risk_score += 0.2
    elif coolant_level < 50:
        risk_score += 0.1
    
    # Çalışma saati riski (çok yüksek saat kötü)
    if hours > 50000:
        risk_score += 0.2
    elif hours > 30000:
        risk_score += 0.1
    
    # AI override riski
    if ai_overrides > 5:
        risk_score += 0.2
    elif ai_overrides > 2:
        risk_score += 0.1
    
    # Yaş riski (eski makineler daha riskli)
    machine_age = 2025 - install_year
    if machine_age > 10:
        risk_score += 0.15
    elif machine_age > 5:
        risk_score += 0.05
    
    # Son bakım riski
    if last_maintenance > 90:
        risk_score += 0.25
    elif last_maintenance > 60:
        risk_score += 0.15
    elif last_maintenance > 30:
        risk_score += 0.05
    
    # Risk skorunu sınırla
    risk_score = min(risk_score, 0.95)
    
    # 7 gün içinde arıza tahmini
    failure_prediction = risk_score > 0.5
    
    # Kalan kullanışlı ömür (risk ne kadar yüksekse o kadar az)
    remaining_life = max(5, int(100 * (1 - risk_score)))
    
    return {
        "Failure_Within_7_Days": failure_prediction,
        "Failure_Probability": round(risk_score, 4),
        "Remaining_Useful_Life_days": remaining_life
    }

# Web Routes

@app.route('/')
def index():
    """Main web interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        # Fallback if template fails
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Makine Arıza Tahmini API</title></head>
        <body>
            <h1>Makine Arıza Tahmini API</h1>
            <p>API çalışıyor! Test için: <a href="/predict">/predict</a></p>
            <p>Dokümantasyon: <a href="/docs">/docs</a></p>
            <p>n8n Entegrasyonu: <a href="/n8n_integration">/n8n_integration</a></p>
        </body>
        </html>
        """

@app.route('/docs')
def docs():
    """API documentation page"""
    return render_template('docs.html')

@app.route('/history')
def history():
    """View prediction history"""
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.order_by(Prediction.prediction_date.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template('history.html', predictions=predictions)

@app.route('/machine/<machine_id>')
def machine_detail(machine_id):
    """View specific machine details and history"""
    machine = Machine.query.filter_by(machine_id=machine_id).first_or_404()
    predictions = Prediction.query.filter_by(machine_id=machine.id).order_by(
        Prediction.prediction_date.desc()
    ).limit(50).all()
    return render_template('machine_detail.html', machine=machine, predictions=predictions)

@app.route('/n8n')
def n8n_integration():
    """n8n Integration guide page"""
    return render_template('n8n_integration.html')

def save_prediction_to_db(form_data, result):
    """Save prediction to database"""
    try:
        machine_id_str = request.form.get('machine_id', f"WEB_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        # Find or create machine
        machine = Machine.query.filter_by(machine_id=machine_id_str).first()
        if not machine:
            machine = Machine(
                machine_id=machine_id_str,
                machine_type=form_data['Machine_Type']
            )
            db.session.add(machine)
            db.session.commit()
        
        # Create prediction record
        prediction = Prediction(
            machine_id=machine.id,
            temperature_c=form_data['Temperature_C'],
            vibration_mms=form_data['Vibration_mms'],
            power_consumption_kw=form_data['Power_Consumption_kW'],
            operational_hours=form_data['Operational_Hours'],
            error_codes_last_30_days=form_data['Error_Codes_Last_30_Days'],
            oil_level_pct=form_data['Oil_Level_pct'],
            coolant_level_pct=form_data['Coolant_Level_pct'],
            maintenance_history_count=form_data['Maintenance_History_Count'],
            failure_history_count=form_data['Failure_History_Count'],
            machine_type=form_data['Machine_Type'],
            failure_within_7_days=result['Failure_Within_7_Days'],
            failure_probability=result['Failure_Probability'],
            remaining_useful_life_days=result['Remaining_Useful_Life_days'],
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', '')[:500]
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        return prediction.id
        
    except Exception as e:
        logging.error(f"Database save error: {e}")
        db.session.rollback()
        return None

@app.route('/predict_web', methods=['POST'])
def predict_web():
    """Web form prediction handler"""
    try:
        # Get form data
        form_data = {
            'Temperature_C': float(request.form.get('temperature', 0)),
            'Vibration_mms': float(request.form.get('vibration', 0)),
            'Power_Consumption_kW': float(request.form.get('power_consumption', 0)),
            'Operational_Hours': float(request.form.get('operational_hours', 0)),
            'Error_Codes_Last_30_Days': int(request.form.get('error_codes', 0)),
            'Oil_Level_pct': float(request.form.get('oil_level', 0)),
            'Coolant_Level_pct': float(request.form.get('coolant_level', 0)),
            'Maintenance_History_Count': int(request.form.get('maintenance_count', 0)),
            'Failure_History_Count': int(request.form.get('failure_count', 0)),
            'Machine_Type': request.form.get('machine_type', '3D_Printer'),
            'AI_Override_Events': int(request.form.get('ai_override_events', 0)),
            'Installation_Year': int(request.form.get('installation_year', 2020)),
            'Last_Maintenance_Days_Ago': int(request.form.get('last_maintenance_days_ago', 30)),
            'Sound_dB': float(request.form.get('sound_db', 50.0))
        }
        
        # Make prediction
        result = make_prediction(form_data)
        
        # Save to database
        prediction_id = save_prediction_to_db(form_data, result)
        
        return render_template('predict.html', 
                             result=result, 
                             form_data=form_data,
                             prediction_id=prediction_id,
                             success=True)
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        flash(f"Prediction failed: {str(e)}", 'danger')
        return redirect(url_for('index'))

# API Routes (existing functionality)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint - supports both single and batch predictions"""
    try:
        # Get JSON data
        input_data = request.get_json()
        
        # Check if input is a list (batch) or single object
        if isinstance(input_data, list):
            # Batch processing
            results = []
            for i, single_data in enumerate(input_data):
                try:
                    result = make_prediction(single_data)
                    result['batch_index'] = i
                    results.append(result)
                except Exception as e:
                    error_result = {
                        'batch_index': i,
                        'error': str(e),
                        'Failure_Within_7_Days': None,
                        'Failure_Probability': None,
                        'Remaining_Useful_Life_days': None
                    }
                    # Add Machine_ID and Machine_Type to error results too
                    if 'Machine_ID' in single_data:
                        error_result['Machine_ID'] = single_data['Machine_ID']
                    if 'Machine_Type' in single_data:
                        error_result['Machine_Type'] = single_data['Machine_Type']
                    results.append(error_result)
            
            # Return results directly as array for easier n8n processing
            return jsonify(results)
        else:
            # Single prediction (existing functionality)
            result = make_prediction(input_data)
            return jsonify(result)

    except Exception as e:
        logging.error(f"API prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api_status')
def api_status():
    """API status check"""
    models_loaded = all([
        clf_model is not None,
        reg_model is not None,
        scaler is not None,
        imputer is not None,
        preprocessor is not None
    ])
    
    return jsonify({
        "status": "running",
        "models_loaded": models_loaded,
        "message": "🔧 Predictive Maintenance API is Running!"
    })

@app.route('/feature_importance')
def feature_importance():
    """Get feature importance analysis"""
    importance_data = get_feature_importance()
    
    if importance_data is None:
        return jsonify({"error": "Model not available for feature importance analysis"}), 500
    
    return jsonify({
        "feature_importance": importance_data,
        "explanation": {
            "classification_importance": "En çok arıza tahminini etkileyen özellikler",
            "regression_importance": "En çok kalan ömür tahminini etkileyen özellikler", 
            "average_importance": "Genel olarak en etkili özellikler"
        }
    })

@app.route('/feature_analysis')
def feature_analysis():
    """Feature importance analysis page"""
    return render_template('feature_importance.html')

def make_prediction_without_hours(input_data):
    """Operational_Hours olmadan tahmin yapma fonksiyonu"""
    try:
        # DataFrame oluştur
        df = pd.DataFrame([input_data])
        
        # Feature engineering
        df = perform_feature_engineering(df)
        
        # Categorical encoding
        df = handle_categorical_encoding(df)
        
        # Machine_ID'yi çıkar
        if 'Machine_ID' in df.columns:
            df.drop(columns=['Machine_ID'], inplace=True)
        
        # High correlation features'ı çıkar
        df.drop(columns=[col for col in to_drop_high_corr if col in df.columns], inplace=True)
        
        # Operational_Hours'u sabit değer ile değiştir (veri sızıntısını önlemek için)
        if 'Operational_Hours' in df.columns:
            df['Operational_Hours'] = 25000  # Orta seviye değer
            logging.info("Operational_Hours set to constant value (25000) to prevent data leakage")
        
        # Tüm özellikleri ensure et
        df = ensure_all_features(df)
        
        # Eksik değer doldurma
        numeric_cols = df.select_dtypes(include=np.number).columns
        if imputer is not None:
            df[numeric_cols] = imputer.transform(df[numeric_cols])
        
        # Scaling
        if scaler is not None:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
            logging.info("Features scaled successfully (without hours)")
        
        # Prediction
        global clf_model, reg_model
        failure_prob = float(clf_model.predict_proba(df)[0][1])
        failure_within_7_days = bool(clf_model.predict(df)[0])
        remaining_life = float(reg_model.predict(df)[0])
        
        return {
            'Machine_ID': input_data.get('Machine_ID', 'Unknown'),
            'Machine_Type': input_data.get('Machine_Type', 'Unknown'),
            'Failure_Probability': failure_prob,
            'Failure_Within_7_Days': failure_within_7_days,
            'Remaining_Useful_Life_days': remaining_life,
            'Note': 'Prediction made without Operational_Hours'
        }
        
    except Exception as e:
        logging.error(f"Prediction without hours error: {str(e)}")
        raise e

def get_feature_importance_without_hours():
    """Operational_Hours olmadan feature importance analizi - Farklı değerlerle test edilerek"""
    try:
        # Farklı makine durumları için test örnekleri
        test_samples = [
            {
                'Machine_ID': 'SAMPLE_001',
                'Machine_Type': 'CNC_Lathe',
                'Temperature_C': 45.0,
                'Vibration_mms': 2.5,
                'Power_Consumption_kW': 80.0,
                'Operational_Hours': 25000,
                'Error_Codes_Last_30_Days': 1,
                'Oil_Level_pct': 90.0,
                'Coolant_Level_pct': 85.0,
                'Maintenance_History_Count': 10,
                'Failure_History_Count': 0,
                'AI_Override_Events': 0,
                'Installation_Year': 2020,
                'Last_Maintenance_Days_Ago': 30,
                'Sound_dB': 55.0
            },
            {
                'Machine_ID': 'SAMPLE_002',
                'Machine_Type': '3D_Printer',
                'Temperature_C': 85.0,
                'Vibration_mms': 8.5,
                'Power_Consumption_kW': 120.0,
                'Operational_Hours': 25000,
                'Error_Codes_Last_30_Days': 8,
                'Oil_Level_pct': 30.0,
                'Coolant_Level_pct': 25.0,
                'Maintenance_History_Count': 2,
                'Failure_History_Count': 4,
                'AI_Override_Events': 5,
                'Installation_Year': 2015,
                'Last_Maintenance_Days_Ago': 180,
                'Sound_dB': 95.0
            },
            {
                'Machine_ID': 'SAMPLE_003',
                'Machine_Type': 'Robot_Arm',
                'Temperature_C': 65.0,
                'Vibration_mms': 5.0,
                'Power_Consumption_kW': 100.0,
                'Operational_Hours': 25000,
                'Error_Codes_Last_30_Days': 4,
                'Oil_Level_pct': 60.0,
                'Coolant_Level_pct': 55.0,
                'Maintenance_History_Count': 6,
                'Failure_History_Count': 1,
                'AI_Override_Events': 2,
                'Installation_Year': 2018,
                'Last_Maintenance_Days_Ago': 60,
                'Sound_dB': 75.0
            }
        ]
        
        # Her sample için tahmin yap ve feature importance'ı simulate et
        predictions = []
        for sample in test_samples:
            pred = make_prediction_without_hours(sample)
            predictions.append({
                'sample': sample,
                'prediction': pred
            })
        
        # Feature names'i almak için bir sample process et
        df = pd.DataFrame([test_samples[0]])
        df = perform_feature_engineering(df)
        df = handle_categorical_encoding(df)
        if 'Machine_ID' in df.columns:
            df.drop(columns=['Machine_ID'], inplace=True)
        df.drop(columns=[col for col in to_drop_high_corr if col in df.columns], inplace=True)
        
        # Operational_Hours'u sabit değer ile değiştir
        if 'Operational_Hours' in df.columns:
            df['Operational_Hours'] = 25000
        
        df = ensure_all_features(df)
        
        # Gerçek model importance'ını al ama Operational_Hours'u minimize et
        global clf_model, reg_model
        clf_importance = clf_model.feature_importances_
        reg_importance = reg_model.feature_importances_
        
        # Calculate average importance
        avg_importance = (clf_importance + reg_importance) / 2
        
        # Create feature importance list
        feature_importance = []
        for i, feature in enumerate(df.columns):
            importance_val = float(avg_importance[i])
            
            # Operational_Hours'u minimize et (sıfırla)
            if feature == 'Operational_Hours':
                importance_val = 0.0
            
            feature_importance.append({
                'feature': feature,
                'classifier_importance': float(clf_importance[i]) if feature != 'Operational_Hours' else 0.0,
                'regressor_importance': float(reg_importance[i]) if feature != 'Operational_Hours' else 0.0,
                'average_importance': importance_val
            })
        
        # Remove Operational_Hours from list
        feature_importance = [f for f in feature_importance if f['feature'] != 'Operational_Hours']
        
        # Re-normalize importance values
        total_importance = sum(f['average_importance'] for f in feature_importance)
        if total_importance > 0:
            for f in feature_importance:
                f['average_importance'] = f['average_importance'] / total_importance
                f['classifier_importance'] = f['classifier_importance'] / total_importance
                f['regressor_importance'] = f['regressor_importance'] / total_importance
        
        # Group machine type features together (same logic as main function)
        machine_type_features = [f for f in feature_importance if f['feature'].startswith('Machine_Type_')]
        other_features = [f for f in feature_importance if not f['feature'].startswith('Machine_Type_')]
        
        # Calculate total machine type importance
        total_machine_type_clf = sum(f['classifier_importance'] for f in machine_type_features)
        total_machine_type_reg = sum(f['regressor_importance'] for f in machine_type_features)
        total_machine_type_avg = sum(f['average_importance'] for f in machine_type_features)
        
        # Create final result
        result_list = []
        
        # Add machine type as single grouped feature
        if len(machine_type_features) > 0:
            result_list.append({
                'feature': 'Machine_Type (Toplam Etkisi)',
                'classifier_importance': float(total_machine_type_clf),
                'regressor_importance': float(total_machine_type_reg),
                'average_importance': float(total_machine_type_avg),
                'description': 'Makine tipinin toplam etkisi (tüm makine tipleri)'
            })
        
        # Add other features with Turkish names
        for f in other_features:
            feature_name = f['feature']
            # Better Turkish names for features
            if feature_name == 'Temperature_C':
                display_name = 'Sıcaklık'
                description = 'Makine çalışma sıcaklığı (°C)'
            elif feature_name == 'Vibration_mms':
                display_name = 'Titreşim'
                description = 'Makine titreşim seviyesi (mm/s)'
            elif feature_name == 'Power_Consumption_kW':
                display_name = 'Güç Tüketimi'
                description = 'Anlık güç tüketimi (kW)'
            elif feature_name == 'Oil_Level_pct':
                display_name = 'Yağ Seviyesi'
                description = 'Yağ seviyesi yüzdesi (%)'
            elif feature_name == 'Coolant_Level_pct':
                display_name = 'Soğutucu Seviyesi'
                description = 'Soğutucu seviyesi yüzdesi (%)'
            elif feature_name == 'Error_Codes_Last_30_Days':
                display_name = 'Hata Kodu Sayısı'
                description = 'Son 30 gündeki hata kodu sayısı'
            elif feature_name == 'Maintenance_History_Count':
                display_name = 'Bakım Geçmişi'
                description = 'Toplam bakım sayısı'
            elif feature_name == 'Failure_History_Count':
                display_name = 'Arıza Geçmişi'
                description = 'Toplam arıza sayısı'
            elif feature_name == 'Last_Maintenance_Days_Ago':
                display_name = 'Son Bakımdan Bu Yana'
                description = 'Son bakımdan geçen gün sayısı'
            elif feature_name == 'Sound_dB':
                display_name = 'Ses Seviyesi'
                description = 'Makine ses seviyesi (dB)'
            elif feature_name == 'Power_per_Hour':
                display_name = 'Saatlik Güç Tüketimi'
                description = 'Saat başına güç tüketimi (kW/h)'
            elif feature_name == 'Maintenance_to_Failure_Ratio':
                display_name = 'Bakım/Arıza Oranı'
                description = 'Bakım sayısının arıza sayısına oranı'
            elif feature_name == 'Installation_Year':
                display_name = 'Kurulum Yılı'
                description = 'Makinenin kurulum yılı'
            elif feature_name == 'AI_Override_Events':
                display_name = 'AI Müdahale Sayısı'
                description = 'AI müdahale olay sayısı'
            else:
                display_name = feature_name
                description = f'{feature_name} özelliğinin etkisi'
            
            result_list.append({
                'feature': display_name,
                'classifier_importance': float(f['classifier_importance']),
                'regressor_importance': float(f['regressor_importance']),
                'average_importance': float(f['average_importance']),
                'description': description
            })
        
        # Sort by average importance
        result_list.sort(key=lambda x: x['average_importance'], reverse=True)
        
        return {
            'feature_importance': result_list[:15],
            'total_features': len(result_list),
            'note': 'Feature importance analysis without Operational_Hours - Normalized and Grouped'
        }
        
    except Exception as e:
        logging.error(f"Feature importance without hours error: {str(e)}")
        raise e

@app.route('/predict_without_hours', methods=['POST'])
def predict_without_hours():
    """API endpoint - Operational_Hours olmadan tahmin (veri sızıntısını önlemek için)"""
    try:
        # JSON verisi al
        if request.is_json:
            data = request.get_json()
            if isinstance(data, list):
                # Batch prediction
                results = []
                for item in data:
                    result = make_prediction_without_hours(item)
                    results.append(result)
                return jsonify(results)
            else:
                # Single prediction
                result = make_prediction_without_hours(data)
                return jsonify(result)
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/feature_importance_without_hours')
def feature_importance_without_hours():
    """Feature importance analysis without Operational_Hours"""
    try:
        importance_data = get_feature_importance_without_hours()
        return jsonify(importance_data)
    except Exception as e:
        logging.error(f"Feature importance error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feature_analysis_without_hours')
def feature_analysis_without_hours():
    """Feature importance analysis page without Operational_Hours"""
    return render_template('feature_analysis_without_hours.html')

# Google Sheets data reading function
def read_google_sheets_data(sheet_url):
    """Read maintenance data from Google Sheets"""
    try:
        # Convert Google Sheets URL to CSV export URL
        sheet_id = sheet_url.split('/d/')[1].split('/')[0]
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Read CSV data with proper encoding
        response = requests.get(csv_url)
        if response.status_code == 200:
            # Try different encodings for Turkish characters
            import io
            try:
                # First try UTF-8
                data = pd.read_csv(io.StringIO(response.text), encoding='utf-8')
            except:
                try:
                    # Try with latin-1 if UTF-8 fails
                    response.encoding = 'latin-1'
                    data = pd.read_csv(io.StringIO(response.text), encoding='latin-1')
                except:
                    # Fallback to default
                    data = pd.read_csv(io.StringIO(response.text))
            
            # Data loaded successfully
            
            return data
        else:
            return None
    except Exception as e:
        logging.error(f"Error reading Google Sheets: {str(e)}")
        return None

# Maintenance chatbot function
def maintenance_chatbot(user_question, maintenance_data):
    """Generate maintenance plan advice using Gemini"""
    try:
        # Prepare data summary for context
        data_summary = ""
        if maintenance_data is not None and not maintenance_data.empty:
            # Arıza riski var olan makineleri belirt (True boolean değerleri)
            risk_machines = pd.DataFrame()
            if 'tahmin' in maintenance_data.columns:
                risk_machines = maintenance_data[maintenance_data['tahmin'] == True]
            data_summary = f"""
Mevcut bakım verileri:
- Toplam kayıt sayısı: {len(maintenance_data)}
- Veri sütunları: {', '.join(maintenance_data.columns.tolist())}
- ACİL BAKIM GEREKEN (Arıza Riski Var): {len(risk_machines)} makine
- Son 5 kayıt:
{maintenance_data.tail().to_string()}

ACİL BAKIM GEREKEN MAKİNELER:
{risk_machines.to_string() if not risk_machines.empty else 'Yok'}
"""
        
        # Create system prompt
        system_prompt = f"""
Sen endüstriyel bakım uzmanısın. Kullanıcıların bakım sorularını cevaplayıp, makine verilerini analiz ediyorsun.

MEVCUT VERİLER:
{data_summary}

GÖREVIN: Kullanıcının sorusuna göre uygun analiz ve öneriler yap.

SORU TİPLERİ VE CEVAP STİLLERİ:
- Sıralama soruları: Machine_ID, tür, kalan ömür listesi
- Acil durum soruları: Kritik makineleri belirt
- Genel bakım soruları: Kısa pratik öneriler
- Belirli makine soruları: O makineye özgü bilgi

KURALLER:
- Tahmin=True olan makineler her zaman öncelikli
- Kalan kullanım ömrü düşük olanlar kritik
- Maksimum 150 kelime, net ve anlaşılır cevap ver

Kullanıcı sorusu: {user_question}
"""
        
        # Generate response with Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt
        )
        
        return response.text if response.text else "Üzgünüm, şu anda cevap veremiyorum. Lütfen tekrar deneyin."
        
    except Exception as e:
        logging.error(f"Chatbot error: {str(e)}")
        return f"Hata oluştu: {str(e)}"

@app.route('/maintenance_assistant')
def maintenance_assistant():
    """Maintenance assistant chatbot page"""
    return render_template('maintenance_assistant.html')

@app.route('/maintenance_chat', methods=['POST'])
def maintenance_chat():
    """Handle maintenance chatbot requests"""
    try:
        data = request.get_json()
        user_question = data.get('question', '')
        
        if not user_question:
            return jsonify({'error': 'Soru boş olamaz'}), 400
        
        # Google Sheets URL
        sheets_url = "https://docs.google.com/spreadsheets/d/1koE-uGlrOVn4ownUS86caT0Ws4x5bUYcXaf-TAjPgNY/edit?usp=sharing"
        
        # Read maintenance data
        maintenance_data = read_google_sheets_data(sheets_url)
        
        # Generate chatbot response
        response = maintenance_chatbot(user_question, maintenance_data)
        
        return jsonify({
            'response': response,
            'data_available': maintenance_data is not None and not maintenance_data.empty,
            'data_rows': len(maintenance_data) if maintenance_data is not None else 0
        })
        
    except Exception as e:
        logging.error(f"Maintenance chat error: {str(e)}")
        return jsonify({'error': f'Hata: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
