import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Config
MODEL_PATH = 'xgboost_model.pkl'
DATA_URL = "https://raw.githubusercontent.com/JunTheCoder62/Final-Project-Menyelesaikan-Permasalahan-Institusi-Pendidikan/refs/heads/main/Data/new_data.csv"  # Sample dataset

@st.cache_data
def load_sample_data():
    """Load sample student dropout dataset"""
    try:
        df = pd.read_csv(DATA_URL, sep=';')
        st.success(f"✅ Dataset berhasil dimuat: {len(df)} baris")
        return df
    except:
        st.error("❌ Gagal memuat dataset. Gunakan data lokal.")
        return None

@st.cache_resource
def load_or_train_model():
    """Load existing model or train new one"""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.success(f"✅ Model dimuat dari {MODEL_PATH}")
            return model
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat model: {e}. Melatih model baru...")
    
    # Train new model if not found
    return train_model()

def preprocess_data(df):
    """Preprocess student data for XGBoost"""
    # Target variable (simplified - you may need to adjust based on your data)
    df['Target'] = (df['G3'] < 10).astype(int)  # Dropout if final grade < 10
    
    # Select features (adjust based on your model requirements)
    feature_cols = [
        'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
        'freetime', 'goout', 'health', 'absences', 'G1', 'G2'
    ]
    
    X = df[feature_cols].fillna(0)
    y = df['Target']
    
    return X, y, feature_cols

def train_model():
    """Train new XGBoost model"""
    with st.spinner("Training XGBoost model..."):
        df = load_sample_data()
        if df is None:
            st.error("Tidak ada data untuk training!")
            return None
        
        X, y, feature_cols = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success(f"✅ Model berhasil dilatih! Akurasi: {accuracy:.2%}")
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        st.info(f"💾 Model disimpan ke {MODEL_PATH}")
        
        return model

def create_features(input_dict, feature_cols):
    """Create feature vector matching trained model"""
    data = {}
    for col in feature_cols:
        if col in input_dict:
            data[col] = [input_dict[col]]
        else:
            data[col] = [0]  # Default value
    
    return pd.DataFrame(data)

# UI Components (same as before but simplified)
def display_simple_sidebar():
    """Simplified input for demo"""
    st.sidebar.subheader("📊 Input Data Siswa")
    
    # Simplified inputs matching sample dataset
    inputs = {}
    
    st.sidebar.subheader("Keluarga")
    inputs['Medu'] = st.sidebar.slider("Pendidikan Ibu (0-4)", 0, 4, 2)
    inputs['Fedu'] = st.sidebar.slider("Pendidikan Ayah (0-4)", 0, 4, 2)
    
    st.sidebar.subheader("Belajar")
    inputs['traveltime'] = st.sidebar.slider("Waktu Perjalanan (1-4)", 1, 4, 1)
    inputs['studytime'] = st.sidebar.slider("Waktu Belajar (1-4)", 1, 4, 2)
    inputs['failures'] = st.sidebar.slider("Jumlah Kegagalan (0-3)", 0, 3, 0)
    
    st.sidebar.subheader("Lifestyle")
    inputs['famrel'] = st.sidebar.slider("Hubungan Keluarga (1-5)", 1, 5, 4)
    inputs['freetime'] = st.sidebar.slider("Waktu Luang (1-5)", 1, 5, 3)
    inputs['goout'] = st.sidebar.slider("Keluar Malam (1-5)", 1, 5, 3)
    inputs['health'] = st.sidebar.slider("Kesehatan (1-5)", 1, 5, 5)
    
    st.sidebar.subheader("Akademik")
    inputs['absences'] = st.sidebar.slider("Ketidakhadiran", 0, 75, 4)
    inputs['G1'] = st.sidebar.slider("Nilai Semester 1 (0-20)", 0, 20, 12)
    inputs['G2'] = st.sidebar.slider("Nilai Semester 2 (0-20)", 0, 20, 13)
    
    return inputs

def main():
    st.set_page_config(page_title="Prediksi Dropout 🎓", layout="wide")
    
    st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
    st.markdown("**Aplikasi XGBoost dengan auto-training jika model tidak ditemukan**")
    
    # Model management tab
    tab1, tab2 = st.tabs(["🔮 Prediksi", "⚙️ Model Management"])
    
    with tab1:
        # Load/Train model
        model = load_or_train_model()
        if model is None:
            st.error("❌ Gagal memuat atau melatih model!")
            st.stop()
        
        # Get prediction inputs
        input_data = display_simple_sidebar()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
                # Predict
                X_pred = create_features(input_data, model.feature_names_in_)
                probas = model.predict_proba(X_pred)[0]
                
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                # Results
                dropout_prob = probas[1]
                st.progress(dropout_prob)
                
                if dropout_prob > 0.5:
                    st.error(f"**⚠️ RISIKO TINGGI** - {dropout_prob:.1%}")
                else:
                    st.success(f"**✅ RISIKO RENDAH** - {dropout_prob:.1%}")
        
        with col2:
            st.info("**Fitur yang digunakan:**")
            for key, value in input_data.items():
                st.metric(key, value)
    
    with tab2:
        st.header("⚙️ Model Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Train Ulang Model", type="secondary"):
                model = train_model()
                st.rerun()
            
            if st.button("🗑️ Hapus Model"):
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                    st.success("Model dihapus!")
                    st.rerun()
        
        with col2:
            if os.path.exists(MODEL_PATH):
                st.success(f"✅ Model ada: {os.path.getsize(MODEL_PATH)/1024:.1f} KB")
            else:
                st.warning("❌ Model tidak ditemukan")
        
        # Show sample data
        if st.checkbox("📊 Lihat Sample Data"):
            df = load_sample_data()
            if df is not None:
                st.dataframe(df.head(), use_container_width=True)

if __name__ == "__main__":
    main()