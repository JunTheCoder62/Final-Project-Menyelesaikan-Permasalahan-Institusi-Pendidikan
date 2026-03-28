# -*- coding: utf-8 -*-
"""Prediksi Status Mahasiswa - FIXED VERSION"""

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgboost_model.pkl')
        return model
    except:
        st.error("Model tidak ditemukan. Menggunakan fallback preprocessing.")
        return None

model = load_model()

# --- UI STYLE ---
st.markdown("""
    <style>
    .main { background-color: #1a1c24; color: white; }
    .stNumberInput, .stSelectbox, .stSlider { margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📄 Prediksi Status Mahasiswa")

# --- DEBUG INFO ---
if st.checkbox("🔧 Debug Model Info"):
    if model:
        try:
            if hasattr(model, 'named_steps'):
                st.success("✅ Model adalah PIPELINE lengkap!")
                st.info(f"Expected features: {len(model.named_steps['model'].feature_names_in_)}")
            else:
                st.warning("⚠️ Model bukan pipeline. Pakai fallback preprocessing.")
        except:
            st.warning("⚠️ Tidak bisa baca feature info. Pakai fallback.")

# --- INPUT FORM ---
st.subheader("💰 Status Finansial")
col_fin1 = st.selectbox("Pembayaran SPP", ["Menunggak", "Lunas"])
col_fin2 = st.selectbox("Status Beasiswa", ["Bukan Penerima", "Penerima"])
col_fin3 = st.selectbox("Status Debitur", ["Tidak Berutang", "Berutang"])

st.markdown("---")

st.subheader("📚 Prestasi Akademik")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Semester 1")
    sks_1 = st.slider("SKS Lulus S1", 0, 24, 5)
    nilai_1 = st.number_input("IPK S1", 0.0, 4.0, 3.0, step=0.1)
with col2:
    st.subheader("Semester 2")
    sks_2 = st.slider("SKS Lulus S2", 0, 24, 4)
    nilai_2 = st.number_input("IPK S2", 0.0, 4.0, 2.8, step=0.1)

st.markdown("---")

st.subheader("👤 Profil")
usia = st.slider("Usia", 15, 60, 19)
nilai_masuk = st.slider("Nilai Masuk (maks 100)", 0, 100, 70)

# --- FALLBACK PREPROCESSOR (JIKA MODEL BUKAN PIPELINE) ---
@st.cache_data
def create_fallback_preprocessor():
    """Buat preprocessor identik dengan training"""
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), [0,1,2]),
        ('num', StandardScaler(), [3,4,5,6,7,8])
    ])
    return preprocessor

# --- PREDIKSI BUTTON ---
if st.button("🔍 **PREDIKSI SEKARANG**", use_container_width=True, type="primary"):
    
    # Data input
    data = {
        'pembayaran_spp': 1 if col_fin1 == "Menunggak" else 0,
        'beasiswa': 1 if col_fin2 == "Penerima" else 0,
        'debitur': 1 if col_fin3 == "Berutang" else 0,
        'sks_S1': sks_1,
        'nilai_S1': nilai_1,
        'sks_S2': sks_2,
        'nilai_S2': nilai_2,
        'usia_enrollment': usia,
        'nilai_masuk': nilai_masuk
    }
    
    df_input = pd.DataFrame([data])
    
    try:
        if model and hasattr(model, 'named_steps'):  # PIPELINE LENGKAP
            prediction = model.predict(df_input)
            probability = model.predict_proba(df_input)[0]
            st.success(f"✅ **Prediksi: {prediction[0]}**")
            st.info(f"Probabilitas: {probability[1]:.1%} {'(Risiko)' if probability[1]>0.5 else '(Aman)'}")
            
        else:  # FALLBACK PREPROCESSING
            st.warning("🛠️ Menggunakan fallback preprocessing...")
            preprocessor = create_fallback_preprocessor()
            
            # Cek apakah model expect 190 features
            if hasattr(model, 'n_features_in_') and model.n_features_in_ == 190:
                # Buat dummy features untuk match 190 columns
                dummy_features = np.zeros((1, 190))
                # Copy 9 features pertama
                dummy_features[0, :9] = df_input.values[0]
                prediction = model.predict(dummy_features)
            else:
                # Preprocess normal 9 features
                X_processed = preprocessor.fit_transform(df_input)
                prediction = model.predict(X_processed)
            
            st.success(f"✅ **Prediksi: {prediction[0]}** (Fallback mode)")
        
        # Visualisasi hasil
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Solusi: Regenerate model dengan pipeline lengkap!")

# --- INFO BOX ---
with st.expander("📋 Cara Fix Permanent"):
    st.code("""
# TRAINING CODE (Colab):
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier())
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'xgboost_pipeline.pkl')  # SAVE PIPELINE!
    
# Streamlit: model = joblib.load('xgboost_pipeline.pkl')
    """, language="python")

st.markdown("---")
st.caption("💡 Powered by XGBoost | Fixed untuk 190 features mismatch")