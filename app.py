import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor_model.pkl')
    model = joblib.load('xgboost_model.pkl')
    return preprocessor, model

preprocessor, model = load_models()

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="centered")

# Custom CSS untuk styling mirip gambar (Dark Theme & Red Accents)
st.markdown("""
    <style>
    .main { background-color: #121726; color: white; }
    .stButton>button { width: 100%; background-color: #e76f51; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Data Mahasiswa")

# --- FORM INPUT ---
with st.container():
    st.subheader("💰 Status Finansial")
    
    # Mapping sederhana untuk UI ke Nilai Numerik (sesuaikan dengan dataset asli)
    tuition = st.selectbox("Pembayaran SPP", ["Tepat Waktu", "Menunggak"], index=1)
    scholarship = st.selectbox("Status Beasiswa", ["Penerima", "Bukan Penerima"], index=1)
    debtor = st.selectbox("Status Debitur / Utang", ["Berhutang", "Tidak Berhutang"], index=1)

    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Semester 1")
        sks_1 = st.slider("SKS Lulus (Sem 1)", 0, 30, 5)
        grade_1 = st.number_input("Nilai Rata-rata (0-20) - S1", 0.0, 20.0, 12.00)

    with col2:
        st.subheader("📚 Semester 2")
        sks_2 = st.slider("SKS Lulus (Sem 2)", 0, 30, 4)
        grade_2 = st.number_input("Nilai Rata-rata (0-20) - S2", 0.0, 20.0, 12.00)

    st.divider()

    st.subheader("👤 Profil Mahasiswa")
    age = st.slider("Usia saat Enrollment", 14, 60, 19)
    admission_grade = st.number_input("Nilai Masuk (0-200)", 0.0, 200.0, 130.0)

# --- PREDICTION LOGIC ---
if st.button("🔍 Prediksi Sekarang"):
    # 1. Siapkan DataFrame mentah sesuai kolom yang diharapkan preprocessor
    # Catatan: Sesuaikan nama kolom di bawah dengan nama kolom di df asli Anda
    input_data = pd.DataFrame({
        'Tuition_fees_up_to_date': [1 if tuition == "Tepat Waktu" else 0],
        'Scholarship_holder': [1 if scholarship == "Penerima" else 0],
        'Debtor': [1 if debtor == "Berhutang" else 0],
        'Curricular_units_1st_sem_approved': [sks_1],
        'Curricular_units_1st_sem_grade': [grade_1],
        'Curricular_units_2nd_sem_approved': [sks_2],
        'Curricular_units_2nd_sem_grade': [grade_2],
        'Age_at_enrollment': [age],
        'Admission_grade': [admission_grade],
        # Tambahkan kolom lain yang dibutuhkan preprocessor dengan nilai default jika tidak ada di UI
        'Previous_qualification_grade': [120.0], 
        'Curricular_units_1st_sem_credited': [0],
        'Curricular_units_1st_sem_evaluations': [0],
        'Curricular_units_1st_sem_without_evaluations': [0],
        'Curricular_units_2nd_sem_without_evaluations': [0],
        'Unemployment_rate': [11.0],
        'Inflation_rate': [0.5],
        'GDP': [1.0],
        'Application_mode': [1],
        'Application_order': [1],
        'Course': [1],
        'Mothers_qualification': [1],
        'Fathers_qualification': [1],
        'Mothers_occupation': [1],
        'Fathers_occupation': [1],
        'Displaced': [1],
        'Gender': [1]
    })

    try:
        # 2. Transformasi data menggunakan preprocessor yang sudah di-load
        processed_data = preprocessor.transform(input_data)
        
        # 3. Prediksi
        prediction = model.predict(processed_data)
        
        # 4. Tampilkan Hasil
        st.markdown("---")
        if prediction[0] == 0:
            st.error("### Hasil Prediksi: **Dropout**")
            st.write("Mahasiswa berisiko tinggi tidak menyelesaikan pendidikan.")
        else:
            st.success("### Hasil Prediksi: **Graduate**")
            st.write("Mahasiswa diprediksi akan menyelesaikan pendidikan dengan baik.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")