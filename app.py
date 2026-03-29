import streamlit as st
import pandas as pd
import joblib
import os

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Mengambil path absolut dari folder script berada
    base_path = os.path.dirname(__file__)
    preprocessor_path = os.path.join(base_path, 'preprocessor_model.pkl')
    model_path = os.path.join(base_path, 'xgboost_model.pkl')
    
    # Cek apakah file ada sebelum di-load
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan! Pastikan '{preprocessor_path}' dan '{model_path}' ada di folder yang sama.")
        st.stop()
        
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="centered")

# Load models
preprocessor, model = load_models()

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #121726; color: white; }
    .stButton>button { width: 100%; background-color: #e76f51; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Data Mahasiswa")

# --- FORM INPUT ---
with st.form("prediction_form"):
    st.subheader("💰 Status Finansial")
    
    tuition = st.selectbox("Pembayaran SPP", ["Tepat Waktu", "Menunggak"])
    scholarship = st.selectbox("Status Beasiswa", ["Penerima", "Bukan Penerima"])
    debtor = st.selectbox("Status Debitur / Utang", ["Berhutang", "Tidak Berhutang"])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 Semester 1")
        sks_1 = st.slider("SKS Lulus (Sem 1)", 0, 30, 5)
        grade_1 = st.number_input("Nilai Rata-rata (0-20) - S1", 0.0, 20.0, 12.0)

    with col2:
        st.subheader("📚 Semester 2")
        sks_2 = st.slider("SKS Lulus (Sem 2)", 0, 30, 4)
        grade_2 = st.number_input("Nilai Rata-rata (0-20) - S2", 0.0, 20.0, 12.0)

    st.divider()

    st.subheader("👤 Profil Mahasiswa")
    age = st.slider("Usia saat Enrollment", 14, 60, 19)
    admission_grade = st.number_input("Nilai Masuk (0-200)", 0.0, 200.0, 130.0)

    submit_button = st.form_submit_button("🔍 Prediksi Sekarang")

# --- PREDICTION LOGIC ---
if submit_button:
    input_dict = {
        'Tuition_fees_up_to_date': [1 if tuition == "Tepat Waktu" else 0],
        'Scholarship_holder': [1 if scholarship == "Penerima" else 0],
        'Debtor': [1 if debtor == "Berhutang" else 0],
        'Curricular_units_1st_sem_approved': [sks_1],
        'Curricular_units_1st_sem_grade': [grade_1],
        'Curricular_units_2nd_sem_approved': [sks_2],
        'Curricular_units_2nd_sem_grade': [grade_2],
        'Age_at_enrollment': [age],
        'Admission_grade': [admission_grade],
        
        # Kolom pendukung (sesuaikan dengan training data)
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
    }
    
    input_data = pd.DataFrame(input_dict)

    try:
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)
        
        st.markdown("---")
        if prediction[0] == 0:
            st.error("### Hasil Prediksi: **Dropout**")
            st.write("Mahasiswa berisiko tinggi tidak menyelesaikan pendidikan.")
        else:
            st.success("### Hasil Prediksi: **Graduate**")
            st.write("Mahasiswa diprediksi akan menyelesaikan pendidikan dengan baik.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")