import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
# Pastikan file model ada di direktori yang sama
try:
    model = joblib.load('xgboost_model.pkl')
except:
    st.error("Model 'xgboost_model.pkl' tidak ditemukan. Pastikan file tersedia.")

# Load Image (Opsional: tambahkan handling jika image tidak ada)
image = 'Juniyara Parisya Setiawan-Dashboard.jpg'

def display_sidebar():
    st.sidebar.subheader("Masukkan Data Siswa")

    # MAPPING: Sesuaikan angka di bawah ini dengan urutan saat Anda training model
    # Contoh: Single = 1, Married = 2, dst.
    
    marital_status = st.sidebar.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
    
    # ... (Dropdown lainnya tetap sama) ...
    
    # Contoh penggunaan Slider & Number Input
    previous_qualification_grade = st.sidebar.slider("Nilai Kualifikasi Sebelumnya", 0, 200, 100)
    admission_grade = st.sidebar.slider("Nilai Penerimaan", 0, 200, 100)
    age_at_enrollment = st.sidebar.number_input("Usia Saat Pendaftaran", value=18)
    
    # Input Akademik
    c1_credited = st.sidebar.number_input("Units 1st Sem (credited)", 0)
    c1_enrolled = st.sidebar.number_input("Units 1st Sem (enrolled)", 0)
    c1_eval = st.sidebar.number_input("Units 1st Sem (evaluations)", 0)
    c1_approved = st.sidebar.number_input("Units 1st Sem (approved)", 0)
    c1_grade = st.sidebar.number_input("Units 1st Sem (grade)", 0.0)
    
    # FITUR BINARY (MAPPING KE 0 DAN 1)
    # Sangat penting: XGBoost biasanya tidak menerima teks "Yes"/"No"
    displaced = 1 if st.sidebar.selectbox("Terlantar?", ["Yes", "No"]) == "Yes" else 0
    scholarship = 1 if st.sidebar.selectbox("Penerima Beasiswa?", ["Yes", "No"]) == "Yes" else 0
    tuition_up_to_date = 1 if st.sidebar.selectbox("Biaya Kuliah Lunas?", ["Yes", "No"]) == "Yes" else 0
    gender = 1 if st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"]) == "Male" else 0

    # Kumpulkan SEMUA kolom yang digunakan saat training (Urutan harus sama!)
    # Saya asumsikan ini adalah list kolom yang dibutuhkan XGBoost Anda
    data = {
        'Previous_qualification_grade': previous_qualification_grade,
        'Admission_grade': admission_grade,
        'Age_at_enrollment': age_at_enrollment,
        'Displaced': displaced,
        'Scholarship_holder': scholarship,
        'Tuition_fees_up_to_date': tuition_up_to_date,
        'Gender': gender,
        'Curricular_units_1st_sem_credited': c1_credited,
        'Curricular_units_1st_sem_enrolled': c1_enrolled,
        'Curricular_units_1st_sem_evaluations': c1_eval,
        'Curricular_units_1st_sem_approved': c1_approved,
        'Curricular_units_1st_sem_grade': c1_grade,
        # Tambahkan fitur lainnya di sini sesuai jumlah kolom saat training...
    }
    
    return pd.DataFrame([data])

def main():
    st.set_page_config(page_title="Prediksi Dropout Siswa", layout="wide")

    # Header Image
    try:
        st.image(image, use_container_width=True) # Versi terbaru pakai use_container_width
    except:
        st.title("Aplikasi Prediksi Dropout Siswa")

    input_data = display_sidebar()

    st.write("### Data yang Dimasukkan:")
    st.dataframe(input_data) # Menampilkan data agar user bisa cek

    if st.sidebar.button("Prediksi"):
        # Prediksi
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.markdown("---")
        st.subheader("Hasil Analisis:")

        # Asumsi: 1 adalah Dropout, 0 adalah Graduated/Enrolled
        prob_dropout = prediction_proba[0][1] 
        
        if prediction[0] == 1:
            st.error(f"⚠️ **HASIL: DROPOUT** (Probabilitas: {prob_dropout:.2%})")
            st.info("Saran: Siswa ini memerlukan pendampingan akademik lebih lanjut.")
        else:
            st.success(f"✅ **HASIL: TIDAK DROPOUT** (Probabilitas Aman: {1 - prob_dropout:.2%})")

if __name__ == "__main__":
    main()