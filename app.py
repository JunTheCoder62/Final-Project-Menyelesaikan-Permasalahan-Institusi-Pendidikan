import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Optimasi Loading Model: Menggunakan cache_resource agar hanya dimuat 1x selama app berjalan
@st.cache_resource
def load_model():
    # Pastikan file model ada
    if os.path.exists('xgboost_model.pkl'):
        return joblib.load('xgboost_model.pkl')
    return None

def main():
    # Set page config (Harus paling atas)
    st.set_page_config(page_title="Prediksi Dropout Siswa", page_icon=":bar_chart:", layout="wide")

    # Header Gambar (Gunakan use_container_width yang lebih baru dari use_column_width)
    image_path = 'Juniyara Parisya Setiawan-Dashboard.jpg'
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    model = load_model()
    if model is None:
        st.error("Model 'xgboost_model.pkl' tidak ditemukan!")
        return

    # 2. GUNAKAN FORM: Ini kunci agar aplikasi tidak lambat/rerun setiap input diubah
    with st.sidebar.form(key='student_data_form'):
        st.subheader("📝 Masukkan Data Siswa")
        
        # Kolom Sidebar (Dikelompokkan agar lebih rapi)
        marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
        application_mode = st.selectbox("Mode Pendaftaran", ["1st phase - general contingent", "2nd phase - general contingent", "International student (bachelor)", "Over 23 years old", "Change of course", "Others"])
        application_order = st.selectbox("Pilihan Kursus", ["First Choice", "Second Choice"])
        course = st.selectbox("Program Studi", ["Informatics Engineering", "Management", "Social Service", "Tourism", "Others"])
        
        # Slider & Number Input
        previous_qualification_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0, 200, 100)
        admission_grade = st.slider("Nilai Penerimaan", 0, 200, 100)
        age_at_enrollment = st.number_input("Usia Saat Pendaftaran", min_value=15, max_value=100, value=20)
        
        # Input Akademik Semesters
        st.markdown("---")
        st.caption("Data Akademik Semester 1 & 2")
        cu1_credited = st.number_input("Units 1st Sem (Credited)", 0)
        cu1_eval = st.number_input("Units 1st Sem (Evaluations)", 0)
        cu1_grade = st.number_input("Units 1st Sem (Grade)", 0.0)
        cu1_no_eval = st.number_input("Units 1st Sem (Without Eval)", 0)
        cu2_no_eval = st.number_input("Units 2nd Sem (Without Eval)", 0)

        # Input Ekonomi
        st.markdown("---")
        unemployment = st.number_input("Unemployment Rate", 0.0)
        inflation = st.number_input("Inflation Rate", 0.0)
        gdp = st.number_input("GDP", 0.0)

        # Dropdown kategori lainnya
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        displaced = st.selectbox("Terlantar?", ["Yes", "No"])
        scholarship = st.selectbox("Penerima Beasiswa?", ["Yes", "No"])

        # Tombol Submit Form
        submit_button = st.form_submit_button(label='🚀 Lakukan Prediksi')

    # 3. Logika Prediksi: Hanya berjalan jika tombol ditekan
    if submit_button:
        # Menyiapkan data untuk prediksi
        # PENTING: Jika model XGBoost Anda hasil trainingnya menggunakan angka (0,1), 
        # pastikan Anda melakukan mapping/encoding di sini sebelum dimasukkan ke DataFrame.
        
        input_dict = {
            'Previous_qualification_grade': previous_qualification_grade,
            'Admission_grade': admission_grade,
            'Age_at_enrollment': age_at_enrollment,
            'Curricular_units_1st_sem_credited': cu1_credited,
            'Curricular_units_1st_sem_evaluations': cu1_eval,
            'Curricular_units_1st_sem_grade': cu1_grade,
            'Curricular_units_1st_sem_without_evaluations': cu1_no_eval,
            'Curricular_units_2nd_sem_without_evaluations': cu2_no_eval,
            'Unemployment_rate': unemployment,
            'Inflation_rate': inflation,
            'GDP': gdp,
            # Perhatian: Tambahkan mapping untuk kolom kategorikal di bawah ini jika model Anda bukan native categorical
            'Application_mode': application_mode,
            'Application_order': application_order,
            'Course': course,
            'Displaced': displaced,
            'Gender': gender
        }

        df_input = pd.DataFrame([input_dict])

        with st.spinner('Menganalisis data...'):
