import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & LOADING
# ---------------------------------------------------------
st.set_page_config(page_title="Prediksi Dropout Siswa", page_icon=":bar_chart:", layout="wide")

@st.cache_resource
def load_model():
    """Memuat model machine learning secara efisien."""
    model_path = 'xgboost_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def main():
    # Header Gambar
    image_path = 'Juniyara Parisya Setiawan-Dashboard.jpg'
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.title("🎓 Sistem Prediksi Dropout Siswa")

    model = load_model()
    if model is None:
        st.error("⚠️ File model 'xgboost_model.pkl' tidak ditemukan di direktori!")
        return

    # ---------------------------------------------------------
    # 2. SIDEBAR: INPUT DATA (MENGGUNAKAN FORM)
    # ---------------------------------------------------------
    with st.sidebar.form(key='student_data_form'):
        st.subheader("📝 Masukkan Data Siswa")
        
        # Data Demografi & Sosial
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Others"])
        displaced = st.selectbox("Terlantar/Displaced?", ["Yes", "No"])
        scholarship = st.selectbox("Penerima Beasiswa?", ["Yes", "No"])
        
        # Data Akademik & Pendaftaran
        st.markdown("---")
        course = st.selectbox("Program Studi", ["Informatics Engineering", "Management", "Social Service", "Tourism", "Others"])
        application_mode = st.selectbox("Mode Pendaftaran", ["1st phase", "2nd phase", "International", "Over 23 years old", "Change of course", "Others"])
        application_order = st.selectbox("Pilihan Kursus", ["First Choice", "Second Choice", "Others"])
        age_at_enrollment = st.number_input("Usia Saat Pendaftaran", min_value=15, max_value=100, value=20)
        
        # Nilai & Kualifikasi
        previous_qualification_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0, 200, 100)
        admission_grade = st.slider("Nilai Penerimaan", 0, 200, 100)
        
        # Unit Akademik (Semester 1 & 2)
        st.markdown("---")
        st.caption("Data Akademik Unit Kurikuler")
        cu1_credited = st.number_input("Units 1st Sem (Credited)", 0)
        cu1_eval = st.number_input("Units 1st Sem (Evaluations)", 0)
        cu1_grade = st.number_input("Units 1st Sem (Grade)", 0.0)
        cu1_no_eval = st.number_input("Units 1st Sem (Without Eval)", 0)
        cu2_no_eval = st.number_input("Units 2nd Sem (Without Eval)", 0)

        # Faktor Ekonomi Nasional
        st.markdown("---")
        unemployment = st.number_input("Unemployment Rate", 0.0)
        inflation = st.number_input("Inflation Rate", 0.0)
        gdp = st.number_input("GDP", 0.0)

        # Tombol Submit
        submit_button = st.form_submit_button(label='🚀 Lakukan Prediksi')

    # ---------------------------------------------------------
    # 3. LOGIKA PREDIKSI & MAPPING
    # ---------------------------------------------------------
    if submit_button:
        # Mapping Kategorikal ke Numerik (Sesuaikan dengan LabelEncoder saat training)
        # Contoh mapping standar (silakan disesuaikan dengan dataset Anda):
        gender_map = 1 if gender == "Male" else 0
        scholarship_map = 1 if scholarship == "Yes" else 0
        displaced_map = 1 if displaced == "Yes" else 0
        
        # Mapping sederhana untuk Course & Mode (Misal: Others = 0, sisanya berurutan)
        # CATATAN: Idealnya Anda menggunakan joblib.load('encoder.pkl') jika ada.
        course_dict = {"Informatics Engineering": 1, "Management": 2, "Social Service": 3, "Tourism": 4, "Others": 0}
        mode_dict = {"1st phase": 1, "2nd phase": 2, "International": 3, "Over 23 years old": 4, "Change of course": 5, "Others": 0}

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
            'Application_mode': mode_dict.get(application_mode, 0),
            'Course': course_dict.get(course, 0),
            'Displaced': displaced_map,
            'Gender': gender_map,
            'Scholarship_holder': scholarship_map
        }

        # Konversi ke DataFrame
        df_input = pd.DataFrame([input_dict])

        with st.spinner('Menganalisis data...'):
            try:
                # Proses Prediksi
                prediction = model.predict(df_input)
                prediction_proba = model.predict_proba(df_input)

                # 4. TAMPILAN HASIL
                st.divider()
                st.subheader("📊 Hasil Analisis Prediksi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Asumsi: 1 = Dropout, 0 = Graduate/Enrolled
                    if prediction[0] == 1:
                        st.error("### 🟥 Prediksi: **POTENSI DROPOUT**")
                        st.write("Siswa ini memiliki risiko tinggi untuk tidak menyelesaikan studinya.")
                    else:
                        st.success("### 🟩 Prediksi: **BERTAHAN (SUCCESS)**")
                        st.write("Siswa ini diprediksi akan melanjutkan atau menyelesaikan studinya.")
                
                with col2:
                    confidence = np.max(prediction_proba) * 100
                    st.metric("Tingkat Kepercayaan (Confidence)", f"{confidence:.2f}%")
                    
                    # Progres Bar Kepercayaan
                    st.progress(confidence / 100)

                # Opsional: Tampilkan data yang dikirim ke model (untuk debugging)
                with st.expander("Lihat Detail Data Input"):
                    st.dataframe(df_input)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan teknis: {e}")

if __name__ == "__main__":
    main()
