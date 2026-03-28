import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb  # Import XGBoost untuk handle version mismatch

# 1. OPTIMASI LOADING MODEL - Handle semua format dan version mismatch
@st.cache_resource
def load_model():
    """Load model dengan fallback untuk semua kemungkinan error"""
    model_paths = ['xgboost_model.json', 'xgboost_model.pkl', 'xgboost_model.bin']
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                st.info(f"🔄 Loading {model_path}...")
                
                if model_path.endswith('.json') or model_path.endswith('.bin'):
                    # Native XGBoost format (paling aman)
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)
                    st.success(f"✅ Model loaded dari {model_path}")
                    return model
                
                elif model_path.endswith('.pkl'):
                    # Joblib/Pickle dengan error handling
                    model = joblib.load(model_path)
                    st.success(f"✅ Model loaded dari {model_path}")
                    return model
                    
            except Exception as e:
                st.warning(f"⚠️ Gagal load {model_path}: {str(e)[:100]}")
                continue
    
    st.error("❌ Tidak ada model yang valid ditemukan!")
    st.info("Pastikan ada file: xgboost_model.json, .pkl, atau .bin")
    return None

def main():
    # Set page config (Harus paling atas)
    st.set_page_config(
        page_title="Prediksi Dropout Siswa",
        page_icon=":bar_chart:",
        layout="wide"
    )

    # Header Gambar
    image_path = 'Juniyara Parisya Setiawan-Dashboard.jpg'
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    # Load model dengan robust error handling
    model = load_model()
    if model is None:
        st.stop()  # Stop execution jika model gagal load

    # SIDEBAR FORM - Input data siswa
    with st.sidebar.form(key='student_data_form'):
        st.subheader("📝 Masukkan Data Siswa")

        # Kolom kategorikal
        col1, col2 = st.columns(2)
        with col1:
            marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        with col2:
            application_order = st.selectbox("Pilihan Kursus", ["First Choice", "Second Choice"])
            displaced = st.selectbox("Terlantar?", ["No", "Yes"])
            scholarship = st.selectbox("Beasiswa?", ["No", "Yes"])

        # Nilai akademik
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            previous_qualification_grade = st.slider("Nilai Kualifikasi", 0, 200, 100)
            admission_grade = st.slider("Nilai Penerimaan", 0, 200, 100)
            age_at_enrollment = st.number_input("Usia", 15, 100, 20)
        with col2:
            course = st.selectbox("Program Studi", [
                "Informatics Engineering", "Management", "Social Service", 
                "Tourism", "Others"
            ])
            application_mode = st.selectbox("Mode Pendaftaran", [
                "1st phase - general contingent", "2nd phase - general contingent",
                "International student", "Over 23 years old", "Others"
            ])

        # Data akademik semester 1 & 2
        st.markdown("---")
        st.caption("📚 Data Akademik Semester 1 & 2")
        cu1_credited = st.number_input("Units 1st Sem (Credited)", 0, 30, 0)
        cu1_eval = st.number_input("Units 1st Sem (Evaluations)", 0, 30, 0)
        cu1_grade = st.number_input("Units 1st Sem (Grade)", 0.0, 20.0, 10.0)
        cu1_no_eval = st.number_input("Units 1st Sem (No Eval)", 0, 10, 0)
        cu2_no_eval = st.number_input("Units 2nd Sem (No Eval)", 0, 10, 0)

        # Data ekonomi
        st.markdown("---")
        st.caption("💰 Data Ekonomi")
        col1, col2, col3 = st.columns(3)
        with col1: unemployment = st.number_input("Unemployment Rate", 0.0, 20.0, 10.0)
        with col2: inflation = st.number_input("Inflation Rate", -5.0, 20.0, 2.0)
        with col3: gdp = st.number_input("GDP", -10.0, 10.0, 1.0)

        # Tombol submit
        submit_button = st.form_submit_button('🚀 Prediksi Risiko Dropout', use_container_width=True)

    # LOGIKA PREDIKSI
    if submit_button:
        with st.spinner('🔮 Menganalisis risiko dropout...'):
            try:
                # Mapping kategorikal
                mapping = {
                    'Gender': {'Male': 0, 'Female': 1},
                    'Displaced': {'No': 0, 'Yes': 1},
                    'Scholarship': {'No': 0, 'Yes': 1},
                    'Application_order': {'First Choice': 0, 'Second Choice': 1}
                }

                # Buat input dictionary
                input_data = {
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
                    'Displaced': mapping['Displaced'][displaced],
                    'Scholarship': mapping['Scholarship'][scholarship],
                    'Gender': mapping['Gender'][gender],
                    'Application_order': mapping['Application_order'][application_order]
                }

                # Convert ke DataFrame
                df_input = pd.DataFrame([input_data])

                # Match kolom dengan model (handle missing columns)
                if hasattr(model, 'feature_names_in_'):
                    expected_cols = model.feature_names_in_
                    missing_cols = set(expected_cols) - set(df_input.columns)
                    for col in missing_cols:
                        df_input[col] = 0
                    
                    # Reorder columns
                    df_input = df_input[expected_cols]

                # Prediksi
                prediction = model.predict(df_input)[0]
                probabilities = model.predict_proba(df_input)[0]

                # Tampilkan hasil
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **RISIKO DROPOUT TINGGI**")
                        st.markdown("**Probabilitas:**")
                        st.metric("Dropout", f"{probabilities[1]:.1%}", delta=None)
                    else:
                        st.success("✅ **AMAN**")
                        st.markdown("**Probabilitas:**")
                        st.metric("Tidak Dropout", f"{probabilities[0]:.1%}", delta=None)
                
                with col2:
                    st.metric("Probabilitas Dropout", f"{probabilities[1]:.1%}")
                    st.metric("Probabilitas Bertahan", f"{probabilities[0]:.1%}")

                # Rekomendasi
                st.markdown("### 💡 Rekomendasi")
                if prediction == 1 and probabilities[1] > 0.7:
                    st.warning("**Prioritas Tinggi:** Segera lakukan intervensi akademik!")
                    st.info("- Konseling akademik\n- Bimbingan karir\n- Dukungan finansial")
                elif prediction == 1:
                    st.info("**Pantau Ketat:** Perhatikan perkembangan akademik")
                else:
                    st.success("**Status Baik:** Tetap monitor perkembangan")

            except Exception as e:
                st.error(f"❌ Error prediksi: {str(e)}")
                st.info("Cek format data input atau hubungi developer")

if __name__ == "__main__":
    main()
