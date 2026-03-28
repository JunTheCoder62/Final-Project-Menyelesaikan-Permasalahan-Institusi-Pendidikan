import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Status Prediction",
    layout="wide"
)

st.title("🎓 Prediksi Status Mahasiswa Jaya Jaya Institut")
st.write("Aplikasi ini memprediksi apakah seorang mahasiswa berpotensi *Dropout* atau *Graduate* berdasarkan data akademik dan demografi menggunakan model XGBoost.")

# 2. Fungsi untuk memuat model (menggunakan caching agar lebih cepat)
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('preprocessor_model.pkl')
        model = joblib.load('xgboost_model.pkl')
        return preprocessor, model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file .pkl berada di direktori yang sama. Error: {e}")
        return None, None

preprocessor, model = load_models()

# 3. Membuat Form Input Pengguna
st.sidebar.header("Masukkan Data Mahasiswa")

# Menggunakan form agar prediksi tidak berjalan otomatis setiap kali angka diubah
with st.sidebar.form("input_form"):
    st.subheader("Data Demografi & Background (Kategorik)")
    # Berdasarkan dataset aslinya, data kategorik ini berwujud angka (encoded integer)
    application_mode = st.number_input('Application mode', value=1, step=1)
    application_order = st.number_input('Application order', value=1, step=1)
    course = st.number_input('Course', value=33, step=1)
    mothers_qualification = st.number_input("Mother's qualification", value=1, step=1)
    fathers_qualification = st.number_input("Father's qualification", value=1, step=1)
    mothers_occupation = st.number_input("Mother's occupation", value=1, step=1)
    fathers_occupation = st.number_input("Father's occupation", value=1, step=1)
    displaced = st.selectbox('Displaced', [0, 1])
    gender = st.selectbox('Gender (0: Female, 1: Male)', [0, 1])

    st.subheader("Data Akademik & Ekonomi (Numerik)")
    prev_grade = st.number_input('Previous qualification grade', value=120.0, step=1.0)
    admission_grade = st.number_input('Admission grade', value=120.0, step=1.0)
    age_enrollment = st.number_input('Age at enrollment', value=20, step=1)
    
    sem1_credited = st.number_input('Curricular units 1st sem (credited)', value=0, step=1)
    sem1_evaluations = st.number_input('Curricular units 1st sem (evaluations)', value=0, step=1)
    sem1_grade = st.number_input('Curricular units 1st sem (grade)', value=12.0, step=0.1)
    sem1_without_evals = st.number_input('Curricular units 1st sem (without evaluations)', value=0, step=1)
    
    sem2_without_evals = st.number_input('Curricular units 2nd sem (without evaluations)', value=0, step=1)
    
    unemployment_rate = st.number_input('Unemployment rate', value=10.0, step=0.1)
    inflation_rate = st.number_input('Inflation rate', value=1.0, step=0.1)
    gdp = st.number_input('GDP', value=1.0, step=0.1)

    submitted = st.form_submit_button("Prediksi Status")

# 4. Proses Prediksi
if submitted:
    if preprocessor is not None and model is not None:
        # Menyatukan input ke dalam format DataFrame yang sama dengan X_train
        input_data = pd.DataFrame({
            'Previous_qualification_grade': [prev_grade],
            'Admission_grade': [admission_grade],
            'Age_at_enrollment': [age_enrollment],
            'Curricular_units_1st_sem_credited': [sem1_credited],
            'Curricular_units_1st_sem_evaluations': [sem1_evaluations],
            'Curricular_units_1st_sem_grade': [sem1_grade],
            'Curricular_units_1st_sem_without_evaluations': [sem1_without_evals],
            'Curricular_units_2nd_sem_without_evaluations': [sem2_without_evals],
            'Unemployment_rate': [unemployment_rate],
            'Inflation_rate': [inflation_rate],
            'GDP': [gdp],
            'Application_mode': [application_mode],
            'Application_order': [application_order],
            'Course': [course],
            'Mothers_qualification': [mothers_qualification],
            'Fathers_qualification': [fathers_qualification],
            'Mothers_occupation': [mothers_occupation],
            'Fathers_occupation': [fathers_occupation],
            'Displaced': [displaced],
            'Gender': [gender]
        })

        # Preprocessing Data Baru
        try:
            # preprocessor.transform() membutuhkan urutan dan nama kolom yang sesuai
            input_processed = preprocessor.transform(input_data)
            
            # Prediksi dengan XGBoost
            prediction = model.predict(input_processed)
            
            # Label Encoder biasanya mengurutkan abjad: 0 = Dropout, 1 = Graduate (tergantung data asli)
            # Pada notebook kamu, status 2 (Enrolled) sudah direplace dan diubah ke tipe integer
            result = prediction[0]
            
            st.write("### Hasil Prediksi:")
            if result == 0:
                st.error("⚠️ Mahasiswa ini berpotensi **Dropout**.")
            elif result == 1:
                st.success("✅ Mahasiswa ini diprediksi akan **Graduate**.")
            else:
                st.info(f"Kategori Status (Encoded): {result}")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
