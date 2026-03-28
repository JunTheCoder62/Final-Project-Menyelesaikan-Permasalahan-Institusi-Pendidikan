import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. KONFIGURASI & STYLE ---
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .result-container {
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        font-family: 'sans-serif';
    }
    .high-risk { background-color: #fdf2f2; border: 1px solid #f8d7da; color: #721c24; }
    .low-risk { background-color: #f0fff4; border: 1px solid #c6f6d5; color: #22543d; }
    .prob-text { font-size: 64px; font-weight: 800; margin: 0; }
    .status-text { font-size: 22px; font-weight: 700; margin: 10px 0; }
    .sub-text { font-size: 14px; color: #666; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    try:
        # Memuat file model Anda
        model = joblib.load('xgboost_model.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_trained_model()

# --- 3. UI INPUT ---
st.title("📄 Data Mahasiswa")

with st.container():
    st.subheader("💰 Status Finansial")
    col_spp = st.selectbox("Pembayaran SPP", ["Menunggak", "Lunas"])
    col_beasiswa = st.selectbox("Status Beasiswa", ["Bukan Penerima", "Penerima"])
    col_utang = st.selectbox("Status Debitur / Utang", ["Tidak Berutang", "Berutang"])

st.markdown("---")

with st.container():
    st.subheader("📚 Semester 1")
    sks1 = st.slider("SKS Lulus", 0, 30, 5, key="s1_sks")
    nilai1 = st.number_input("Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, key="s1_val")

st.markdown("---")

with st.container():
    st.subheader("📚 Semester 2")
    sks2 = st.slider("SKS Lulus", 0, 30, 4, key="s2_sks")
    nilai2 = st.number_input("Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, key="s2_val")

st.markdown("---")

with st.container():
    st.subheader("👤 Profil Mahasiswa")
    usia = st.slider("Usia saat Enrollment", 15, 60, 19)
    nilai_masuk = st.number_input("Nilai Masuk (0-200)", 0.0, 200.0, 130.0)

# --- 4. LOGIKA PREDIKSI & ALIGNMENT ---
if st.button("🔍 Prediksi Sekarang", use_container_width=True):
    if model is not None:
        try:
            # A. Buat DataFrame awal dari input UI
            # Pastikan nama kolom di sini (kiri) sesuai dengan nama kolom SEBELUM encoding saat training
            input_df = pd.DataFrame([{
                'Tuition fees up to date': 1 if col_spp == "Lunas" else 0,
                'Scholarship holder': 1 if col_beasiswa == "Penerima" else 0,
                'Debtor': 1 if col_utang == "Berutang" else 1,
                'Curricular units 1st sem (approved)': sks1,
                'Curricular units 1st sem (grade)': nilai1,
                'Curricular units 2nd sem (approved)': sks2,
                'Curricular units 2nd sem (grade)': nilai2,
                'Age at enrollment': usia,
                'Admission grade': nilai_masuk
            }])

            # B. One-Hot Encoding pada input baru
            input_encoded = pd.get_dummies(input_df)

            # C. SINKRONISASI FITUR (Mengatasi Error 190 vs 9)
            # Mengambil daftar fitur yang diharapkan model
            if hasattr(model, "feature_names_in_"):
                model_features = model.feature_names_in_
                # Reindex akan menambah kolom yang hilang (isi 0) dan membuang kolom yang tidak perlu
                final_input = input_encoded.reindex(columns=model_features, fill_value=0)
            else:
                # Jika model bukan pipeline/dataframe-based, Anda mungkin perlu cara manual
                st.warning("Atribut feature_names_in_ tidak ditemukan. Menggunakan input apa adanya.")
                final_input = input_encoded

            # D. Prediksi Probabilitas
            # [0, 1] -> Indeks 1 biasanya adalah kelas 'Dropout'
            prob = model.predict_proba(final_input)[0][1] * 100

            # E. Tampilan Hasil (Sesuai Gambar)
            risk_class = "high-risk" if prob >= 50 else "low-risk"
            risk_msg = "Risiko Tinggi — Butuh Intervensi Segera" if prob >= 50 else "Risiko Rendah — Mahasiswa Aman"
            icon = "🚨" if prob >= 50 else "✅"

            st.markdown(f"""
                <div class="result-container {risk_class}">
                    <p class="prob-text">{prob:.1f}%</p>
                    <p class="status-text">{icon} {risk_msg}</p>
                    <p class="sub-text">Probabilitas mahasiswa ini mengalami dropout</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan data: {e}")
    else:
        st.error("Model tidak tersedia.")