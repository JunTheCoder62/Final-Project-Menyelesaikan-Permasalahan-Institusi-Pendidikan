import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Menggunakan nama file sesuai permintaan Anda
        model = joblib.load('xgboost_model.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# --- CUSTOM CSS (Untuk styling hasil prediksi) ---
st.markdown("""
    <style>
    .result-container {
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .high-risk {
        background-color: #fff5f5;
        color: #c53030;
    }
    .low-risk {
        background-color: #f0fff4;
        color: #2f855a;
    }
    .prob-text {
        font-size: 60px;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .status-text {
        font-size: 20px;
        font-weight: 600;
        margin-top: 10px;
    }
    .sub-text {
        font-size: 14px;
        color: #718096;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📄 Data Mahasiswa")

# --- INPUT FORM (Berdasarkan UI sebelumnya) ---
with st.expander("💰 Status Finansial", expanded=True):
    col_fin1 = st.selectbox("Pembayaran SPP", ["Menunggak", "Lunas"])
    col_fin2 = st.selectbox("Status Beasiswa", ["Bukan Penerima", "Penerima"])
    col_fin3 = st.selectbox("Status Debitur / Utang", ["Tidak Berutang", "Berutang"])

with st.expander("📚 Data Akademik", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Semester 1**")
        sks_1 = st.slider("SKS Lulus (S1)", 0, 24, 5)
        nilai_1 = st.number_input("Nilai Rata-rata (0-20) (S1)", 0.0, 20.0, 12.0)
    with col2:
        st.write("**Semester 2**")
        sks_2 = st.slider("SKS Lulus (S2)", 0, 24, 4)
        nilai_2 = st.number_input("Nilai Rata-rata (0-20) (S2)", 0.0, 20.0, 12.0)

with st.expander("👤 Profil Mahasiswa", expanded=True):
    usia = st.slider("Usia saat Enrollment", 15, 60, 19)
    nilai_masuk = st.number_input("Nilai Masuk (0-200)", 0.0, 200.0, 130.0)

# --- LOGIKA PREDIKSI ---
if st.button("🔍 Prediksi Sekarang", use_container_width=True):
    if model is not None:
        # 1. Menyiapkan Data Input
        # Sesuaikan urutan kolom ini dengan fitur saat training model
        input_data = pd.DataFrame([{
            'pembayaran_spp': 1 if col_fin1 == "Menunggak" else 0,
            'beasiswa': 1 if col_fin2 == "Penerima" else 0,
            'debitur': 1 if col_fin3 == "Berutang" else 0,
            'sks_S1': sks_1,
            'nilai_S1': nilai_1,
            'sks_S2': sks_2,
            'nilai_S2': nilai_2,
            'usia': usia,
            'nilai_masuk': nilai_masuk
        }])

        # 2. Prediksi Probabilitas
        # predict_proba menghasilkan [prob_class_0, prob_class_1]
        # Kita ambil indeks [1] untuk probabilitas "Dropout"
        probabilities = model.predict_proba(input_data)[0]
        dropout_prob = probabilities[1] * 100 

        # 3. Tampilkan Hasil ala Gambar
        status_class = "high-risk" if dropout_prob >= 50 else "low-risk"
        status_label = "Risiko Tinggi — Butuh Intervensi Segera" if dropout_prob >= 70 else "Risiko Rendah / Aman"
        emoji = "🚨" if dropout_prob >= 70 else "✅"

        st.markdown(f"""
            <div class="result-container {status_class}">
                <p class="prob-text">{dropout_prob:.1f}%</p>
                <p class="status-text">{emoji} {status_label}</p>
                <p class="sub-text">Probabilitas mahasiswa ini mengalami dropout</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model tidak ditemukan. Pastikan file 'xgboost_model.pkl' ada di folder yang sama.")