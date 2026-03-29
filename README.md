# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout. Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis
1.  Masalah ekonomi keluarga mahasiswa yang membuat dukungan untuk menempuh pendidikan.
2.  Adanya ketidakpastian dalam prestasi akademik (Nilai) yang memnyebabkan tingginya Dropout pada mahasiswa.
3.  Latarbelakang Pendidikan mahaiswa sebelum menempuh pendidikan.
5.  Kurangnya pemantauan pada mahasiswa sehingga resiko Dropout sangat tinggi.

### Cakupan Proyek
* Melakukan EDA (Exploratory Data Analysis) terhadap data untuk mengidentifikasi faktor - faktor yang mempengaruhi Dropout
* Visualisasi dan pemodelan data untuk mengidentifikasi pola - pola data sebagai features model
* Pembuatan Dashboard dengan LookerStudio untuk visualisasi data
* Pembuatan model machine learning ``` xgboost_model.pkl ```
* Deployment model machine learning dengan menggunakan streamlit sebagai prediksi Dropout.

### Persiapan

Sumber data: [Link](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Setup environment:
- Download repositories ini
- lakukan :
```
!pip install -r requirements.txt
```

## Business Dashboard
Link : [Dashboard](https://lookerstudio.google.com/reporting/b4ee73bb-408d-4d93-8b9c-820bc122413f)

<img width="1652" height="1236" alt="Screenshot 2026-03-27 210206" src="https://github.com/user-attachments/assets/b4174e6d-402e-4594-ae74-a2abb918f2de" />

Kondisi akademik mahasiswa di Jaya Jaya Institut dari total populasi sebanyak 4.424 mahasiswa. Secara keseluruhan, hampir separuh dari populasi berhasil menyelesaikan studi dengan status Graduate (Lulus) yang mencapai 49,9%, sedangkan mahasiswa yang masih aktif belajar atau Enrolled (Terdaftar) berada pada angka 17,9%. Namun, hal yang patut menjadi perhatian serius bagi institusi adalah tingginya persentase mahasiswa yang berstatus Dropout, yaitu sebesar 32,1% (hampir sepertiga dari total keseluruhan). Jika dianalisis lebih detail melalui grafik dan tabel berdasarkan usia saat pendaftaran, kasus dropout ini paling banyak dialami oleh mahasiswa berusia muda, dengan puncak tertinggi terjadi pada mahasiswa usia 19 tahun (207 kasus) dan 18 tahun (202 kasus). 

Pada bagian bawah dashboard juga mengindikasikan adanya korelasi antara performa akademik dengan tingkat retensi; di mana mahasiswa yang dropout cenderung memiliki rekam jejak penyelesaian unit kurikuler (mata kuliah yang disetujui/lulus) yang jauh lebih rendah dibandingkan dengan yang sudah lulus

## Menjalankan Sistem Machine Learning
Akses Streamlit: [Link]()
Untuk menjalankan Predict model ```xgboost_model.pkl``` dapat dilakukan dengan:
1. Download Repositories
   ```
   https://github.com/JunTheCoder62/Final-Project-Menyelesaikan-Permasalahan-Institusi-Pendidikan
   ```
2. Setup library yang dibutuhkan untuk menjalankan model seperti yang dilakukan pada ```Setup enviroment``` diatas atau dapat dengan melakukan
   ```
   !pip install -r requirements.txt
   ```
3. streamlit di lokal dapat menggunakan kode berikut di terminal:
   ```
   streamlit run app.py
   ```

## Conclusion
Berdasarkan analisis data Status Dropout, didapat kesimpulan:
1. **Tingkat Dropout yang Signifikan:** Sebanyak 32,1% mahasiswa di Jaya Jaya Institut berstatus dropout. Angka ini merupakan sinyal merah bagi stabilitas institusi, di mana kelompok usia muda (18-19 tahun) menjadi segmen yang paling rentan.
2. **Performa Akademik sebagai Prediktor Utama:** Terdapat korelasi kuat antara rendahnya penyelesaian unit kurikuler di semester awal dengan keputusan mahasiswa untuk dropout. Mahasiswa yang gagal mengamankan kredit di semester 1 dan 2 memiliki probabilitas jauh lebih tinggi untuk tidak melanjutkan studi.
3. **Faktor Eksternal:** Selain akademik, variabel seperti status pemegang beasiswa dan kondisi ekonomi (yang tercermin dalam tingkat pengangguran di dashboard) turut memengaruhi kemampuan mahasiswa untuk bertahan hingga lulus.
4. **Efektivitas Prediksi:** Dengan digunakannya model ```xgboost_model.pkl```, institusi kini memiliki kemampuan proaktif untuk mengidentifikasi mahasiswa berisiko tinggi sebelum mereka benar-benar keluar, sehingga intervensi dapat dilakukan lebih dini.

### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
- **Sistem Peringatan Dini (Early Warning System):**
Mengintegrasikan model machine learning XGBoost ke dalam sistem informasi akademik untuk menandai mahasiswa yang memiliki skor risiko dropout tinggi berdasarkan performa semester pertama.
- **Program Mentoring Khusus Usia Muda:**
Mengingat tingginya angka dropout pada usia 18-19 tahun, institusi perlu mengadakan program orientasi akademik dan bimbingan konseling yang lebih intensif pada tahun pertama untuk membantu transisi mahasiswa dari sekolah menengah ke perguruan tinggi.
- **Evaluasi Beasiswa dan Dukungan Finansial:**
Melakukan peninjauan kembali terhadap distribusi beasiswa. Memberikan bantuan finansial tambahan atau skema cicilan khusus bagi mahasiswa berprestasi yang teridentifikasi memiliki kendala ekonomi agar mereka tidak terhenti di tengah jalan.

