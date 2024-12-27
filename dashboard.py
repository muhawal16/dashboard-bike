import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
day = pd.read_csv("day.csv")

# Menyiapkan data untuk prediksi
features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
target = 'cnt'

# Split data untuk pelatihan dan pengujian
X = day[features]
y = day[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Menambahkan fitur input pengguna untuk prediksi
st.title("Aplikasi Prediksi Penyewaan Sepeda")
st.sidebar.header("Masukkan Kondisi untuk Prediksi")

season = st.sidebar.selectbox("Musim", options=[1, 2, 3, 4], format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])
yr = st.sidebar.selectbox("Tahun", options=[0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
mnth = st.sidebar.slider("Bulan", min_value=1, max_value=12, step=1)
holiday = st.sidebar.selectbox("Hari Libur?", options=[0, 1], format_func=lambda x: "Bukan Libur" if x == 0 else "Libur")
weekday = st.sidebar.slider("Hari dalam Minggu (0=Senin, 6=Minggu)", min_value=0, max_value=6)
workingday = st.sidebar.selectbox("Hari Kerja?", options=[0, 1], format_func=lambda x: "Bukan Hari Kerja" if x == 0 else "Hari Kerja")
weathersit = st.sidebar.selectbox("Kondisi Cuaca", options=[1, 2, 3, 4], format_func=lambda x: {
    1: "Clear/Few Clouds",
    2: "Mist/Cloudy",
    3: "Light Rain/Snow",
    4: "Heavy Rain/Snow"
}[x])
temp = st.sidebar.slider("Suhu (dalam skala normalisasi)", min_value=0.0, max_value=1.0, step=0.01)
atemp = st.sidebar.slider("Suhu Terasa (dalam skala normalisasi)", min_value=0.0, max_value=1.0, step=0.01)
hum = st.sidebar.slider("Kelembapan (dalam skala normalisasi)", min_value=0.0, max_value=1.0, step=0.01)
windspeed = st.sidebar.slider("Kecepatan Angin (dalam skala normalisasi)", min_value=0.0, max_value=1.0, step=0.01)

# Membuat prediksi berdasarkan input pengguna
user_input = [[season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed]]
predicted_rentals = model.predict(user_input)[0]

st.header("Hasil Prediksi")
st.write(f"Jumlah penyewaan sepeda yang diprediksi: **{int(predicted_rentals)}** sepeda")

# Menampilkan evaluasi model
st.sidebar.subheader("Evaluasi Model")
st.sidebar.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.sidebar.write(f"R2 Score: {r2:.2f}")

# Grafik yang lebih menarik untuk hasil prediksi
st.subheader("Hasil Prediksi Penyewaan Sepeda")

# Menggunakan seaborn untuk desain yang lebih menarik
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")

# Membuat bar plot dengan warna gradient
sns.barplot(x=['Penyewaan Sepeda Prediksi'], y=[predicted_rentals], palette="Blues_d")

# Menambahkan detail pada grafik
plt.title('Prediksi Jumlah Penyewaan Sepeda', fontsize=16)
plt.xlabel('Jenis Prediksi', fontsize=12)
plt.ylabel('Jumlah Penyewaan Sepeda', fontsize=12)

# Menambahkan angka pada setiap bar untuk memudahkan pembacaan
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=14, color='black', fontweight='bold', xytext=(0, 5),
                       textcoords='offset points')

st.pyplot(plt)
