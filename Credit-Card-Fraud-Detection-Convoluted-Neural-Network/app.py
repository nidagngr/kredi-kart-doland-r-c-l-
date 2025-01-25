import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import scale, StandardScaler

# Modeli yükleme
model_path = "C:\\Users\\asus\\nida2\\Credit-Card-Fraud-Detection-Convoluted-Neural-Network\\model - Kopya.pkl"
data_path = "C:\\Users\\asus\\nida2\\Credit-Card-Fraud-Detection-Convoluted-Neural-Network\\creditcard.csv"

try:
    # Modeli pickle formatında yükleme
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")
    st.stop()

# Veri setini yükleme
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    pca_features = [f"V{i}" for i in range(1, 29)]
    df["Time"] = scale(df["Time"])
    df["Amount"] = scale(df["Amount"])
    return df, pca_features

df_data, pca_features = load_data()

# Streamlit arayüzü
st.title("Kredi Kartı Dolandırıcılık Tespit Sistemi")
st.write("Bu sistem, CNN modeli kullanarak kredi kartı işlemlerindeki dolandırıcılıkları tespit eder.")

# Zamanı saniyeye çevirme fonksiyonu
def time_to_seconds(time_str):
    try:
        hours, minutes = map(int, time_str.split(":"))
        return hours * 3600 + minutes * 60
    except ValueError:
        st.error(f"Zaman formatı hatalı: {time_str}. Lütfen 'HH:MM' formatında girin.")
        return None

# Giriş türü seçimi
input_type = st.radio("Veri giriş yöntemi seçin:", ["Manuel Veri Girişi", "CSV Dosyası Yükleme"])

if input_type == "Manuel Veri Girişi":
    st.subheader("Manuel Veri Girişi")
    
    # Zamanlar ve miktarlar için liste oluşturuluyor
    times = []
    amounts = []
    for i in range(1, 11):
        time = st.text_input(f"İşlem {i} Zamanı (HH:MM):", value="00:00", key=f"time_{i}")
        amount = st.number_input(f"İşlem {i} Miktarı:", value=0.0, key=f"amount_{i}")
        times.append(time)
        amounts.append(amount)

    # Kullanıcının diğer özellikleri rastgele doldurmak için seçenek
    use_random_features = st.checkbox("Diğer özellikleri rastgele doldur")

    if st.button("Tahmin Yap"):
        try:
            # Zamanları saniyeye çevirme
            times_in_seconds = []
            for time in times:
                if time:
                    total_seconds = time_to_seconds(time)
                    if total_seconds is not None:  # Eğer zaman dönüştürülemediyse, işlemi sonlandır
                        times_in_seconds.append(total_seconds)

            # Zaman farklarını hesaplama
            time_differences = np.diff(times_in_seconds).tolist() if len(times_in_seconds) > 1 else [0]
            total_amount = sum(amounts)  # Miktarların toplamı

            # PCA özelliklerinden rastgele bir örnek alınır
            random_sample = df_data[pca_features].sample(n=1, random_state=42).values.flatten()

            # Zaman farkları ve toplam miktarı düz bir listeye ekleyin
            input_data = np.hstack((time_differences[:1], [total_amount], random_sample[:28]))  # 30 özellik

            # Giriş verisini şekillendir
            input_data = input_data.reshape(1, -1, 1)

            # Tahmin
            y_pred = model.predict(input_data)

            # İkili sınıflandırma: Dolandırıcılık olasılığı
            fraud_probability = y_pred[0][1]  # İkinci sınıfın olasılığı
            predicted_class = int(fraud_probability > 0.5)  # Olasılık > 0.5 => Dolandırıcılık

            # Sonuçları göster
            st.success(f"Tahmin Edilen Sınıf: {'Dolandırıcılık' if predicted_class == 1 else 'Normal İşlem'}")
            st.write(f"Modelin Dolandırıcılık Olasılığı: {fraud_probability:.2%}")
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

elif input_type == "CSV Dosyası Yükleme":
    st.subheader("CSV Dosyası Yükleme")
    uploaded_file = st.file_uploader("CSV dosyasını yükleyin:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Dosyayı yükle ve göster
            data = pd.read_csv(uploaded_file)
            st.write("Yüklenen Veri:")
            st.write(data.head())

            if st.button("Tahmin Yap"):
                # Veri hazırlama
                X = data.drop("Class", axis=1, errors="ignore").values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = X_scaled.reshape(X_scaled.shape[0], -1, 1)

                # Tahmin
                predictions = model.predict(X_scaled)
                fraud_probabilities = predictions[:, 1]  # İkinci sınıfın olasılığı
                predicted_classes = (fraud_probabilities > 0.5).astype(int)

                # Sonuçları ekle
                data["Predicted_Class"] = predicted_classes
                data["Fraud_Probability"] = fraud_probabilities
                st.write("Tahmin Sonuçları:")
                st.write(data[["Predicted_Class", "Fraud_Probability"]])
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
