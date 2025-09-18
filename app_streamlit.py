import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_tomat.joblib")
scaler = joblib.load("scaler_klasifikasi_tomat.joblib")

st.title("Klasifikasi Tomat")

berat = st.slider("Berat Tomat",0, 200, 50)
kekenyalan = st.slider("Tingkat Kekenyalan",0.0, 10.0, 3.5)
kadar_gula = st.slider("Kadar Gula", 0.0, 10.0, 4.2)
tebal_kulit = st.slider("Tebal Kulit", 0.0, 1.0, 0.8)

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[berat,kekenyalan,kadar_gula,tebal_kulit]], columns=["berat", "kekenyalan", "kadar_gula", "tebal_kulit"])	
	data_baru_scaled = scaler.transform(data_baru)
	prediksi = model.predict(data_baru_scaled)[0]
	presentase = max(model.predict_proba(data_baru_scaled)[0])
	st.success(f"Prediksi {prediksi} keyakinan {presentase*100:.2f}%")
	st.balloons()