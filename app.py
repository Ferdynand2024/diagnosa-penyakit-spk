import streamlit as st
import pandas as pd
import pickle

st.image("dokter.png", width=720)

# Load model + encoders
data = pickle.load(open("model.pkl", "rb"))
model = data["model"]
encoders = data["encoders"]

st.title("🩺 Sistem Diagnosis Penyakit Berdasarkan Gejala")
st.write("Masukkan data berikut untuk memprediksi penyakit")

# Input data
mapper = {"Tidak": 0, "Ya": 1}

fever = st.selectbox("Demam?", ["Tidak", "Ya"])
cough = st.selectbox("Batuk?", ["Tidak", "Ya"])
fatigue = st.selectbox("Kelelahan?", ["Tidak", "Ya"])
breathing = st.selectbox("Kesulitan Bernafas?", ["Tidak", "Ya"])

age = st.number_input("Usia", 0, 120)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
blood_pressure = st.number_input("Tekanan Darah", 50, 200)
cholesterol = st.number_input("Kolesterol", 100, 400)

# Mapping gender ke angka sesuai encoder
gender_encoded = encoders["Gender"].transform([gender])[0]

# Siapkan dataframe sesuai fitur model
df = pd.DataFrame([{
    "Fever": mapper[fever],
    "Cough": mapper[cough],
    "Fatigue": mapper[fatigue],
    "Difficulty Breathing": mapper[breathing],
    "Age": age,
    "Gender": gender_encoded,
    "Blood Pressure": blood_pressure,
    "Cholesterol Level": cholesterol,
    "Outcome Variable": 0  # dummy karena model butuh
}])

if st.button("Diagnosa"):
    pred = model.predict(df)[0]

    # Decode hasil penyakit
    disease = encoders["Disease"].inverse_transform([pred])[0]

    st.success(f"Hasil Prediksi Penyakit: **{disease}**")


