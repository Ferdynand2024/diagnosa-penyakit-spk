import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("data/Disease_symptom_and_patient_profile_dataset.csv")

# Simpan encoder untuk setiap kolom kategori
encoders = {}

# Label encode semua kolom bertipe object (teks)
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Tentukan fitur dan label
X = df.drop("Disease", axis=1)   # Disease = label utama
y = df["Disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Buat model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Simpan model + encoders
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": encoders, "features": list(X.columns)}, f)

print("Model berhasil dibuat!")
