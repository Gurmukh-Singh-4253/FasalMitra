import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

# Load Dataset
df = pd.read_csv("Fertilizer_Prediction_Synthetic.csv")
df.rename(columns={"Temparature": "Temperature", "Phosphorous": "Phosphorus"}, inplace=True)

# Encode categorical features
label_encoders = {}
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (EXCLUDE current N, P, K from inputs)
X = df[['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type']]
y_n = df['Nitrogen']
y_p = df['Phosphorus']
y_k = df['Potassium']

# Split dataset
X_train, X_test, y_n_train, y_n_test, y_p_train, y_p_test, y_k_train, y_k_test = train_test_split(
    X, y_n, y_p, y_k, test_size=0.2, random_state=42
)

# Train and evaluate models
models = {}
for nutrient, y_train, y_test in zip(['Nitrogen', 'Phosphorus', 'Potassium'],
                                     [y_n_train, y_p_train, y_k_train],
                                     [y_n_test, y_p_test, y_k_test]):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )

    model.fit(X_train, y_train)
    models[nutrient] = model

    # Save model
    joblib.dump(model, f"{nutrient}_model.pkl")

def load_models():
    try:
        return {
            "Nitrogen": joblib.load("Nitrogen_model.pkl"),
            "Phosphorus": joblib.load("Phosphorus_model.pkl"),
            "Potassium": joblib.load("Potassium_model.pkl")
        }
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Function to predict NPK
def predict_npk(temp, humidity, moisture, soil_type, crop_type):
    try:
        soil_type_val = label_encoders['Soil Type'].transform([soil_type])[0]
        crop_type_val = label_encoders['Crop Type'].transform([crop_type])[0]

        features = pd.DataFrame([[temp, humidity, moisture, soil_type_val, crop_type_val]],
                                columns=X.columns)

        n_pred = models['Nitrogen'].predict(features)[0]
        p_pred = models['Phosphorus'].predict(features)[0]
        k_pred = models['Potassium'].predict(features)[0]

        return round(n_pred, 2), round(p_pred, 2), round(k_pred, 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, 0, 0

# Function to Convert NPK to Fertilizer
def convert_npk_to_fertilizer(n, p, k):
    urea = round(n / 0.46, 2) if n > 0 else 0
    dap = round(p / 0.46, 2) if p > 0 else 0
    mop = round(k / 0.6, 2) if k > 0 else 0
    return urea, dap, mop

# Save results to JSON
def save_results_to_json(recommended_n, recommended_p, recommended_k, urea, dap, mop):
    data = {
        "recommended_npk": {
            "nitrogen": recommended_n,
            "phosphorus": recommended_p,
            "potassium": recommended_k
        },
        "urea": f"{urea} kg/ha",
        "dap": f"{dap} kg/ha",
        "mop": f"{mop} kg/ha"
    }

    with open("fertilizer_recommendation.json", "w") as file:
        json.dump(data, file, indent=4)
    print("Recommendation saved to fertilizer_recommendation.json ✅")

# Main recommendation function
def recommend_fertilizer(temp, humidity, moisture, soil_type, crop_type):
    models = load_models()
    if models is None:
        print("Models not found. Please train and save the models first.")
        return

    recommended_n, recommended_p, recommended_k = predict_npk(temp, humidity, moisture, soil_type, crop_type)
    urea, dap, mop = convert_npk_to_fertilizer(recommended_n, recommended_p, recommended_k)

    print(f"Recommended NPK: N={recommended_n}, P={recommended_p}, K={recommended_k}")
    print(f"Recommended Fertilizers: Urea={urea} kg/ha, DAP={dap} kg/ha, MOP={mop} kg/ha")

    save_results_to_json(recommended_n, recommended_p, recommended_k, urea, dap, mop)

# Interactive input
if __name__ == "__main__":
    print("Enter the following values to get fertilizer recommendations:")
    temp = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    moisture = float(input("Moisture (%): "))
    soil_type = input("Soil Type (e.g., Clay, Loam, Sandy): ")
    crop_type = input("Crop Type (e.g., Rice, Wheat, Maize): ")

    recommend_fertilizer(temp, humidity, moisture, soil_type, crop_type)
