from flask import Flask, request, jsonify
import json
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def load_models():
    try:
        models = {
            'Nitrogen': joblib.load(r'C:\Users\HP\OneDrive\Desktop\model\Nitrogen_model.h5'),
            'Phosphorus': joblib.load(r"C:\Users\HP\OneDrive\Desktop\model\Phosphorus_model.h5"),
            'Potassium': joblib.load(r"C:\Users\HP\OneDrive\Desktop\model\Potassium_model.h5")
        }

        # Define all possible categories (replace with your actual lists)
        all_soil_types = ['black-soil', 'clayey-soil', 'loamy-soil', 'red-soil', 'sandy-soil']
        all_crop_types = ['pulses', 'sugarcane', 'tobacco', 'wheat', 'barley', 'cotton', 'ground-nuts', 'maize', 'millets', 'oil-seeds', 'paddy']

        label_encoders = {}
        for col, categories in [('Soil Type', all_soil_types), ('Crop Type', all_crop_types)]:
            le = LabelEncoder()
            le.fit(categories)  # Train the LabelEncoder
            label_encoders[col] = le
        return models, label_encoders
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

models, label_encoders = load_models()

def predict_npk(temp, humidity, moisture, soil_type, crop_type):
    try:
        soil_type_val = label_encoders['Soil Type'].transform([soil_type])[0]
        crop_type_val = label_encoders['Crop Type'].transform([crop_type])[0]
        features = pd.DataFrame([[temp, humidity, moisture, soil_type_val, crop_type_val]],
                                columns=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type'])

        n_pred = models['Nitrogen'].predict(features)[0]
        p_pred = models['Phosphorus'].predict(features)[0]
        k_pred = models['Potassium'].predict(features)[0]

        return round(n_pred, 2), round(p_pred, 2), round(k_pred, 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, 0, 0

@app.route('/recommendations', methods=['POST'])
def recommend_fertilizer():
    data = request.get_json()
    temp = data.get('temperature')
    humidity = data.get('humidity')
    moisture = data.get('moisture')
    soil_type = data.get('soilType')
    crop_type = data.get('cropType')
    n, p, k = predict_npk(temp, humidity, moisture, soil_type, crop_type)
    return jsonify({'nitrogen': n, 'phosphorus': p, 'potassium': k, 'mixture': 'Recommended mixture'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)