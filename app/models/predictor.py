import numpy as np
import joblib
import os

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rockfall_model.pkl")

# Load model at import
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model. Train first using train_model.py. Error: {e}")
    model = None


def predict(features: dict):
    """
    Predicts rockfall risk from given features.
    Features dict must contain:
    - rainfall
    - temperature
    - slope
    - wind_speed
    - displacement_rate
    - vibration
    """

    if model is None:
        raise RuntimeError("Model not loaded. Train first with train_model.py")

    input_vec = np.array([[ 
        features["rainfall"],
        features["temperature"],
        features["slope"],
        features["wind_speed"],
        features["displacement_rate"],
        features["vibration"],
    ]])

    prob = model.predict_proba(input_vec)[0][1]
    label = int(prob > 0.5)

    # consistent API response
    return {
        "risk": "high" if prob > 0.7 else ("medium" if prob > 0.4 else "low"),
        "probability": round(float(prob), 3),
        "rockfall_predicted": label
    }



# Example run (manual test)
if __name__ == "_main_":
    sample_features = {
        "rain": 35,
        "temperature": 24,
        "slope": 38,
        "wind_speed": 6,
        "displacement_rate": 0.09,
        "vibration": 2.1,
    }
    result = predict(sample_features)
    print("🔮 Prediction result:", result)