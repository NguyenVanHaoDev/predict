from flask import request, jsonify, render_template
from app.model_utils import load_model, load_scaler
from . import main_bp

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data.get("features", [])

    model = load_model()
    scaler = load_scaler()

    if model is None or scaler is None:
        return jsonify({"error": "Lỗi khi tải mô hình hoặc scaler."}), 500

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)

    definitions = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    return jsonify({"prediction": definitions[prediction[0]]})
