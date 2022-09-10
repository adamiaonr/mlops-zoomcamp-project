import os

import mlflow
from flask import Flask, jsonify, request

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'nyc-bus-delay-predictor')


def prepare_features(data: dict):
    features = {}
    # categorical features
    features[
        'BusLine_Direction'
    ] = f"{data['PublishedLineName']}_{data['DirectionRef']}"
    features['NextStopPointName'] = data['NextStopPointName']
    # numerical features
    features['TimeOfDayInSeconds'] = int(data['TimeOfDayInSeconds'])
    features['DayOfWeek'] = int(data['DayOfWeek'])

    return features


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")


def predict(features: dict) -> float:
    y_pred = model.predict(features)
    return float(y_pred[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    stop_info = request.get_json()
    features = prepare_features(stop_info)
    y_pred = predict(features)

    result = {'bus delay': y_pred, 'model_version': model.metadata.run_id}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
