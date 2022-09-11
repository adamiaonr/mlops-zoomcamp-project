import os

import mlflow
import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient

# mlflow vars
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'nyc-bus-delay-predictor')

# mongodb vars
MONGO_DB_ADDRESS = os.getenv('MONGO_DB_ADDRESS', 'mongodb://localhost:27017')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'nyc_bus')
MONGO_DB_COLLECTION = os.getenv('MONGO_DB_COLLECTION', 'data')

# evidently ai monitoring vars
EVIDENTLY_SERVICE_ADDRESS = os.getenv(
    'EVIDENTLY_SERVICE_ADDRESS', 'http://localhost:5500'
)
EVIDENTLY_DATASET_NAME = os.getenv('EVIDENTLY_DATASET_NAME', 'nyc_bus')


# init mongodb client for monitoring services
mongo_client = MongoClient(MONGO_DB_ADDRESS)
mongo_db = mongo_client.get_database(MONGO_DB_NAME)
mongo_collection = mongo_db.get_collection(MONGO_DB_COLLECTION)

# extract production model from s3 artifact store using mlflow
# pylint: disable-next=W0511
# FIXME: make this independent of mlflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")


def prepare_features(data: dict):
    """
    Processes the bus stop info enclosed in web-service requests.
    Makes bus stop info ready to be passed to delay prediction models.
    """
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


def predict(features: dict) -> float:
    """
    Takes bus stop info features (already processed),
    returns a delay prediction, in seconds.
    """
    y_pred = model.predict(features)
    return float(y_pred[0])


# bus delay flask application
app = Flask('nyc-bus-delay-prediction')


def save_to_monitoring_db(stop_info: dict, prediction: int):
    """
    Saves a pair of {input features, prediction value} on
    monitoring database.
    """
    record = stop_info.copy()
    record['prediction'] = prediction
    # store record in monitoring (mongodb) database
    mongo_collection.insert_one(record)


def send_to_monitoring_service(stop_info: dict, prediction: int):
    """
    Sends a pair of {input features, prediction value} to
    monitoring service.
    """
    record = stop_info.copy()
    record['prediction'] = prediction

    requests.post(
        f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/{EVIDENTLY_DATASET_NAME}", json=[record]
    )


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Flask app endpoint function.
    Listens to web-service requests carrying bus stop info.
    Returns the corresponding bus delay.
    """
    stop_info = request.get_json()
    features = prepare_features(stop_info)
    y_pred = predict(features)

    result = {'bus delay': y_pred, 'model_version': model.metadata.run_id}

    # monitoring functions
    save_to_monitoring_db(features, y_pred)
    send_to_monitoring_service(features, y_pred)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
