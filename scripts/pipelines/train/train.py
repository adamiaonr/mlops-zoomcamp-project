import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


@task
def load_datasets(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and validation datasets
    """
    train_data = pd.read_pickle(Path(input_dir) / 'train.pkl')
    val_data = pd.read_pickle(Path(input_dir) / 'validation.pkl')

    return train_data, val_data


@task
def prepare_features(
    train_data: pd.DataFrame, val_data: pd.DataFrame, feature_types: dict
):
    """
    Prepares features for a one-hot encoder DictVectorizer input,
    which we use in our {one-hot encoder, model} pipelines
    """
    # isolate {categorical, numerical} features
    features = feature_types['categorical'] + feature_types['numerical']
    # train data
    X_train = train_data[features].to_dict(orient='records')
    y_train = train_data[feature_types['target']].values
    # validation data
    X_val = val_data[features].to_dict(orient='records')
    y_val = val_data[feature_types['target']].values

    return X_train, y_train, X_val, y_val


@task
def train_xgboost(X_train: dict, y_train: np.ndarray, X_val: dict, y_val: np.ndarray):
    """
    Runs hyperparameter optimization experiments on XGBoost regressor.
    Uploads metrics, tags and artifacts to MLFlow tracking server.
    """
    # objective function to be used by hyperopt
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag('model', 'xgboost-regressor')
            mlflow.log_params(params)

            # create sklearn pipeline with steps:
            # - one-hot encoder (aka dict vectorizer)
            # - xgboost regressor
            xgb_regressor = Pipeline(
                [
                    ('ohe', DictVectorizer()),
                    ('xgboost-regressor', xgb.XGBRegressor(**params)),
                ]
            )

            # fit xgb regressor
            xgb_regressor.fit(X_train, y_train)
            # measure inference time
            inference_time = time.time()
            y_pred = xgb_regressor.predict(X_val)
            inference_time = time.time() - inference_time
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            # log rmse and inference time
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('inference_time', inference_time / len(y_pred))
            # log (dict vectorizer, xgboost regressor) pipeline
            mlflow.sklearn.log_model(xgb_regressor, artifact_path='model')

        return {'loss': rmse, 'status': STATUS_OK}

    # hyperparameter search space for xgboost model
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:squarederror',
        'seed': 42,
    }

    # run hyperopt optimization
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials(),
    )


@task
def train_random_forest_regressor(
    X_train: dict, y_train: np.ndarray, X_val: dict, y_val: np.ndarray
):
    """
    Runs hyperparameter optimization experiments on RandomForestRegressor.
    Uploads metrics, tags and artifacts to MLFlow tracking server.
    """

    def objective(params):
        with mlflow.start_run():
            # set model type tag
            mlflow.set_tag('model', 'random-forest-regressor')
            mlflow.log_params(params)

            # create sklearn pipeline with steps:
            # - one-hot encoder (aka dict vectorizer)
            # - random forest regressor
            rf = Pipeline(
                [
                    ('ohe', DictVectorizer()),
                    ('rf-regressor', RandomForestRegressor(**params, n_jobs=-1)),
                ]
            )

            # train RandomForestRegressor
            rf.fit(X_train, y_train)

            # measure inference time on validation data
            inference_time = time.time()
            y_pred = rf.predict(X_val)
            inference_time = time.time() - inference_time
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('inference_time', inference_time / len(y_pred))
            # log (dict vectorizer, rf model) pipeline
            mlflow.sklearn.log_model(rf, artifact_path='model')

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42,
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
        rstate=rstate,
    )


@flow(task_runner=SequentialTaskRunner())
def train(
    input_dir: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    feature_types: dict,
):
    """
    Runs hyperoptimization experiments in two types of models:
        - sklearn's random forest regressor
        - xgboost regressor
    Logs artifacts, parameters, metrics and tags to MLFlow.
    """
    # initialize mlflow : tracking uri and experiment name
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    # load datasets
    train_data, val_data = load_datasets(input_dir)
    # prepare features
    X_train, y_train, X_val, y_val = prepare_features(
        train_data, val_data, feature_types
    )
    # train and evaluate xgboost model
    train_xgboost(X_train, y_train, X_val, y_val)
    # train and evaluate random forest regressor model
    train_random_forest_regressor(X_train, y_train, X_val, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        help="location of 'featurized' NYC bus dataset train, validation and test splits",
        required=True,
    )

    args = parser.parse_args()

    # expand script arguments with mlflow parameters
    train_args = vars(args) | {
        'mlflow_tracking_uri': os.getenv(
            'MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'
        ),
        'mlflow_experiment': os.getenv(
            'MLFLOW_HPO_EXPERIMENT_NAME', 'nyc-bus-delay-predictor-hpo'
        ),
        'feature_types': {
            'categorical': ['BusLine_Direction', 'NextStopPointName'],
            'numerical': ['TimeOfDayInSeconds', 'DayOfWeek'],
            'target': 'DelayAtStop',
        },
    }

    # start training
    train(**train_args)

    sys.exit(0)
