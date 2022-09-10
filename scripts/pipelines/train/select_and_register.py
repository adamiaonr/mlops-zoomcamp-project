import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Run, ViewType
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.metrics import mean_squared_error


@task
def load_features(input_dir: str, feature_types: dict) -> tuple[dict, np.ndarray]:
    """
    Loads test dataset.
    Prepares features for a one-hot encoder DictVectorizer input,
    which we use in our {one-hot encoder, model} pipelines
    """
    # load test dataset
    test_data = pd.read_pickle(Path(input_dir) / 'test.pkl')
    # isolate {categorical, numerical} features
    features = feature_types['categorical'] + feature_types['numerical']
    # prepare features
    X_test = test_data[features].to_dict(orient='records')
    y_test = test_data[feature_types['target']].values

    return X_test, y_test


@task
def get_top_n_runs(mlflow_hpo_experiment: str, number_top_runs: int) -> PagedList[Run]:
    """
    Returns top n best runs from 'hpo' MLFlow experiments, i.e.
    In this case, 'best' is equivalent to 'lowest RMSE'.
    """
    client = MlflowClient()

    # extract hpo experiment
    hpo_experiment = client.get_experiment_by_name(mlflow_hpo_experiment)
    # search top n runs according to rmse
    runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=number_top_runs,
        order_by=["metrics.rmse ASC"],
    )

    return runs


@task
def test_model(X_test: dict, y_test: np.ndarray, run: Run):
    """
    Runs a 'select' MLFlow experiment, i.e.:
        - Loads model from MLFlow experiment given a run
        - Calculates RMSE w/ test data
        - Logs 'select' experiment artifacts, parameters, metrics and tags on MLFlow
    """
    # load model from run
    model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

    with mlflow.start_run():
        mlflow.set_tag('model', run.data.tags['model'])
        mlflow.log_params(run.data.params)

        # run predictions on test set
        inference_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - inference_time

        # log inference time
        mlflow.log_metric('inference_time', inference_time / len(y_pred))
        # log validation error
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric('test_rmse', test_rmse)

        # log (dict vectorizer, model) pipeline
        mlflow.sklearn.log_model(model, artifact_path='model')


@task
def register_best_model(mlflow_select_experiment: str, model_name: str) -> bool:
    """
    Registers best model from 'select' experiments to model registry, w/ stage 'Production'.
    Registration occurs if the test RMSE of the best 'select' run is lower
    than that of the current model in production.
    Returns True if the model has been registered, False otherwise.
    """
    client = MlflowClient()

    # select model with lowest test RMSE
    select_experiment = client.get_experiment_by_name(mlflow_select_experiment)
    best_run = client.search_runs(
        experiment_ids=select_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    # extract rmse of current production model
    try:
        production = client.get_latest_versions(model_name, ['Production'])
    except mlflow.exceptions.RestException:
        production = []

    if production:
        production_run = client.get_run(run_id=production[-1].run_id)
        production_rmse = production_run.data.metrics['test_rmse']

    # register and transition to 'Production' if candidate model performs better than
    # current production model or if no production model exists
    if not production or best_run.data.metrics['test_rmse'] < production_rmse:
        # registration
        model_version = mlflow.register_model(
            model_uri=f"runs:/{best_run.info.run_id}/model",
            name=model_name,
            tags=best_run.data.tags,
        )

        # transition to 'Production'
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Production',
            archive_existing_versions=True,
        )

        return True

    return False


@flow(task_runner=SequentialTaskRunner())
def select_and_register(
    input_dir: str, number_top_runs: int, mlflow_params: dict, feature_types: dict
) -> bool:
    """
    Selects best model according to test RMSE, and registers it in the
    model registry, if the test RMSE is lower than production test RMSE.
    Returns True if the model has been registered, False otherwise.
    """
    # initialize mlflow : tracking uri and experiment name
    mlflow.set_tracking_uri(mlflow_params['mlflow_tracking_uri'])
    mlflow.set_experiment(mlflow_params['mlflow_select_experiment'])

    # load features from test dataset
    X_test, y_test = load_features(input_dir, feature_types)
    # get top n performing experiment runs
    runs = get_top_n_runs(mlflow_params['mlflow_hpo_experiment'], number_top_runs)
    # run top performing experiments on test data
    for run in runs:
        test_model(X_test, y_test, run)
    # register best candidate model
    res = register_best_model(
        mlflow_params['mlflow_select_experiment'], mlflow_params['mlflow_model_name']
    )

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        help="location of 'featurized' NYC bus dataset train, validation and test splits",
        required=True,
    )

    parser.add_argument(
        "--number_top_runs",
        help="choose top n runs for final selection (default: 5)",
        default=5,
        required=False,
    )

    args = parser.parse_args()

    # expand script arguments with mlflow parameters
    mlflow_args = {
        'mlflow_tracking_uri': os.getenv(
            'MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'
        ),
        'mlflow_hpo_experiment': os.getenv(
            'MLFLOW_HPO_EXPERIMENT_NAME', 'nyc-bus-delay-predictor-hpo'
        ),
        'mlflow_select_experiment': os.getenv(
            'MLFLOW_SELECT_EXPERIMENT_NAME', 'nyc-bus-delay-predictor-select'
        ),
        'mlflow_model_name': os.getenv('MLFLOW_MODEL_NAME', 'nyc-bus-delay-predictor'),
    }

    feat_types = {
        'categorical': ['BusLine_Direction', 'NextStopPointName'],
        'numerical': ['TimeOfDayInSeconds', 'DayOfWeek'],
        'target': 'DelayAtStop',
    }

    rc = select_and_register(
        args.input_dir, args.number_top_runs, mlflow_args, feat_types
    )

    print(
        f"model {mlflow_args['mlflow_model_name']} {'updated' if rc else 'not updated'}"
    )

    sys.exit(0)
