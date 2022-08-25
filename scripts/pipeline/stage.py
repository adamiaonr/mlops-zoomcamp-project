import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def stage(mlflow_tracking_uri: str, mlflow_model_name: str):
    # initialize mlflow : tracking uri and experiment name
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = MlflowClient()

    # get latest production version
    try:
        latest_production_versions = client.get_latest_versions(
            mlflow_model_name, ['Production']
        )
    except mlflow.exceptions.RestException:
        latest_production_versions = []

    if latest_production_versions:
        production_run = client.get_run(run_id=latest_production_versions[-1].run_id)
        production_rmse = production_run.data.metrics['test_rmse']
        production_inference_time = production_run.data.metrics['inference_time']

    # get latest staging versions
    latest_staging_versions = client.get_latest_versions(mlflow_model_name, ['Staging'])

    # update version in production if :
    #  - (i)    : test_rmse is lower than current production model
    #  - (ii)   : inference time is within 10% of that of production
    #  - (iii)  : inference time is < .5 seconds
    for version in latest_staging_versions:
        run = client.get_run(run_id=version.run_id)
        if not latest_production_versions or (
            run.data.metrics['test_rmse'] < production_rmse
            and run.data.metrics['inference_time'] < (production_inference_time * 1.1)
            and run.data.metrics['inference_time'] < 0.5
        ):
            client.transition_model_version_stage(
                name=mlflow_model_name,
                version=version.version,
                stage='Production',
                archive_existing_versions=True,
            )

            print(
                f'promoted {mlflow_model_name} version {version.version} to Production'
            )


if __name__ == '__main__':
    # expand script arguments with mlflow parameters
    stage_args = {
        'mlflow_tracking_uri': os.getenv(
            'MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'
        ),
        'mlflow_model_name': os.getenv('MLFLOW_MODEL_NAME', 'nyc-bus-delay-predictor'),
    }

    # select best model and register it
    stage(**stage_args)

    sys.exit(0)
