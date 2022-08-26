import os
import sys
import typing
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task


@dataclass
class ProductionModelInfo:
    run_id: str
    rmse: float
    inference_time: float


@task
def get_latest_production_info(
    client: MlflowClient, mlflow_model_name: str
) -> typing.Union[ProductionModelInfo, None]:
    # get latest production version
    try:
        latest_production_versions = client.get_latest_versions(
            mlflow_model_name, ['Production']
        )
    except mlflow.exceptions.RestException:
        latest_production_versions = []

    if latest_production_versions:
        run = client.get_run(run_id=latest_production_versions[-1].run_id)
        rmse = run.data.metrics['test_rmse']
        inference_time = run.data.metrics['inference_time']

        return ProductionModelInfo(
            latest_production_versions[-1].run_id, rmse, inference_time
        )

    return None


@task
def promote_candidate_model(
    client: MlflowClient, mlflow_model_name: str, pmi: ProductionModelInfo
):
    # get latest versions in 'Staging'
    latest_staging_versions = client.get_latest_versions(mlflow_model_name, ['Staging'])

    # promote candidate model to production if :
    #  - (i)    : test_rmse is lower than current production model
    #  - (ii)   : inference time is within 10% of that of production
    #  - (iii)  : inference time is < .5 seconds
    for version in latest_staging_versions:
        run = client.get_run(run_id=version.run_id)
        if not pmi or (
            run.data.metrics['test_rmse'] < pmi.rmse
            and run.data.metrics['inference_time'] < (pmi.inference_time * 1.1)
            and run.data.metrics['inference_time'] < 0.5
        ):
            client.transition_model_version_stage(
                name=mlflow_model_name,
                version=version.version,
                stage='Production',
                archive_existing_versions=True,
            )

            print(
                f'promoted {mlflow_model_name} w/ version {version.version} to Production'
            )


@flow
def promote(mlflow_tracking_uri: str, mlflow_model_name: str):
    # initialize mlflow : tracking uri and experiment name
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = MlflowClient()

    # get latest production version
    pmi = get_latest_production_info(client, mlflow_model_name)
    # promote candidate model to production (if conditions are satisfied)
    promote_candidate_model(client, mlflow_model_name, pmi)


if __name__ == '__main__':
    # expand script arguments with mlflow parameters
    promote_args = {
        'mlflow_tracking_uri': os.getenv(
            'MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'
        ),
        'mlflow_model_name': os.getenv('MLFLOW_MODEL_NAME', 'nyc-bus-delay-predictor'),
    }

    # select best model and register it
    promote(**promote_args)

    sys.exit(0)
