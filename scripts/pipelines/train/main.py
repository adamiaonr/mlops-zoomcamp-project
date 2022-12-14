import argparse
import os
import time
from pathlib import Path

from prefect import flow
from prefect.context import get_run_context
from prefect.task_runners import SequentialTaskRunner
from prepare import prepare
from select_and_register import select_and_register
from train import train


@flow(name='training-pipeline', task_runner=SequentialTaskRunner())
def main(input_dir: str = "data", output_dir: str = "data/featurized"):
    ctx = get_run_context()

    env_vars = {
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
        'dataset_sample': os.getenv('DATASET_SAMPLE', '100000'),
        'number_top_runs': os.getenv('NUMBER_TOP_RUNS', '5'),
    }

    feature_types = {
        'categorical': ['BusLine_Direction', 'NextStopPointName'],
        'numerical': ['TimeOfDayInSeconds', 'DayOfWeek'],
        'target': 'DelayAtStop',
    }

    # ensure data dirs exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    months = [6]
    mlflow_suffix = f"{int(time.mktime(ctx.flow_run.expected_start_time.timetuple()))}"

    # download and pre-process dataset
    prepare(
        input_dir,
        output_dir,
        months,
        feature_types,
        nrows=int(env_vars['dataset_sample']),
    )

    # run hyperparameter optimization experiments
    train(
        output_dir,
        env_vars['mlflow_tracking_uri'],
        f"{env_vars['mlflow_hpo_experiment']}-{mlflow_suffix}",
        feature_types,
    )

    # select best model and save it in registry
    select_and_register(
        output_dir,
        int(env_vars['number_top_runs']),
        {
            'mlflow_tracking_uri': env_vars['mlflow_tracking_uri'],
            'mlflow_hpo_experiment': f"{env_vars['mlflow_hpo_experiment']}-{mlflow_suffix}",
            'mlflow_select_experiment': f"{env_vars['mlflow_select_experiment']}-{mlflow_suffix}",
            'mlflow_model_name': env_vars['mlflow_model_name'],
        },
        feature_types,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir", help="location of 'raw' NYC bus dataset", required=True
    )

    parser.add_argument(
        "--output_dir",
        help="location where processed data will be saved",
        required=True,
    )

    args = parser.parse_args()

    main(**vars(args))
