import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.model_selection import train_test_split

from src import preprocessing
from utils import get_kaggle_client


@task(retries=3)
def download_dataset(kaggle_dataset_id: str, output_dir: str, months: list[int] = None):
    # if no month list is specified, download everything
    if not months:
        # do not download if files already exist
        if not list(Path(output_dir).glob('mta_17*.csv*')):
            get_kaggle_client().dataset_download_files(
                kaggle_dataset_id, path=Path(output_dir), force=False, unzip=True
            )
    else:
        for m in months:
            filename = f"mta_17{int(m):02d}.csv"
            get_kaggle_client().dataset_download_file(
                kaggle_dataset_id,
                file_name=filename,
                path=Path(output_dir),
                force=False,
            )


@task
def load_data(input_dir: str, months: list[int], **kwargs) -> pd.DataFrame:
    return preprocessing.load_data(Path(input_dir), months=months, **kwargs)


@task
def clean_data(data: pd.DataFrame, feature_types: dict) -> pd.DataFrame:
    preprocessing.fix_datetime_columns(data)
    preprocessing.fix_scheduled_arrival_time_column(data)
    preprocessing.add_extra_columns(data)
    # remove data points with abnormal delay durations
    data = preprocessing.remove_outliers(data, {'DelayAtStop': [-500, 2500]})

    categorical = feature_types['categorical']
    numerical = feature_types['numerical']
    target = [feature_types['target']]

    # only keep columns of interest, drop nan
    data = data[categorical + numerical + target].dropna().reset_index(drop=True)
    # encode categorical columns as strings, 'stripped' of newlines and padding / trailing spaces
    data[categorical] = data[categorical].astype(str)
    data[categorical] = data[categorical].apply(lambda s: s.str.strip())
    # encode numerical + target variables as int
    data[numerical + target] = data[numerical + target].astype(int)

    return data


@task
def split_datasets(data: pd.DataFrame, output_dir: str):
    # split dataset into train, validation and test sets
    train_data, test_data = train_test_split(data, test_size=0.30, shuffle=False)
    train_data, validation_data = train_test_split(
        train_data, test_size=0.20, shuffle=False
    )

    print(
        (
            f"dataset split sizes :\n\ttrain : {len(train_data)}"
            f"\n\tvalidation : {len(validation_data)}\n\ttest : {len(test_data)}"
        )
    )

    # save datasets
    train_data.to_pickle(Path(output_dir) / 'train.pkl')
    validation_data.to_pickle(Path(output_dir) / 'validation.pkl')
    test_data.to_pickle(Path(output_dir) / 'test.pkl')


@flow(task_runner=SequentialTaskRunner())
def prepare(
    input_dir: str, output_dir: str, months: list[int], feature_types: dict, **kwargs
):
    """
    Downloads data from kaggle.
    Cleans data.
    Saves train, validation and test datasets in .parquet format,
    given input and output directories.
    """
    # download data from kaggle into input_dir
    kaggle_id = os.getenv(
        'KAGGLE_DATASET_ID', 'stoney71/new-york-city-transport-statistics'
    )
    download_dataset(kaggle_id, input_dir, months)
    # load data
    data = load_data(Path(input_dir), months=months, **kwargs)
    # cleanup data
    data = clean_data(data, feature_types)
    # split datasets and save
    split_datasets(data, output_dir)


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

    parser.add_argument(
        "--months",
        nargs='+',
        help=(
            "NYC bus dataset months to be used, separated by whitspaces, "
            "e.g. '--months 6 8' (default: [6])"
        ),
        default=['6'],
        required=False,
    )

    args = parser.parse_args()

    prepare_args = vars(args) | {
        'feature_types': {
            'categorical': ['BusLine_Direction', 'NextStopPointName'],
            'numerical': ['TimeOfDayInSeconds', 'DayOfWeek'],
            'target': 'DelayAtStop',
        },
    }

    prepare(**prepare_args, nrows=100000)

    sys.exit(0)
