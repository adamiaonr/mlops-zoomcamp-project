import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi  # pylint: disable-msg=E0611
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from src import preprocessing
from utils import FeaturizedData, save_featurized_data


@task(retries=3)
def download_dataset(kaggle_dataset_id: str, output_dir: str):
    # do not download if files already exist
    if not list(Path(output_dir).glob('mta_17*.csv')):
        # create kaggle API object and authenticate (assumes ~/.kaggle exists)
        api = KaggleApi()
        api.authenticate()

        # download dataset from kaggle
        api.dataset_download_files(
            kaggle_dataset_id, path=Path(output_dir), force=False, unzip=True
        )


@task
def load_data(input_dir: str, months: list[int], **kwargs) -> pd.DataFrame:
    return preprocessing.load_data(Path(input_dir), months=months, **kwargs)


@task
def clean_data(data: pd.DataFrame):
    # apply dataset 'cleaning' functions
    preprocessing.fix_datetime_columns(data)
    preprocessing.fix_scheduled_arrival_time_column(data)
    # add extra columns for extra features
    preprocessing.add_extra_columns(data)


def __featurize(
    data: pd.DataFrame,
    dv: DictVectorizer,
    categorical: list[str],
    numerical: list[str],
    fit_dv: bool = False,
) -> tuple[np.ndarray, DictVectorizer]:
    feature_dicts = data[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        X = dv.fit_transform(feature_dicts)
    else:
        X = dv.transform(feature_dicts)

    return X, dv


@task
def featurize(data: pd.DataFrame) -> FeaturizedData:
    """
    Returns feature arrays and a dict vectorizer.
    """
    # define categorial and numerical features
    categorical = ['BusLine_Direction', 'NextStopPointName']
    numerical = ['TimeOfDayInSeconds', 'DayOfWeek']
    # target variable
    target = 'DelayAtStop'

    # drop nan values in required columns
    data = data.dropna(subset=categorical + numerical + [target]).reset_index(drop=True)

    # split dataset into train, validation and test sets
    train_data, test_data = train_test_split(data, test_size=0.30)
    train_data, validation_data = train_test_split(train_data, test_size=0.20)

    print(
        (
            f"dataset split sizes :\n\ttrain : {len(train_data)}"
            f"\n\tvalidation : {len(validation_data)}\n\ttest : {len(test_data)}"
        )
    )

    # featurize splits
    dv = DictVectorizer()
    # - get dict vectorizer, train, validation and test splits
    X_train, dv = __featurize(train_data, dv, categorical, numerical, fit_dv=True)
    X_val, _ = __featurize(validation_data, dv, categorical, numerical, fit_dv=False)
    X_test, _ = __featurize(test_data, dv, categorical, numerical, fit_dv=False)

    return FeaturizedData(
        dv,
        X_train,
        train_data[target].values,
        X_val,
        validation_data[target].values,
        X_test,
        test_data[target].values,
    )


@flow(task_runner=SequentialTaskRunner())
def prepare(
    input_dir: str, output_dir: str, months: list[int], kaggle_id: str, **kwargs
):
    """
    Saves feature and target arrays + dict vectorizer, given input and output data paths.
    """
    # download data from kaggle into input_dir
    download_dataset(kaggle_id, input_dir)
    # load data
    data = load_data(Path(input_dir), months=months, **kwargs)
    # cleanup data
    clean_data(data)
    # get dict vectorizer and feature arrays
    feturized_data = featurize(data)
    # save featurized data
    save_featurized_data(output_dir, feturized_data)


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
            "e.g. '--months 6 8' (default: [6, 8])"
        ),
        default=['6', '8'],
        required=False,
    )

    args = parser.parse_args()

    prepare_args = vars(args) | {
        'kaggle_id': os.getenv(
            'KAGGLE_DATASET_ID', 'stoney71/new-york-city-transport-statistics'
        )
    }

    prepare(**prepare_args, nrows=150000)

    sys.exit(0)
