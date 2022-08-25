import sys
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from src import preprocessing
from src.utils import dump_pickle


@dataclass
class FeaturizedData:
    dv: DictVectorizer
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


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


@task
def save_featurized_data(
    output_dir: str,
    fd: FeaturizedData,
):
    # save dictvectorizer and dataset splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_pickle(fd.dv, output_dir / "dv.pkl")
    dump_pickle((fd.X_train, fd.y_train), output_dir / "train.pkl")
    dump_pickle((fd.X_val, fd.y_val), output_dir / "validation.pkl")
    dump_pickle((fd.X_test, fd.y_test), output_dir / "test.pkl")


@flow(task_runner=SequentialTaskRunner())
def prepare(input_dir: str, output_dir: str, months: list[int], **kwargs):
    """
    Saves feature and target arrays + dict vectorizer, given input and output data paths.
    """
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

    prepare(**vars(args), nrows=150000)

    sys.exit(0)
