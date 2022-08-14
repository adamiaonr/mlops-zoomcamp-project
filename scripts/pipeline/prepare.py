import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from pathlib import Path

from src.utils import dump_pickle
from src.preprocessing import load_data, fix_datetime_columns, fix_scheduled_arrival_time_column, add_extra_columns


def featurize(
    data: pd.DataFrame,
    dv: DictVectorizer,
    categorical: list[str],
    numerical: list[str],
    fit_dv: bool = False
) -> tuple[np.ndarray, DictVectorizer]:
    feature_dicts = data[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        X = dv.fit_transform(feature_dicts)
    else:
        X = dv.transform(feature_dicts)

    return X, dv


def prepare(input_dir: str, output_dir: str, months: list[int], **kwargs):
    # load data
    data = load_data(Path(input_dir), months=months, **kwargs)

    # apply dataset 'cleaning' functions
    fix_datetime_columns(data)
    fix_scheduled_arrival_time_column(data)

    # add extra columns for extra features
    add_extra_columns(data)

    # define categorial and numerical features
    categorical = ['BusLine_Direction', 'NextStopPointName']
    numerical = ['TimeOfDayInSeconds', 'DayOfWeek']
    # target variable
    target = 'DelayAtStop'

    # drop nan values in required columns
    data = data.dropna(subset=categorical + numerical + [target]).reset_index(drop = True)

    # split dataset into train, validation and test sets
    train_data, test_data = train_test_split(data, test_size=0.30)
    train_data, validation_data = train_test_split(train_data, test_size=0.20)

    print(f"dataset split sizes :\n\ttrain : {len(train_data)}\n\tvalidation : {len(validation_data)}\n\ttest : {len(test_data)}")

    # featurize splits
    dv = DictVectorizer()
    # - get dict vectorizer, train, validation and test splits
    X_train, dv = featurize(train_data, dv, categorical, numerical, fit_dv=True)
    X_val, _ = featurize(validation_data, dv, categorical, numerical, fit_dv=False)
    X_test, _ = featurize(test_data, dv, categorical, numerical, fit_dv=False)

    # save dictvectorizer and dataset splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_pickle(dv, output_dir / "dv.pkl")
    dump_pickle((X_train, train_data[target].values), output_dir / "train.pkl")
    dump_pickle((X_val, validation_data[target].values), output_dir / "validation.pkl")
    dump_pickle((X_test, test_data[target].values), output_dir / "test.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        help="location of 'raw' NYC bus dataset",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        help="location where processed data will be saved",
        required=True
    )
    
    parser.add_argument(
        "--months",
        nargs='+',
        help="NYC bus dataset months to be used, separated by whitspaces, e.g. '--months 6 8' (default: [6, 8])",
        default=['6', '8'],
        required=False
    )

    args = parser.parse_args()

    prepare(**vars(args), nrows=150000)
