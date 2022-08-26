from pathlib import Path
from dataclasses import dataclass

import numpy as np
from prefect import task
from sklearn.feature_extraction import DictVectorizer

from src.utils import dump_pickle, load_pickle


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
def load_featurized_data(input_dir: str) -> FeaturizedData:
    input_dir = Path(input_dir)

    dv = load_pickle(input_dir / "dv.pkl")
    X_train, y_train = load_pickle(input_dir / "train.pkl")
    X_val, y_val = load_pickle(input_dir / "validation.pkl")
    X_test, y_test = load_pickle(input_dir / "test.pkl")

    return FeaturizedData(dv, X_train, y_train, X_val, y_val, X_test, y_test)


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
