#!/usr/bin/env bash

cd "$(dirname "$0")"

export PROJECT_DIR="$( cd .. && pwd )"
export NYC_BUS_DATASET_KAGGLE_ID="stoney71/new-york-city-transport-statistics"

# download dataset into data/ folder
pipenv run kaggle datasets download $NYC_BUS_DATASET_KAGGLE_ID -p $PROJECT_DIR/data --unzip

exit 0
