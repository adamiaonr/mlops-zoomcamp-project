from kaggle.api.kaggle_api_extended import KaggleApi  # pylint: disable-msg=E0611


def get_kaggle_client() -> KaggleApi:
    # create kaggle client object and authenticate
    # (assumes ~/.kaggle exists OR KAGGLE_USERNAME and KAGGLE_KEY env variables set)
    client = KaggleApi()
    client.authenticate()

    return client
