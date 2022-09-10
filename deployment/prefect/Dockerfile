FROM prefecthq/prefect:2.2.0-python3.9

RUN pip install pipenv

WORKDIR /opt/prefect

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy
