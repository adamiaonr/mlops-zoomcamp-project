FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /mlflow/

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

EXPOSE 5000

ENTRYPOINT ["mlflow", "server"]
