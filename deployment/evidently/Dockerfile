FROM python:3.8-slim-buster

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "app.py", "./" ]
COPY [ "datasets/*", "datasets/" ]

EXPOSE 8085

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8085", "app:app" ]
