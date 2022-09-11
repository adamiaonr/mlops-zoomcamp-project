FROM python:3.8-slim-buster

RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "app.py", "./" ]
COPY [ "datasets/*", "./datasets/" ]

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8085"]
