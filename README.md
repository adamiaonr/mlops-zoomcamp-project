# NYC Bus Delay Prediction Service
A end-to-end(ish) ML pipeline for a simple bus delay prediction service, which can be deployed in a Kubernetes cluster.

Final project for [MLOps Zoomcamp course, 2022 edition](https://github.com/DataTalksClub/mlops-zoomcamp).

## Motivation

The core idea of the project uses ML to provide an estimate of the delay of a New York City bus (NYC), given features such as (a) the bus line / direction of the bus; (b) a station where to catch the bus; (c) the time of the day; (d) day of the week.

The typical usage of the system is:

1. A bus user issues a request towards a 'always-on' bus delay prediction service, requesting a delay estimate for bus line X at stop Y
2. The bus delay prediction service replies with a current estimate of a delay, in minutes
3. The user decides to adjust his/her schedule according to the estimate provided by the service

## Dataset

This project uses the [NYC Bus Data](https://www.kaggle.com/datasets/stoney71/new-york-city-transport-statistics) dataset from Kaggle, with Kaggle ID `stoney71/new-york-city-transport-statistics`.

### Calculation of target variable

The 'ground truth' - i.e., the actual delay at stop Y for line X - is not directly present in the dataset.
To calculate the actual delay - referred to as `DelayAtStop` in our code - we follow this approach:
1. Select all rows for which `ArrivalProximityText`'s value is 'at stop'
2. For the previous selection, set `DelayAtStop` as the difference in seconds between `RecordedAtTime` and `ScheduledArrivalTime`

An initial dataset analysis (and according to the description on Kaggle), NYC buses can be either behind or ahead of schedule, meaning that `DelayAtStop` can be negative.

### Issue with `ScheduledArrivalTime` column

The `ScheduledArrivalTime` column sometimes shows time values such as `24:05:00`, without a date, which is quite annoying...
In order to pass it to a format like `2022-08-01 00:05:00` we apply the function [`fix_scheduled_arrival_time_column()`](https://github.com/adamiaonr/mlops-zoomcamp-project/blob/master/src/preprocessing.py#L36).

## MLOps pipeline

### Overview

* **Cloud:** not developed in the Cloud, but project is deployed to Kubernetes (tested locally on a Minikube cluster)
* **Experiment tracking & model registry:** MLFlow for experiment tracking, with model registering, using a dedicated [MINIO](https://min.io/) S3-compatible storage deployment as artifact store
* **Workflow orchestration:** Basic orchestration of an ML training pipeline workflow, i.e.:
  1. Download of data directly from Kaggle
  2. Running hyper-parameter tuning experiments with MLFlow
  3. selection and registration of best model.

Used a Prefect Kubernetes deployment with (a) a Prefect Orion server; and (b) a Prefect Agent that starts a Kubernetes pod to run the workflow.

* **Model deployment:** Model is deployed in a 'web-service' and 'containerized' fashion. Accompanied with:
  1. Dockerfile and Docker Compose files for image building
  2. Kubernetes `.yaml` files for deployment in a Kubernetes cluster
  3. Commands for deployment in a Kubernetes cluster in a `Makefile`

* **Model monitoring:** Basic model monitoring deployment in Kubernetes that shows data drift in a Grafana dashboard, with backend composed by MongoDB, Prometheus and EvidentlyAI services.

* **Reproducibility:** `Makefile` and instructions for deployment in a local Minikube Kubernetes cluster.

* **Best practices:**
  1. Unit tests
  2. Linter + code formatters used
  3. Makefile
  4. Pre-commit hooks

### Requirements

To run this project, you must install the following:
* [Minikube](https://minikube.sigs.k8s.io/docs/start/): for testing a local Kubernetes cluster on your PC
* [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/): the Kubernetes command-line tool, used to interact with the Kubernetes cluster
* [aws CLI tools](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html): to interact with S3 storage
* docker-compose: to build custom Docker images
* [Kaggle API](https://www.kaggle.com/docs/api) credentials (i.e., `KAGGLE_USERNAME`, `KAGGLE_KEY`): used to download the dataset
* `.env` filled with missing values

### Usage

#### Setup environment
---

1. Fill in `.env` with any missing variable values (e.g., `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY`)

2. Install dependencies and activate Python virtual environment

```
$ make init
$ pipenv shell
```

3. Open a terminal window and run the following to setup the environment variables:

```
$ cd <repo-base-dir>
$ set -a; source .env;
```

#### Deploy ML training pipeline
---

1. Start Minikube, build custom docker images and deploy all necessary 'pods' for the ML training pipeline in a local Kubernetes cluster:

```
$ make deploy_train
```

In this case, the 'local Kubernetes cluster' is managed by Minikube.
The instructions to deploy each 'pod' are enclosed in special `.yaml` files, within the [`deployment/` directory](https://github.com/adamiaonr/mlops-zoomcamp-project/tree/master/deployment).
You can also inspect the `deploy_train` target in the `Makefile` to understand the `kubectl` commands that trigger the deployment on the Kubernetes cluster.

**NOTE:** if the process fails, it may be necessary to re-run `make deploy_train` several times until it works.

2. Verify that all 'pods' are working by running `kubectl get pods`.
The `STATUS` fields should be set to `Running`, e.g:

```
$ kubectl get pods
```

You shold get an output as follows.
```
NAME                                 READY   STATUS    RESTARTS      AGE
minio-7df5d9dc8-w8vvn                1/1     Running   0             14m
mlflow-deployment-5bb6788775-5hhzt   1/1     Running   0             11m
mlflow-postgres-0                    1/1     Running   0             11m
orion-75c87d759d-jcnmf               1/1     Running   0             11m
prefect-agent-59d9f986b4-mjsdj       1/1     Running   1 (11m ago)   11m
```

3. At this stage, it is helpful to add the following entries to your `/etc/hosts` file, so that your PC can directly access Kubernetes services without port-forwarding for some of the services:

```
<kubernetes-ingress-IP>  minio.local bus-delay-prediction.local
```

The value of `<kubernetes-ingress-IP>` can be found by running the command below, and picking the `ADDRESS` field, e.g.:

```
$ kubectl get ingress
NAME                    CLASS   HOSTS         ADDRESS          PORTS   AGE
minio-service-ingress   nginx   minio.local   192.168.59.100   80      5m15s
```

#### Configure Prefect deployment
---

We should now be ready to build our own Prefect deployment, and run it.
To do so, we need to perform a few of manual steps:

1. On a separate shell, tell `kubectl` to port-forward port 4200 to the Prefect Orion service, and let it running:
```
$ kubectl port-forward service/orion 4200:4200
Forwarding from 127.0.0.1:4200 -> 4200
Forwarding from [::1]:4200 -> 4200
```

2. On a browser, go the Prefect UI at `http://localhost:4200`.
We need to create a storage block for our Prefect deployment.
To do so, go to 'Blocks', choose 'Add Block' and then click the 'Add +' button on 'Remote File System'.

Fill in the menu with the following information:

* Block Name: `prefect-flows` (or whatever you have defined under `PREFECT_S3_BUCKET_NAME` in the `.env` file)
* Basepath: `s3://prefect-flows/`
* Settings:

```json
{
  "client_kwargs": {
    "endpoint_url": "http://minio.local"
  }
}
```

Click 'Create'.

3. Create a Prefect deployment. First, create an initial deployment `.yaml` file by running the following on a terminal:

```
$ cd <repo-base-dir>
$ pipenv run prefect deployment build scripts/pipelines/train/main.py:main  --name training-pipeline --infra kubernetes-job --storage-block remote-file-system/prefect-flows --work-queue kubernetes
```

This should create a `main-deployment.yaml` file on the base of the repository.

4. Open the existing `prefect-deployment.yaml` file in the repository for editing.
Replace the values of the `_block_document_name` and `_block_document_id` fields in with those from the `main-deployment.yaml` file.
Then, run the commands:

```
$ cat prefect-deployment.yaml | envsubst > .prefect-deployment.yaml
$ pipenv run prefect deployment apply .prefect-deployment.yaml
```

**Note:** this is very sub-optimal... but unfortunately I couldn't find a more automated way to create a Prefect deployment.

5. Go to the Prefect UI again (`http://localhost:4200`), choose 'Deployments': you should see a new deployment called 'training-pipeline'.
To run it, you can select 'Run' in the top-right corner, and choose 'Now with defaults'.
You can now use the Prefect UI to monitor the execution of the Prefect flow.

#### Prediction service deployment
---

A the end of the ML training pipeline, we should now have a model staged to 'Production'.
We can now deploy the NYC bus delay prediction service!

1. Confirm the best model is staged to 'Production' in the MLFlow UI. To do so, in a separate shell, run the following port-forwarding command:

```
$ kubectl port-forward service/mlflow-service 5000:5000
```

You can now access the MLFlow UI at `http://localhost:5000`.

2. Deploy the service and monitoring infrastructure by running:

```
make deploy_service
```

3. Test the bus prediction service. On a terminal, run:

```
$ cd <repo-base-dir>
$ pipenv run python deployment/prediction_service/test.py
```

You should get an output as follows, indicating that the service is running:

```
$ pipenv run python deployment/prediction_service/test.py

QUERY: {'PublishedLineName': 'S40', 'DirectionRef': 0, 'NextStopPointName': 'SOUTH AV/ARLINGTON PL', 'TimeOfDayInSeconds': 76528, 'DayOfWeek': 4}
RESPONSE: {'bus delay': 185.15206909179688, 'model_version': '51b72630cb2a4ac09432900a8803a946'}
```

4. To access the data drift dashboards in Grafana, run the following command to port-forward port 3000 on a separate shell:

```
kubectl port-forward service/grafana-service 3000:3000
```

After more than 5 test runs, the drift values should be visible in `http://localhost:3000`.
