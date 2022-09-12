.PHONY: init test_unit format lint minikube_start minikube_stop build_custom_images deploy_storage deploy_train deploy_service

init:
	pipenv install --dev
	pre-commit install

test_unit:
	python -m unittest discover -s tests/unit -v

format:
	black .
	isort .

lint:
	pylint --exit-zero --recursive=y .

minikube_start:
	@if [ -z "$$(ps aux | grep minikube | grep -v grep)" ]; then \
		minikube start; \
		sleep 20; \
	fi
	@eval $$(minikube -p minikube docker-env)

minikube_stop:
	minikube stop

build_custom_images: minikube_start
	@eval $$(minikube -p minikube docker-env); \
	docker-compose -f docker-compose.yaml build

deploy_storage: minikube_start
	cat deployment/minio/minio.k8.yaml | envsubst | kubectl apply -f -
	sleep 20

deploy_train: deploy_storage build_custom_images
	$(eval ENDPOINT_URL=$(shell minikube service minio-service --url))
	@if [ -z "$$(aws --endpoint-url=$(ENDPOINT_URL) s3 ls | grep ${MLFLOW_S3_BUCKET_NAME})" ]; then \
		aws --endpoint-url=${ENDPOINT_URL} s3 mb s3://${MLFLOW_S3_BUCKET_NAME}; \
	fi
	@if [ -z "$$(aws --endpoint-url=$(ENDPOINT_URL) s3 ls | grep ${PREFECT_S3_BUCKET_NAME})" ]; then \
		aws --endpoint-url=${ENDPOINT_URL} s3 mb s3://${PREFECT_S3_BUCKET_NAME}; \
	fi
	kubectl apply -f deployment/mlflow/postgres.k8.yaml
	sleep 5
	cat deployment/mlflow/server.k8.yaml | envsubst | kubectl apply -f -
	kubectl apply -f deployment/prefect/orion.k8.yaml
	sleep 5
	kubectl apply -f deployment/prefect/agent.k8.yaml
	sleep 5

deploy_service: build_custom_images deploy_train
	cat deployment/prediction_service/*.yaml | envsubst | kubectl apply -f -
	sleep 5
	cat deployment/mongodb/*.yaml | envsubst | kubectl apply -f -
	sleep 5
	cat deployment/prometheus/*.yaml | envsubst | kubectl apply -f -
	sleep 5
	cat deployment/grafana/*.yaml | envsubst | kubectl apply -f -
	sleep 5
	cat deployment/evidently/*.yaml | envsubst | kubectl apply -f -
	sleep 5

destroy: minikube_start
	kubectl delete -f deployment/ --recursive
