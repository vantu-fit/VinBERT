login:
	aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws account id>.dkr.ecr.<region>.amazonaws.com
build:
	docker build -t vantufit/flash-attn-cuda -f Dockerfile.base .
	docker build -t vin_bert -f Dockerfile .
tag:
	docker tag vin_bert <aws account id>.dkr.ecr.<region>.amazonaws.com/vin_bert:vin_bert_1
push:
	docker push <aws account id>.dkr.ecr.<region>.amazonaws.com/vin_bert:vin_bert_1

run: build tag push

train:
	python train.py