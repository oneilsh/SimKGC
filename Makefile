.DEFAULT_GOAL := analysis


install-poetry:
	pip3 install poetry

install-deps:
	poetry lock
	poetry install

# make runpod to setup runpod environment
install: install-poetry install-deps

prepare:
	./scripts/preprocess_kghub.sh

train-model:
	./scripts/train_kghub.sh


# make model to run 
train: prepare train-model

# clean up checkpoints folder
clean:
	rm -rf checkpoint/kg_hub/*

