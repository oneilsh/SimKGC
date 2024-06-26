.DEFAULT_GOAL := analysis


install-poetry:
	pip3 install poetry

install-deps:
	poetry lock
	poetry install

# make runpod to setup runpod environment
runpod: install-poetry install-deps

prepare:
	./scripts/preprocess_kghub.sh

train:
	./scripts/train_kghub.sh


# make model to run 
model: prepare train

# clean up checkpoints folder
clean:
	rm -rf checkpoint/kg_hub/*

