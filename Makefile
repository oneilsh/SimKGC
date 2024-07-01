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


# run some predictions - needs work
embed-entities: prepare
	poetry run python3 -u predict.py \
		--task kg_hub/mondo_kgx_tsv.tar.gz \
		--is-test \
		--eval-model-path data/mondo_1epoch.mdl \
		--neighbor-weight 0.05 \
		--rerank-n-hop 2 \
		--entities-json data/kg_hub/mondo_kgx_tsv.tar.gz/entities_small.json \
        --valid-path /Users/oneilsh/Documents/projects/tislab/llm_tests/SimKGC/data/kg_hub/mondo_kgx_tsv.tar.gz/small.txt.json
