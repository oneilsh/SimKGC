.DEFAULT_GOAL := analysis

KG_URL=https://kghub.io/kg-obo/mondo/2024-03-04/mondo_kgx_tsv.tar.gz
KG_BASENAME=$(notdir $(KG_URL))

install-poetry:
	pip3 install poetry

install-deps:
	poetry lock
	poetry install

# make runpod to setup runpod environment
install: install-poetry install-deps

data/kg_hub/$(KG_BASENAME)/entities.json:
	./scripts/preprocess_kghub.sh \
	 --graph $(KG_URL) \
	 --train 0.8 \
	 --valid 0.1 \
	 --test 0.1 \
	 --seed 42

prepare: data/kg_hub/$(KG_BASENAME)/entities.json

train-model:
	poetry run python3 -u trainer.py \
	--model-dir checkpoint/kg_hub/$(KG_BASENAME) \
	--pretrained-model bert-base-uncased \
	--pooling mean \
	--lr 5e-5 \
	--use-link-graph \
	--train-path data/kg_hub/$(KG_BASENAME)/train.txt.json \
	--valid-path data/kg_hub/$(KG_BASENAME)/valid.txt.json \
	--task kg_hub/mondo_kgx_tsv.tar.gz \
	--batch-size 128 \
	--print-freq 20 \
	--additive-margin 0.02 \
	--use-amp \
	--use-self-negative \
	--pre-batch 0 \
	--finetune-t \
	--epochs 50 \
	--workers 4 \
	--max-to-keep 3

# make model to run 
train: prepare train-model

# clean up checkpoints folder
clean:
	rm -rf checkpoint/kg_hub/*


# run some predictions - needs work
embed-entities: prepare
	poetry run python3 -u predict.py \
		--task kg_hub/$(KG_BASENAME) \
		--is-test \
		--eval-model-path checkpoint/kg_hub/$(KG_BASENAME)/model_best.mdl \
		--neighbor-weight 0.05 \
		--rerank-n-hop 2 \
		--entities-json data/kg_hub/$(KG_BASENAME)/entities.json \
		--train-path data/kg_hub/$(KG_BASENAME)/train.txt.json \
		--valid-path data/kg_hub/$(KG_BASENAME)/valid.txt.json \

# not sure why train_path and valid-path is needed here, maybe due to the link graph being used by default to generate entity descriptions if they are short?

#		--eval-model-path data/mondo_1epoch.mdl \

push-huggingface:
	poetry run python3 -u model_huggingface.py \
	--pretrained-model checkpoint/kg_hub/$(KG_BASENAME)/model_best.mdl
	--eval-model-path checkpoint/kg_hub/$(KG_BASENAME)/model_best.mdl \
