#!/usr/bin/env bash

set -x
set -e

model_path=""
# if the first param is set, use that at the model path, otherwise use the latest in checkpoint/kg_hub/*/best_model.pt
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi
if [[ -z "${model_path}" ]]; then
    model_path=$(ls -t checkpoint/kg_hub/*/best_model.pt | head -n 1)
fi

# task is going to take the form of "kg_hub/mondo_kgx_tsv.tar.gz"
task="kg_hub/mondo_kgx_tsv.tar.gz"


DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

test_path="${DATA_DIR}/small.txt.json"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    test_path=$1
    shift
fi

neighbor_weight=0.05
rerank_n_hop=2
if [ "${task}" = "WN18RR" ]; then
# WordNet is a sparse graph, use more neighbors for re-rank
  rerank_n_hop=5
fi
if [ "${task}" = "wiki5m_ind" ]; then
# for inductive setting of wiki5m, test nodes never appear in the training set
  neighbor_weight=0.0
fi

poetry run python3 -u predict.py \
--task "${task}" \
--is-test \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${test_path}" "$@"
