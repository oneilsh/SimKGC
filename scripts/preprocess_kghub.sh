#!/bin/bash

set -x
set -e

# defaults:
GRAPH="https://kg-hub.berkeleybop.io/kg-obo/mfoem/2022-07-19/mfoem_kgx_tsv.tar.gz"
TRAIN_PERCENTAGE=0.8
VALID_PERCENTAGE=0.1
TEST_PERCENTAGE=0.1
SEED=42

# parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -g|--graph)
            GRAPH="$2"
            shift
            shift
            ;;
        -t|--train)
            TRAIN_PERCENTAGE="$2"
            shift
            shift
            ;;
        -v|--valid)
            VALID_PERCENTAGE="$2"
            shift
            shift
            ;;
        -e|--test)
            TEST_PERCENTAGE="$2"
            shift
            shift
            ;;
        -s|--seed)
            SEED="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

GRAPH_BASE="kg_hub/$(basename "${GRAPH}")"

# if ./data/GRAPH_BASE doesn't exist, create it
if [ ! -d "./data/${GRAPH_BASE}" ]; then
    mkdir -p "./data/${GRAPH_BASE}"
fi

# if the graph doesn't exist, download it
if [ ! -f "./data/${GRAPH_BASE}/graph.tsv" ]; then
    wget "${GRAPH}" -O "./data/${GRAPH_BASE}/graph.tar.gz"
    tar -xvzf "./data/${GRAPH_BASE}/graph.tar.gz" -C "./data/${GRAPH_BASE}"
fi

# rename *_nodes.tsv to nodes.tsv and *_edges.tsv to edges.tsv
mv "./data/${GRAPH_BASE}"/*_nodes.tsv "./data/${GRAPH_BASE}/nodes.tsv"
mv "./data/${GRAPH_BASE}"/*_edges.tsv "./data/${GRAPH_BASE}/edges.tsv"

# now we can run kgnodes_tsv_to_entities_json.py on the nodes.tsv file
python3 -u kgnodes_tsv_to_entities_json.py \
    --input_tsv "./data/${GRAPH_BASE}/nodes.tsv" \
    --output_json "./data/${GRAPH_BASE}/entities.json" \
    --entity_name_column "name" \
    --entity_desc_column "description"

# and we an run kgedges_tsv_to_relations_json.py on the edges.tsv file
python3 -u kgedges_tsv_to_relations_json.py \
    --input_tsv "./data/${GRAPH_BASE}/edges.tsv" \
    --output_json "./data/${GRAPH_BASE}/relations.json"

# now lets kgedges_tsv_to_train_val_test_json.py on the edges.tsv file using the train, valid, and test percentages, seed, and the entities.json and relations.json files we just created

python3 -u kgedges_tsv_to_train_val_test_json.py \
    --input_tsv "./data/${GRAPH_BASE}/edges.tsv" \
    --output_dir "./data/${GRAPH_BASE}" \
    --relations_json "./data/${GRAPH_BASE}/relations.json" \
    --entities_json "./data/${GRAPH_BASE}/entities.json" \
    --train "${TRAIN_PERCENTAGE}" \
    --valid "${VALID_PERCENTAGE}" \
    --test "${TEST_PERCENTAGE}" \
    --seed "${SEED}"


exit 0

