#!/usr/bin/env python3
import argparse
import json
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_tsv', required=True, help='input TSV file of KG edges')
parser.add_argument('--output_dir', required=True, help='output directory for JSON files')
parser.add_argument('--relations_json', required=True, help='relations JSON file')
parser.add_argument('--entities_json', required=True, help='entities JSON file')
parser.add_argument('--train', type=float, default=0.8, help='train split')
parser.add_argument('--valid', type=float, default=0.1, help='valid split')
parser.add_argument('--test', type=float, default=0.1, help='test split')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

input_tsv = args.input_tsv
output_dir = args.output_dir
relations_json = args.relations_json
entities_json = args.entities_json
train = args.train
valid = args.valid
test = args.test
seed = args.seed

os.makedirs(output_dir, exist_ok=True)

# Read header to get column indices
with open(input_tsv, 'r') as f:
    header = f.readline().rstrip('\n')
columns = header.split('\t')
try:
    subj_idx = columns.index('subject')
    pred_idx = columns.index('predicate')
    obj_idx = columns.index('object')
except ValueError as e:
    raise RuntimeError(f"Required column missing in TSV header: {e}")

# Load relations mapping
with open(relations_json, 'r') as f:
    relations = json.load(f)

# Load entities and build id -> (entity, entity_desc) dict
with open(entities_json, 'r') as f:
    entities = json.load(f)

entities_dict = {
    e['entity_id']: {
        'entity': e['entity'],
        'entity_desc': e.get('entity_desc')
    }
    for e in entities
}

random.seed(seed)

train_edges = []
valid_edges = []
test_edges = []

# Process TSV rows, build edges in memory
with open(input_tsv, 'r') as f:
    _ = f.readline()  # skip header already read
    for line in f:
        line = line.rstrip('\n')
        if not line:
            continue
        values = line.split('\t')

        subject_id = values[subj_idx]
        predicate_id = values[pred_idx]
        object_id = values[obj_idx]

        # Map subject and object to readable names
        subject = entities_dict.get(subject_id, {}).get('entity', subject_id)
        obj = entities_dict.get(object_id, {}).get('entity', object_id)

        # Map predicate
        predicate = relations.get(predicate_id, predicate_id)

        edge = {
            'head_id': subject_id,
            'head': subject,
            'relation': predicate,
            'tail_id': object_id,
            'tail': obj
        }

        r = random.random()
        if r < train:
            train_edges.append(edge)
        elif r < train + valid:
            valid_edges.append(edge)
        else:
            test_edges.append(edge)

# Write each split once
with open(os.path.join(output_dir, 'train.txt.json'), 'w') as f:
    json.dump(train_edges, f)

with open(os.path.join(output_dir, 'valid.txt.json'), 'w') as f:
    json.dump(valid_edges, f)

with open(os.path.join(output_dir, 'test.txt.json'), 'w') as f:
    json.dump(test_edges, f)

print('Done.')