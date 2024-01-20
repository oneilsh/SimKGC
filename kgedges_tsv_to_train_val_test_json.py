# This script processes an edges.tsv file and creates three outputs: test.txt.json, train.txt.json, and valid.txt.json. The input edges.tsv file is expected to have the following columns: subject, predicate, object. 
# The output JSON files will have the following format:
# [
#     {
#         "head_id": "/m/027rn",
#         "head": "Dominican Republic",
#         "relation": "form of government country location ",
#         "tail_id": "/m/06cx9",
#         "tail": "Republic"
#     },
#     {
#         "head_id": "/m/017dcd",
#         "head": "Mighty Morphin Power Rangers",
#         "relation": "actor regular tv appearance regular cast tv program tv ",
#         "tail_id": "/m/06v8s0",
#         "tail": "Wendee Lee"
#     },
#    ...

# it also takes in a relations.json file, which is used to map the predicate to a more readable form
# and a entities.json file, which is used to map the subject and object to a more readable form of the entity name

import argparse
import json
import random
import sys
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--input_tsv', required=True, help='input TSV file of KG edges')
parser.add_argument('--output_dir', required=True, help='output directory for JSON files')
parser.add_argument('--relations_json', required=True, help='relations JSON file')
parser.add_argument('--entities_json', required=True, help='entities JSON file')
# add args for train/valid/test split and seed
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


# first, determine which columns are in the TSV file
# we will use the first line of the TSV file to determine this

with open(input_tsv, 'r') as f:
    first_line = f.readline()
    first_line = first_line.strip()
    columns = first_line.split('\t')
    print(columns)

# now, let's read in the relations.json file and entities.json files for mapping while processing the edges

with open(relations_json, 'r') as f:
    relations = json.load(f)
    # this returns a dictionary with keys for the predicate and values for the relation

with open(entities_json, 'r') as f:
    entities = json.load(f)
    # this returns a list of dictionaries with keys for entity_id, entity, and entity_desc
    # we want to create a dictionary that maps entity_id to entity and entity_desc
    # we will use this dictionary to map the subject and object to a more readable form of the entity name
    entities_dict = {}
    for entity in entities:
        entity_id = entity['entity_id']
        entity_name = entity['entity']
        entity_desc = entity['entity_desc']
        entities_dict[entity_id] = {
            'entity': entity_name,
            'entity_desc': entity_desc
        }


# now let's initialize the train/test/valid files with an empty list, ie '['
   
with open(output_dir + '/train.txt.json', 'w') as f:
    f.write('[\n')
with open(output_dir + '/test.txt.json', 'w') as f:
    f.write('[\n')
with open(output_dir + '/valid.txt.json', 'w') as f:
    f.write('[\n')


# now, let's process the input_tsv file one line at a time, using the random seed to determine if each row should be added to train/valid/test splits
# we will also use the relations.json file to map the predicate to a more readable form for the relation field
# and the entities.json file to map the subject and object to a more readable form for the head and tail fields

random.seed(seed)

with open(input_tsv, 'r') as f:
    first_line = f.readline()
    for line in f:
        line = line.strip()
        values = line.split('\t')
        subject_id = values[columns.index('subject')]
        predicate_id = values[columns.index('predicate')]
        object_id = values[columns.index('object')]

        # now lets look up the subject and object in the entities.json file
        # if the subject or object is not found, then we will use the original subject or object
        # we will also use the relations.json file to map the predicate to a more readable form
        # if the predicate is not found, then we will use the original predicate

        if subject_id in entities_dict:
            subject = entities_dict[subject_id]['entity']
        else:
            subject = subject_id

        if object_id in entities_dict:
            object = entities_dict[object_id]['entity']
        else:
            object = object_id

        if predicate_id in relations:
            predicate = relations[predicate_id]
        else:
            predicate = predicate_id

        edge = {
            'head_id': subject_id,
            'head': subject,
            'relation': predicate,
            'tail_id': object_id,
            'tail': object
        }

        # now, let's randomly assign this edge to train/valid/test splits
        # we will use the random seed to do this
        # we will use the train/valid/test split percentages to determine which split to assign to
        # if the random number is less than train, then we will assign to train
        # if the random number is less than train + valid, then we will assign to valid
        # otherwise, we will assign to test
        # we will also use the random seed to shuffle the edges within each split

        random_number = random.random()
        print(random_number, train, valid, test)
        if random_number < train:
            # add to train
            with open(output_dir + '/train.txt.json', 'a') as f:
                f.write(json.dumps(edge) + ',\n')
        elif random_number < train + valid:
            # add to valid
            with open(output_dir + '/valid.txt.json', 'a') as f:
                f.write(json.dumps(edge) + ',\n')
        else:
            # add to test
            with open(output_dir + '/test.txt.json', 'a') as f:
                f.write(json.dumps(edge) + ',\n')

# close up the train/test/valid files with a closing bracket, ie ']'
# this will make the JSON file valid
# we also need to remove the last comma from the last line of the file so far
                
with open(output_dir + '/train.txt.json', 'rb+') as f:
    f.seek(-2, os.SEEK_END)
    f.truncate()
    f.write(b'\n]\n')
with open(output_dir + '/test.txt.json', 'rb+') as f:
    f.seek(-2, os.SEEK_END)
    f.truncate()
    f.write(b'\n]\n')
with open(output_dir + '/valid.txt.json', 'rb+') as f:
    f.seek(-2, os.SEEK_END)
    f.truncate()
    f.write(b'\n]\n')


print('Done.')
