import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_tsv', required=True, help='input TSV file of KG nodes')
parser.add_argument('--output_json', required=True, help='output JSON file of entities')
parser.add_argument('--entity_id_column', help='column name for entity id', default='id')
parser.add_argument('--entity_name_column', help='column name for entity name; defaults to category if not specified to conform to KGX format, but should be overridden, likely with "name"', default='category')
parser.add_argument('--entity_desc_column', help='column name for entity description; defaults to category if not specified to conform to KGX format, but should be overridden, likely with "description"', default='category')
args = parser.parse_args()

input_tsv = args.input_tsv
output_json = args.output_json
entity_id_column = args.entity_id_column
entity_name_column = args.entity_name_column
entity_desc_column = args.entity_desc_column

# first, determine which columns are in the TSV file
# we will use the first line of the TSV file to determine this

with open(input_tsv, 'r') as f:
    first_line = f.readline()
    first_line = first_line.strip()
    columns = first_line.split('\t')
    print(columns)

# now, read in the TSV file and create a list of entities
# each entity will be a dictionary with the following keys:
# entity_id, entity, entity_desc
# don't forget to skip the first line of the TSV file, since it contains the column names
    
entities = []

with open(input_tsv, 'r') as f:
    first_line = f.readline()
    for line in f:
        line = line.strip()
        values = line.split('\t')
        entity_id = values[columns.index(entity_id_column)]
        entity_name = values[columns.index(entity_name_column)]
        entity_desc = values[columns.index(entity_desc_column)]
        entity = {
            'entity_id': entity_id,
            'entity': entity_name,
            'entity_desc': entity_desc
        }
        entities.append(entity)

# finally, write out the entities to a JSON file, using a nice format for browsing
        
with open(output_json, 'w') as f:
    json.dump(entities, f, indent=2)

print('Done.')