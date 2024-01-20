# relations.json structure:
# {
#     "/location/country/form_of_government": "form of government country location ",
#     "/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor": "actor regular tv appearance regular cast tv program tv ",
#     "/media_common/netflix_genre/titles": "titles netflix genre media common ",
#     "/award/award_winner/awards_won./award/award_honor/award_winner": "award honor awards won award winner award ",
#     ...
# the input tsv will have the following columns, as specified by the KGX format: subject, predicate, object
# we will only use the predicate column, and if the predicate is of the form "biolink:snake_case", then we will map it to just "snake case" in the relations.json file
# we will also only keep the first occurrence of each predicate, since we don't need to repeat it

import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_tsv', required=True, help='input TSV file of KG edges')
parser.add_argument('--output_json', required=True, help='output JSON file of relations')
args = parser.parse_args()

input_tsv = args.input_tsv
output_json = args.output_json

# first, determine which columns are in the TSV file
# we will use the first line of the TSV file to determine this

with open(input_tsv, 'r') as f:
    first_line = f.readline()
    first_line = first_line.strip()
    columns = first_line.split('\t')
    print(columns)

# now, read in the TSV file and create a list of relations
# don't forget to skip the first line of the TSV file, since it contains the column names
    
relations = {}

with open(input_tsv, 'r') as f:
    first_line = f.readline()
    for line in f:
        line = line.strip()
        values = line.split('\t')
        relation = values[columns.index('predicate')]
        original_relation = relation
        if relation.startswith('biolink:'):
            relation = relation[len('biolink:'):]

        # replace underscores with spaces        
        if '_' in relation:
            relation = relation.replace('_', ' ')
        
        # replace camel case with spaces
        # e.g. "hasPart" -> "has part", "hasPartOf" -> "has part of", "HasPart" -> "has part"
        relation = ''.join([' ' + char.lower() if char.isupper() else char for char in relation]).strip()
        
        if relation not in relations:
            relations[original_relation] = relation

# finally, write out the relations to a JSON file, using a nice format for browsing
print(relations)
with open(output_json, 'w') as f:
    json.dump(relations, f, indent=2)
