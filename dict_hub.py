import os
import glob

from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph
from logger_config import logger

# Global variables

# TripletDicts contain dictionaries of the form {(head_id, relation): {tail_id1, tail_id2, ...}}
train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None

# LinkGraphs are built as a dictionary where each key is a tuple (head_id, relation) and the value is a set of tail ids.
# These are used along with the entity dictionary represent the graph structure.
# EntityDicts store information about entities, such as their id, name, and description, and can be queried by id, index, or value (to get the id or index)
link_graph: LinkGraph = None
entity_dict: EntityDict = None

# Tokenizer for the model
# The tokenizer is built from the pretrained model specified in the config
tokenizer: AutoTokenizer = None


def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(entity_dict.entity_exs)


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))


def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    _init_link_graph()
    return link_graph


def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer
