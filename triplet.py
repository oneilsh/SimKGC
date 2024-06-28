import os
import json

from typing import List
from dataclasses import dataclass
from collections import deque

from logger_config import logger


@dataclass
class EntityExample:
    # example:
    # "entity_id": "MONDO:0002974",
    # "entity": "cervical cancer",
    # "entity_desc": "A primary or metastatic malignant neoplasm involving the cervix."

    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:
    """TripeletDict.hr2tails: {(head_id, relation): {tail_id1, tail_id2, ...}}
       Initialize with TripletDict(path_list=["file1.json", "file2.json", ...])
       
       Where file1.json looks like:
       [ {"head_id": "HGNC:6483", "head": "LAMA3", "relation": "subclass of", "tail_id": "SO:0000704", "tail": "gene"}, ... ]"""

    def __init__(self, path_list: List[str]):
        """path_list: list of paths to the triplet files. Each file should be a list of dictionaries,
        where each dictionary contains 'head_id', 'head', 'relation', 'tail_id', 'tail' keys.
        
        Example:
        [{"head_id": "HGNC:6483", "head": "LAMA3", "relation": "subclass of", "tail_id": "SO:0000704", "tail": "gene"}, ...]

        The dictionary is built as a dictionary where each key is a tuple (head_id, relation) and the value is a set of tail ids.
        These are used along with the entity dictionary represent the graph structure.
        """

        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        """Load triplets from a file and build hr2tails dictionary.
        hr2tails: {(head_id, relation): {tail_id1, tail_id2, ...}}"""
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_triplet(obj) for obj in examples]
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(ex['tail_id'])
        self.triplet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str) -> set:
        """Get neighbors of a head entity given a relation."""
        return self.hr2tails.get((h, r), set())


class EntityDict:
    """EntityDict.entity_exs is a list of EntityExample objects.
       Allows lookup of entities by id, index, or value (EntityExample object).
       
       EntityExample is a dataclass with fields entity_id, entity, entity_desc.
       
       Initialize as EntityDict(entity_dict_dir=<directory containing 'entities.json' file>) OR

       
       Example of 'entities.json': [{"entity_id": "MONDO:0002974", "entity": "cervical cancer", "entity_desc": "A primary or metastatic malignant neoplasm involving the cervix."}, ...]"""

    def __init__(self, entity_dict_dir: str = None, entity_dict_json: str = None, inductive_test_path: str = None):
        """entity_dict_dir: directory containing 'entities.json' file describing entities.
        Each entity should have 'entity_id', 'entity', 'entity_desc' keys.
        
        Example:
        [{"entity_id": "MONDO:0002974", "entity": "cervical cancer", 
          "entity_desc": "A primary or metastatic malignant neoplasm involving the cervix."}, ...]
        
        inductive_test_path: path to a file containing edges (head_id, tail_id) for validating or 
        testing target prediction. If provided, only entities mentioned in the file (as a head or tail) will be loaded."""

        if entity_dict_json:
            path = entity_dict_json
        else:
            path = os.path.join(entity_dict_dir, 'entities.json')

        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]


        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        # build index
        # entity_id -> entity_example
        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        """Get the index of an entity given its id from the dictionary."""
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        """Get an entity given its id from the dictionary."""
        return self.id2entity[entity_id]


    def get_entity_by_idx(self, idx: int) -> EntityExample:
        """Get an entity given its index from the dictionary."""
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


class LinkGraph:
    """LinkGraph.graph looks like {"HGNC:6483": {"SO:0000704", ...}, "SO:0000704": {"HGNC:6483", ...}, ...}
    
       Initialize with LinkGraph(train_path="file.json")
       
       Where 'file.json' looks like:
       [{"head_id": "HGNC:6483", "head": "LAMA3", "relation": "subclass of", "tail_id": "SO:0000704", "tail": "gene"}, ...]"""

    def __init__(self, train_path: str):
        """train_path: path to a file containing triplets.
        Each triplet should have 'head_id', 'head', 'relation', 'tail_id', 'tail' keys.
        
        Example:
        [{"head_id": "HGNC:6483", "head": "LAMA3", "relation": "subclass of", "tail_id": "SO:0000704", "tail": "gene"}, ...]
        
        The graph is built as a dictionary where each key is an entity id and the value is a set of neighbor entity ids. Example:
        {
            "HGNC:6483": {"SO:0000704", ...},
            "SO:0000704": {"HGNC:6483", ...},
            ...
        }
        """
        logger.info('Start to build link graph from {}'.format(train_path))
        # id -> set(id)
        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)
        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
        """Get neighbors of a given entity id. Return at most max_to_keep neighbors.
        If the number of neighbors exceeds max_to_keep, return the first max_to_keep neighbors."""
        # make sure different calls return the same results
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        """Get entities within n hops of the given entity id. Return at most max_nodes entities.
        If the number of entities exceeds max_nodes, return an empty set.
        
        Args:
        entity_id: the id of the entity to start with.
        entity_dict: the entity dictionary containing entity examples; used to convert entity id to index.
        n_hop: the number of hops to search.
        max_nodes: the maximum number of entities to return.
        
        Returns:
        a set of entity *indices* in the provided entity_dict within n hops of the given entity id."""

        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


def reverse_triplet(obj):
    """Reverse a triplet object.
    Example:
    {"head_id": "HGNC:6483", "head": "LAMA3", "relation": "subclass of", "tail_id": "SO:0000704", "tail": "gene"}
    =>
    {"head_id": "SO:0000704", "head": "gene", "relation": "inverse subclass of", "tail_id": "HGNC:6483", "tail": "LAMA3"}
    """
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head']
    }
