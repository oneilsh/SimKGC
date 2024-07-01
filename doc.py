import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()
# the link_graph is used during the reranking process, which is optional but
# improved performance in target prediction by boosting similarity scores
# of nodes nearby the head
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    """Tokenize the text and text_pair if provided. Return the encoded inputs.
    Returns the reult of the BERT tokenizer: a dictionary containing the input_ids, token_type_ids, and attention_mask.
    
    (text_pair refers to BERT's sentence pair input, which is used for tasks like question answering and entailment.)"""
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    """Parse the entity name from the entity string. The wn18rr dataset has entities in the format 'family_alcidae_NN_1',
    so this function removes the last two parts of the entity string to get the name of the entity. For other datasets,
    the entity string is returned as is. If the entity string is None, an empty string is returned."""
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    """Concatenate the entity name and description. If the entity description starts with the entity name, the entity name
    is removed from the description. The entity name and description are concatenated with a colon and space between them."""
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    """Get a string containing the names of the neighbors of the given entity id. The names are separated by spaces.
    If a tail_id is provided, the tail entity is excluded from the list of neighbors."""
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class HRTExample:
    """A class representing a training example. The object holds the head_id, tail_id, and relation, 
    and uses the global entity_dict to get other information about the entities.
    An Example object can be vectorized to get the token ids, token type ids, and masks for the head, tail, and relation.
    """

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        self.hr_embedding = None

    @property
    def head_desc(self):
        """Get the description of the head entity. If the head_id is None, an empty string is returned."""
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        """Get the description of the tail entity. If the tail_id is None, an empty string is returned."""
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        """Get the head entity itself. If the head_id is None, an empty string is returned."""
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        """Get the tail entity itself. If the tail_id is None, an empty string is returned."""
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        """Vectorize the example by tokenizing the head and tail entities and the relation. The head and tail entities
        are tokenized with their descriptions. If the use_link_graph flag is set, descriptions of the head and tail entities
        that are too short (< 20 words) are concatenated with the names of their neighbors, except 
        for the name of the tail."""
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        # tasks-specific fixups
        head_word = _parse_entity_name(self.head)
        # smart concatenation of name and description (removes duplicate name from description if present)
        head_text = _concat_name_desc(head_word, head_desc)

        # the hr will be encoded via BERTs sentence pair input
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        # tails are encoded without sentence pair input, but the name and description are concatenated for better context
        # (and if the description is short, and the use_link_graph flag is set, the neighbor names are added to the description,
        # except for the name of the head entity in this case)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'obj': self}

    def set_hr_embedding(self, hr_embedding: torch.tensor):
        """Set the embedding for the head and relation."""
        self.hr_embedding = hr_embedding


class Dataset(torch.utils.data.dataset.Dataset):
    
    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[HRTExample]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(HRTExample(**obj))
        if add_backward_triplet:
            examples.append(HRTExample(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    """Collate the batch data. The batch data is a list of dictionaries, where each dictionary contains the token ids,
    token type ids, and masks for the head, tail, and relation. The object is also stored in the dictionary. The triplet
    mask and self negative mask are constructed for the batch data."""
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
