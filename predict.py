import os
import json
import tqdm
import torch
import torch.utils.data

from typing import List
from collections import OrderedDict

from doc import collate, Example, Dataset
from config import args
from models import build_model
from utils import AttrDict, move_to_cuda
from dict_hub import build_tokenizer
from logger_config import logger

class BertPredictor:

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False

    def load(self, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path)
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))

    def _setup_args(self):
        # pull args into self.train_args, but don't override ones specified in the model checkpoint
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        logger.info('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        # if the checkpoint used the link graph, we should use it as well
        args.use_link_graph = self.train_args.use_link_graph
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=1,
            batch_size=max(args.batch_size, 512),
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list = [], []
        for idx, batch_dict in enumerate(data_loader):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)

if __name__ == '__main__':
    from dict_hub import entity_dict
    
    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)
    print(entity_tensor)

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
   