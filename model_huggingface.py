import os
import json
import tqdm
import torch
import torch.utils.data

from typing import List
from collections import OrderedDict

from doc import collate, HRTExample, Dataset
from config import args
from models import build_model
from utils import AttrDict, move_to_cuda
from dict_hub import build_tokenizer
from logger_config import logger
from triplet import EntityDict

class BertSaver:

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False

    def load(self, ckt_path, use_data_parallel=False):
        # predict.py calls with ckt_path
        assert os.path.exists(ckt_path)
        # load the model from the checkpoint, which is a dictionary containing the model state and the args
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        # build the model using the args, this will initially use what is specified in pretrained_model 
        # (defaulting to 'bert-base-uncased', two actually, for hr and t encoders)
        self.model = build_model(self.train_args)

        # if there is a model checkpoint (?), we load in the state dict

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



if __name__ == '__main__':
    saver = BertSaver()
    saver.load(ckt_path=args.eval_model_path)
    saver.model.push_to_hub("sim-kgx-dev")
