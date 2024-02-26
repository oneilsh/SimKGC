import torch
import json
import os
import torch.backends.cudnn as cudnn

from config import args
from trainer import Trainer
from logger_config import logger, add_file_handler


def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    # Configure log file path and add file handler
    log_file_path = os.path.join(args.model_dir, 'training.log')
    add_file_handler(log_file_path)

    
    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()

if __name__ == '__main__':
    main()
