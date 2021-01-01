import torch
import torch.nn as nn
import torch.utils.data as data_module
import torch.optim as optim
import os
from tqdm import tqdm
import argparse
import numpy as np
import random
import logging

import loss as loss_module
import model as model_module
import dataset as dataset_module
from parse_config import ConfigParser
import metrics as metrics_module
from trainer import MLMTrainer

def main(config):
  # skips logging INFO from transformers package
  logging.getLogger("transformers").setLevel(logging.CRITICAL)

  logger = config.get_logger(config["logging_verbosity"])
  SEED = config["seed"]
  np.random.seed(SEED)
  random.seed(SEED)

  # verify that current logger level is INFO (1), not DEBUG (2) or WARNING (0)
  #logger.debug(f'DEBUG')
  #logger.info(f'Using {SEED} as seed')
  #logger.warning(f'WARNING')

  # obj - use config.init_obj
  model = config.init_obj('model', model_module)
  #logger.info(model)

  # function - use getattr
  criterion = getattr(loss_module, config['loss'])

  # function - use getattr
  metric_ftns = [getattr(metrics_module, met) for met in config['metrics']]

  trainable_params = filter(lambda p: p.requires_grad, model.parameters())
  # obj - use config.init_obj
  optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

  # obj - use config.init_obj
  lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

  # obj - use config.init_obj
  dataset = config.init_obj('dataset', dataset_module, logger)

  # obj - use config.init_bj
  train_loader = config.init_obj('dataloader', data_module, dataset)

  # (train_loader, val_loader)
  dataloaders = (train_loader, None)

  trainer = MLMTrainer(dataloaders, model, criterion, metric_ftns, optimizer, logger, config, lr_scheduler)

  trainer.train()

if __name__ == "__main__":
  args = argparse.ArgumentParser()
  args.add_argument('-c', '--config', default = None, required = True, type = str, help = 'config file path')
  args.add_argument('-r', '--resume', default = None, required = False, type = str, help = 'path to latest checkpoint (default: None)')
  args.add_argument('-d', '--device', default = None, required = False, type = str, help = 'indices of GPUs to enable (default: all)')
  config = ConfigParser.from_args(args)
  main(config)
