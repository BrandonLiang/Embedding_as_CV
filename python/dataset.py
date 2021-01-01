import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch
import random
from tqdm import tqdm
import copy
import os

CUR_DIR = os.getcwd()

# branched from https://github.com/BrandonLiang/DVMM/blob/master/vcr_label_transformer_and_kmeans/python/dataset.py

random.seed(10)

class ReviewsDataset(Dataset):

  def __init__(self, logger, path = "{}/../data/Reviews.csv".format(CUR_DIR), delimiter = ',', text_field = "Text", max_length = 128, n_samples = -1):
    df = pd.read_csv(path, delimiter = delimiter)
    if n_samples > 0:
        df = df.head(n_samples)
    self.labels = df[text_field].tolist()
    self.masked_token_ids = []
    self.masked_indices = []
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    self.max_length = max_length
    self.size = df.shape[0]
    self.mask_token = self.tokenizer.mask_token
    self.mask_token_id = self.tokenizer.mask_token_id
    self.logger = logger
    self.tokenize_mask()

  def __len__(self):
    return self.size

  def tokenize_mask(self):
    self.labels = [torch.tensor(self.tokenizer.encode(label, max_length = self.max_length)) for label in self.labels]
    self.masked_labels = copy.deepcopy(self.labels)
    for index, label in tqdm(enumerate(self.masked_labels)):
      masked_index = random.randint(1, label.shape[0] - 2) # exclude CLS and SEP from masking
      masked_token_id = copy.deepcopy(label[masked_index]) # need to use copy.deepcopy to avoid change/update in reference
      self.masked_token_ids.append(masked_token_id)
      self.masked_indices.append(masked_index)
      self.masked_labels[index][masked_index] = self.mask_token_id

    # pad
    self.labels = torch.nn.utils.rnn.pad_sequence(self.labels, batch_first = True)
    self.masked_labels = torch.nn.utils.rnn.pad_sequence(self.masked_labels, batch_first = True)

    self.masked_token_ids = torch.tensor(self.masked_token_ids)
    self.masked_indices = torch.tensor(self.masked_indices)

    #self.logger.info("self.labels size: {}".format(self.labels.shape))
    #self.logger.info("self.masked_labels size: {}".format(self.masked_labels.shape))
    #self.logger.info("self.masked_token_ids size: {}".format(self.masked_token_ids.shape))
    #self.logger.info("self.masked_indices size: {}".format(self.masked_indices.shape))

    #self.logger.info(self.labels[0,:])
    #self.logger.info(self.tokenizer.convert_ids_to_tokens(self.labels[0, :].tolist()))
    #self.logger.info(self.tokenizer.convert_ids_to_tokens(self.masked_labels[0].tolist()))



  def __getitem__(self, idx):
    return self.labels[idx], self.masked_labels[idx], self.masked_token_ids[idx], self.masked_indices[idx]
