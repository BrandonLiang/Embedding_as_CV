import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
  """
  Base class for all models
  """
  @abstractmethod
  def forward(self, *inputs):
    """
    Forward pass logic
    :return: Model output
    """
    raise NotImplementedError

  def __str__(self):
    """
    Model prints with number of trainable parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return super().__str__() + '\nTrainable parameters: {}'.format(params)

class MLMBertModel(BaseModel):
  def __init__(self, dropout_rate = 0.4):
    super(MLMBertModel, self).__init__()
    self.bert = BertModel.from_pretrained("bert-base-uncased")
    self.bert_config = self.bert.config
    self.bert_dim = self.bert_config.hidden_size
    self.vocab_size = self.bert_config.vocab_size
    self.dropout_rate = dropout_rate
    self.dropout = nn.Dropout(self.dropout_rate)
    self.batchnorm = nn.BatchNorm1d(self.bert_dim)
    self.mlm_fc = nn.Linear(self.bert_dim, self.vocab_size)
    self.ns_fc = nn.Linear(self.bert_dim, 1)

  def forward(self, x):
    x, _ = self.bert(x)
    x = self.dropout(x)
    x = self.batchnorm(x.permute(0, 2, 1)).permute(0, 2, 1)
    word_preds = self.mlm_fc(x[:, 1:, :]) # Exclude Sentence Embedding @ [CLS]
    #next_sent_preds = self.ns_fc(x[:, 0, :]) # Use Sentence Embedding @ [CLS]
    return word_preds #, next_sent_preds
