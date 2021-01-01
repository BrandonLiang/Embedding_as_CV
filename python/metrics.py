import torch

def accuracy(yhat, masked_token_ids):
  return (torch.argmax(yhat, dim = 1).view(-1) == masked_token_ids.view(-1)).int().sum()
