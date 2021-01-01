import torch.nn as nn

def cross_entropy_loss(pred, label):
  return nn.CrossEntropyLoss(pred, label)
