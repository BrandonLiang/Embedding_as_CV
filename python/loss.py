import torch.nn as nn

def cross_entropy_loss(pred, label):
  cel = nn.CrossEntropyLoss()
  return cel(pred, label)
