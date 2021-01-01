import torch
from tqdm import tqdm
import os

from base_trainer import BaseTrainer
from visualize import visualize
import metrics

CUR_DIR = os.getcwd()

def MLM_load_batch(batch, config):
  original_labels, masked_labels, masked_token_ids, masked_indices = batch
  if config["use_gpu"]:
    original_labels = original_labels.cuda(config["main_gpu"])
    masked_labels = masked_labels.cuda(config["main_gpu"])
    masked_token_ids = masked_token_ids.cuda(config["main_gpu"])
    masked_indices = masked_indices.cuda(config["main_gpu"])
  return original_labels, masked_labels, masked_token_ids, masked_indices

class MLMTrainer(BaseTrainer):
  def __init__(self, dataloders, model, criterion, metric_ftns, optimizer, logger, config, lr_scheduler = None):
    super().__init__(dataloders, model, criterion, metric_ftns, optimizer, logger, config, lr_scheduler = None)
    with torch.no_grad():
      # visualize embedding for a sample word by BERT before training and save as image
      visualize(self.model.bert, epoch = 0, location = CUR_DIR + "/../image")

  def _train_epoch(self, epoch):
    """
    Training logic for an epoch
    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    """
    self.model.train()
    self.train_metrics.reset()
    for batch_idx, batch in tqdm(enumerate(self.train_loader)):

      original_labels, masked_labels, masked_token_ids, masked_indices = MLM_load_batch(batch, self.config)
      actual_batch_size = original_labels.shape[0]

      self.optimizer.zero_grad()

      # https://discuss.pytorch.org/t/selecting-element-on-dimension-from-list-of-indexes/36319/2
      yhat = self.model(masked_labels) # (actual_batch_size, seq_length, vocab_size)
      yhat = yhat[torch.arange(actual_batch_size), masked_indices] # (actual_batch_size, vocab_size)
      
      loss = self.criterion(yhat, masked_token_ids)
      self.writer.add_scalar("Loss/train", loss, self.train_iter_global)
      acc = metrics.accuracy(yhat, masked_token_ids)
      self.writer.add_scalar("Accuracy/train", acc * 1.0 / actual_batch_size, self.train_iter_global)
      self.train_iter_global += 1

      self.train_metrics.update("loss", loss.item())
      for met in self.metric_ftns:
        self.train_metrics.update(met.__name__, met(yhat, masked_token_ids))

      if batch_idx % self.log_step == 0:
        self.logger.info('Train Epoch: {}-{} Loss: {:.6f}'.format(
          epoch,
          self._progress(batch_idx, self.train_loader),
          loss.item()))

      loss.backward()
      self.optimizer.step()

    log = self.train_metrics.result()

    if self.config.do_validation:
      val_log = self._val_epoch(epoch)
      log.update(**{'val_'+k : v for k, v in val_log.items()})

    self.writer.flush()

    with torch.no_grad():
      # visualize embedding for a sample word by BERT as of this epoch and save as image
      visualize(self.model.bert, epoch = epoch, location = CUR_DIR + "/../image")

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()
    return log

  def _val_epoch(self, epoch):
    """
    Validate after training an epoch
    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    """
    self.model.eval()
    self.val_metrics.reset()

    with torch.no_grad():
      for batch_idx, batch in enumerate(self.val_loader):
        original_labels, masked_labels, masked_token_ids, masked_indices = MLM_load_batch(batch, self.config)
        actual_batch_size = original_labels.shape[0]

        # https://discuss.pytorch.org/t/selecting-element-on-dimension-from-list-of-indexes/36319/2
        yhat = self.model(masked_labels) # (actual_batch_size, seq_length, vocab_size)
        yhat = yhat[torch.arange(actual_batch_size), masked_indices] # (actual_batch_size, vocab_size)

        loss = self.criterion(yhat, masked_token_ids)
        writer.add_scalar("Loss/val", loss, self.val_iter_global)
        acc = metrics.accuracy(yhat, masked_token_ids)
        self.writer.add_scalar("Accuracy/val", acc * 1.0 / actual_batch_size, self.val_iter_global)
        self.val_iter_global += 1

        self.val_metrics.update('loss', loss.item())
        for met in self.metric_ftns:
          self.val_metrics.update(met.__name__, met(yhat, masked_token_ids))

        if batch_idx % self.log_step == 0:
          self.logger.info('Val Epoch: {}-{} Loss: {:.6f}'.format(
            epoch,
            self._progress(batch_idx, self.val_loader),
            loss.item()))

    return self.val_metrics.result()


  def _progress(self, batch_idx, dataloader):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(dataloader, 'n_samples'):
        current = batch_idx * dataloader.batch_size
        total = dataloader.n_samples
    else:
        current = batch_idx
        #total = self.len_epoch
        total = len(dataloader)
    return base.format(current, total, 100.0 * current / total)
