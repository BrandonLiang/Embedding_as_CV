import torch
from abc import abstractmethod
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from utils import MetricTracker

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# branched from https://raw.githubusercontent.com/victoresque/pytorch-template/master/base/base_trainer.py


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, dataloaders, model, criterion, metric_ftns, optimizer, logger, config, lr_scheduler = None):
        self.config = config
        self.logger = logger

        self.train_loader, self.val_loader = dataloaders
        self.model = model
        if self.config["use_gpu"]:
          self.model = self.model.cuda(self.config["main_gpu"])
          self.logger.info(f'Using {self.config["main_gpu"]} as the main GPU')
          # also need to handle Distributed GPU training
          # self.logger.info(f'Using GPU ids {} for distributed training')
        else:
          self.logger.info("Running Everything on CPU")
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer.get('save_period', 1)
        self.log_step = cfg_trainer.get('log_step', 1)
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = os.path.join(os.path.dirname(CUR_DIR), cfg_trainer["save_dir"])
        self.logger.info(f'Saving checkpoint to {self.checkpoint_dir}')

        # setup visualization writer instance                
        self.writer = SummaryWriter(os.path.join("{}/{}".format(os.path.dirname(CUR_DIR), config["tb_dir"]), "{}_{}".format(config["name"], config["dataset"]["args"]["n_samples"])))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer = self.writer)
        self.val_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer = self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.train_iter_global = 0
        self.val_iter_global = 0

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        :return: A log that contains average loss and metric in this epoch
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_type = type(self.model).__name__
        state = {
            'model': model_type,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir + '/checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir + '/model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
