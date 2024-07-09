from trainers.trainer import Trainer
import torch
import wandb
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainers.trainer import Trainer
from collections import defaultdict
from data.transform import TrainTransform, TestTransform, Test2TrainTransform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import nn


class LinearProbingTrainer(Trainer):

    def __init__(self, cfg, model, train_dataset, test_dataset):
        super(LinearProbingTrainer, self).__init__(cfg, model, train_dataset, test_dataset)
        self.optimizer = optim.Adam(self.model.linear.parameters(), lr=cfg.trainer.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        self.criterion = nn.CrossEntropyLoss()
        self.model.vicreg.load_state_dict(torch.load(cfg.model.vicreg_ckpt))

    def train_step(self, features):
        imgs, labels = features
        outputs = self.model(imgs.to(self.device))
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, labels, None

    def forward_and_loss(self, features):
        with torch.no_grad():
            imgs, labels = features
            outputs = self.model(imgs.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            return loss, outputs, None

    def run_valid_loop(self):
        self.model.eval()

        ## accuracy
        correct = 0
        total = 0
        total_loss = 0

        for features in tqdm(self.test_dataset, desc=f'valid'):
            imgs, labels = features
            # compute valid loss
            self.optimizer.zero_grad()
            loss, outputs, logger_dict = self.forward_and_loss(features)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

        final_accuracy = (correct / total) * 100
        final_loss = total_loss / total

        self._write_summary_valid(final_loss, logger_dict={'acc': final_accuracy})
        # save checkpoint
        torch.save(self.model.state_dict(), self.cfg.model.dir / f"epoch_{self.epoch}.pt")


    def _write_summary_valid(self, loss, logger_dict=None):

        wandb.log({"valid/accuracy": logger_dict['acc']}, step=self.step)
        wandb.log({"valid/loss": loss}, step=self.step)
