import random
import numpy as np
import torch.nn as nn
import torch
import os
from tqdm import tqdm
import wandb
import torch.optim as optim
from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import CosineAnnealingLR


##### Taken from https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder #####
def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


##############################################################################################


class Trainer(ABC):
    def __init__(self, cfg, model, train_dataset, test_dataset):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.best_valid_loss = 1000000
        self.step = 0
        self.epoch = -1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)


    def train(self):
        self.step = 0
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Starting training for {self.model.__class__.__name__} with {total_params} parameters")
        self.model.train()
        self.run_valid_loop()
        for epoch in range(self.cfg.trainer.epochs):
            self.epoch = epoch
            total_loss = 0
            batch_step = 0
            for features in tqdm(self.train_dataset, desc=f'Epoch: {epoch}'):
                self.optimizer.zero_grad()

                loss, outputs, logger_dict = self.train_step(features)
                loss.backward()
                batch_step += 1
                total_loss += loss.item()

                if self.step % 10 == 0:
                    self._write_summary(total_loss / batch_step, logger_dict=logger_dict)

                self.step += 1
                self.optimizer.step()

            self.scheduler.step()
            self.run_valid_loop()


    def save_to_checkpoint(self):
        save_name = f'{self.cfg.model.dir}/model_{self.epoch}.pt'
        torch.save(self.model.state_dict(), save_name)
        torch.save(self.train_dataset, f"{self.cfg.model.dir}/train.pt")
        torch.save(self.test_dataset, f"{self.cfg.model.dir}/test.pt")

    @abstractmethod
    def train_step(self, features):
        pass

    @abstractmethod
    def forward_and_loss(self, features):
        pass

    @abstractmethod
    def run_valid_loop(self):
        pass

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _write_summary(self, loss, logger_dict=None):
        wandb.log({"train/loss": loss}, step=self.step)
        wandb.log({"train/lr": self.get_lr()}, step=self.step)


    def _write_summary_valid(self, loss, logger_dict=None):
        wandb.log({"valid/loss": loss}, step=self.step)

