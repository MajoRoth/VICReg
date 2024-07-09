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

import matplotlib.pyplot as plt



class VICRegTrainer(Trainer):

    def __init__(self, cfg, model, train_dataset, test_dataset):
        super(VICRegTrainer, self).__init__(cfg, model, train_dataset, test_dataset)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.trainer.lr, betas=(0.9, 0.999), weight_decay=cfg.trainer.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        self.gamma = cfg.trainer.gamma
        self.epsilon = cfg.trainer.epsilon
        self.lambdaa = cfg.trainer.lambdaa
        self.mu = cfg.trainer.mu
        self.nu = cfg.trainer.nu

        self.test_train_transform = Test2TrainTransform()


    def train_step(self, features):
        (x1, x2), labels = features

        z1 = self.model(x1.to(self.device))
        z2 = self.model(x2.to(self.device))

        loss, logger_dict = self.loss(z1, z2)
        return loss, None, logger_dict

    def forward_and_loss(self, features):
        with torch.no_grad():
            (x1, x2), labels = features
            z1 = self.model(x1.to(self.device))
            z2 = self.model(x2.to(self.device))
            loss, logger_dict = self.loss(z1, z2)
            return loss, None, logger_dict

    def loss(self, z1, z2):
        invariance_loss = torch.nn.functional.mse_loss(z1, z2)
        logger_dict = {
            "invariance": self.lambdaa * invariance_loss,
            "variance": self.mu * (self.variance_loss(z1) + self.variance_loss(z2)),
            "covariance": self.nu * (self.covariance_loss(z1) + self.covariance_loss(z2))
        }
        return sum(logger_dict.values()), logger_dict


    def variance_loss(self, z):
        if not self.cfg.trainer.loss_variance:
            # for ablation
            return torch.tensor(0.0)

        sigma = self.gamma - torch.sqrt(torch.var(z, dim=0) + self.epsilon)
        return torch.mean(torch.relu(sigma))

    def covariance_loss(self, z):
        batch_size, dim = z.shape
        z_hat = z - torch.mean(z, dim=0)
        c_z =  z_hat.T @ z_hat / (batch_size - 1)
        c_z.fill_diagonal_(0)
        return c_z.pow_(2).sum() / dim


    def _write_summary(self, loss, logger_dict=None):
        super(VICRegTrainer, self)._write_summary(loss)
        for key, value in logger_dict.items():
            wandb.log({f"train/{key}": value.item()}, step=self.step)

    def run_valid_loop(self):
        self.model.eval()

        total_loss = []
        cumulative_logger_dict = defaultdict(list)

        for features in tqdm(self.test_dataset, desc=f'valid'):
            # compute valid loss
            self.optimizer.zero_grad()
            loss, outputs, logger_dict = self.forward_and_loss(features)
            for key, value in logger_dict.items():
                cumulative_logger_dict[key].append(value.item() if isinstance(value, torch.Tensor) else value)
            total_loss.append(loss)

            # if self.epoch % self.cfg.trainer.log_representation_every == 0:
            #     # plot representations
            #     with torch.no_grad():
            #         z = self.model.encode(imgs.to(self.device))
            #         representations.append(z.cpu().numpy())
            #         targets.append(labels.numpy())

        # log loss
        super(VICRegTrainer, self)._write_summary_valid(torch.mean(torch.stack(total_loss)))
        for key, value in cumulative_logger_dict.items():
            wandb.log({f"valid/{key}": torch.mean(torch.tensor(value))}, step=self.step)

        # if self.epoch % self.cfg.trainer.log_representation_every == 0:
        #     # compute representations
        #     representations = np.concatenate(representations, axis=0)
        #     targets = np.concatenate(targets, axis=0)
        #     pca = PCA(n_components=2)
        #     pca_result = pca.fit_transform(representations)
        #     tsne = TSNE(n_components=2, random_state=42)
        #     tsne_result = tsne.fit_transform(representations)
        #
        #     plot_2d_representations(pca_result, targets, f"PCA of Test Image Representations {self.epoch}")
        #     plot_2d_representations(tsne_result, targets, f"T-SNE of Test Image Representations {self.epoch}")



        # save checkpoint
        torch.save(self.model.state_dict(), self.cfg.model.dir / f"epoch_{self.epoch}.pt")


