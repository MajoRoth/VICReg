from trainers.trainer import Trainer
import torch
import wandb
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import numpy as np



class NormalizingFlowTrainer(Trainer):

    def __init__(self, cfg, model, train_dataset, test_dataset):
        super(NormalizingFlowTrainer, self).__init__(cfg, model, train_dataset, test_dataset)
        self.pz = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64))

    def train_step(self, features):
        return self.loss(features)

    def forward_and_loss(self, features):
        pass

    def loss(self, features):
        x, sum_log_det = self.model.loss(features)
        sum_log_det = -sum_log_det.mean()
        log_prob = -self.pz.log_prob(x).mean()
        logger_dict = {'sum_log_det': sum_log_det, 'log_prob': log_prob}
        return sum_log_det + log_prob, x, logger_dict

    def run_valid_loop(self):
        self.model.eval()
        losses = []
        log_dets = []
        log_probs = []

        with torch.no_grad():
            for features in tqdm(self.test_dataset, desc=f'validation {self.epoch}'):
                loss, outputs, logger_dict = self.loss(features)
                losses.append(loss)
                log_dets.append(logger_dict['sum_log_det'])
                log_probs.append(logger_dict['log_prob'])

            self._write_valid_summary(np.mean(losses), np.mean(log_dets), np.mean(log_probs))



            for i in range(3):
                samples = self.pz.sample((1000,))
                outputs = self.model(samples)
                df_samples = pd.DataFrame(outputs.detach().numpy(), columns=['x', 'y'])

                # Log the scatter plot to wandb
                wandb.log({
                    f"scatter_plot_{i}_step_{self.step}": wandb.plot.scatter(wandb.Table(dataframe=df_samples), "x", "y", title=f"Custom Y vs X Scatter Plot {i}-{self.step}")
                }, step=self.step)

            # save model
            torch.save(self.model.state_dict(), self.cfg.model.dir / f"epoch_{self.epoch}.pt")

        self.model.train()


    def _write_summary(self, loss, logger_dict=None):
        sum_log_det = logger_dict['sum_log_det']
        log_prob = logger_dict['log_prob']
        wandb.log({"train/loss": loss}, step=self.step)
        wandb.log({"train/sum_log_det": sum_log_det}, step=self.step)
        wandb.log({"train/log_prob": log_prob}, step=self.step)
        wandb.log({"train/sum": log_prob + sum_log_det}, step=self.step)

    def _write_valid_summary(self, loss, log_det, log_prob):
        wandb.log({"valid/loss": loss}, step=self.step)
        wandb.log({"valid/log_sum_det": log_det}, step=self.step)
        wandb.log({"valid/log_prob": log_prob}, step=self.step)



