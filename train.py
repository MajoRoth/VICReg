import torch
import json
import argparse
import os
import sys

sys.path.append("/tmp/pycharm_project_765")

import wandb
from pathlib import Path

from trainers.trainer_getter import get_trainer
from models.model_getter import get_model
from confs.conf_getter import get_conf
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data.transform import TrainTransform, TestTransform




def train(args):
    cfg = get_conf(args.conf)
    print(f'ARGS: {args}')
    print(f'PARAMS: {cfg}')

    cfg.model.dir = Path(f"./checkpoints/{args.conf}")

    wandb.init(project="VICReg", name=args.conf, resume="allow", notes=f"{cfg}")

    os.makedirs(cfg.model.dir, exist_ok=True)

    model = get_model(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.trainer.augment:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=TrainTransform())
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=TrainTransform())
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=TestTransform())
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=TestTransform())

    train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=4)

    trainer = get_trainer(cfg)
    trainer(cfg=cfg, model=model, train_dataset=train_loader, test_dataset=test_loader).train()


def get_parser():
    parser = argparse.ArgumentParser(description='train an neural network')
    parser.add_argument('--conf', default="vicreg", type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    train(args)
