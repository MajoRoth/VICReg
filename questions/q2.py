import sys
import torch

sys.path.append("/cs/labs/adiyoss/amitroth/vicreg")

from tqdm import tqdm
import matplotlib.pyplot as plt
from models.model_getter import get_best_ckpt
from models.model_getter import get_model
from confs.conf_getter import get_conf
import torchvision.datasets as datasets
from data.transform import TestTransform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_2d_representations(data, labels, title):
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


def main(args):
    cfg = get_conf(args.conf)
    print(f'ARGS: {args}')
    print(f'PARAMS: {cfg}')

    cfg.model.dir = Path(f"./checkpoints/{args.conf}")
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=TestTransform())
    test_loader = DataLoader(test_dataset, batch_size=cfg.trainer.batch_size, shuffle=False, num_workers=4)

    model = get_model(cfg)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load("/cs/labs/adiyoss/amitroth/vicreg/checkpoints/vicreg/epoch_59.pt"))

    representations = list()
    targets = list()
    for imgs, labels in tqdm(test_loader, desc=f'calculating representations'):
        with torch.no_grad():
            z = model.encode(imgs.to(device))
            representations.append(z.cpu().numpy())
            targets.append(labels.numpy())

    representations = np.concatenate(representations, axis=0)
    targets = np.concatenate(targets, axis=0)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(representations)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(representations)

    plot_2d_representations(pca_result, targets, f"PCA of Test Image Representations")
    plot_2d_representations(tsne_result, targets, f"T-SNE of Test Image Representations")


def get_parser():
    parser = argparse.ArgumentParser(description='train an neural network')
    parser.add_argument('--conf', default="vicreg", type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args)
