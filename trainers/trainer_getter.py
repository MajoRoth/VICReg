from pathlib import Path
from trainers.vicreg_trainer import VICRegTrainer
from trainers.linear_probing_trainer import LinearProbingTrainer



def get_trainer(cfg):
    if cfg.model.name == "vicreg":
        return VICRegTrainer
    if cfg.model.name == "linear_probing":
        return LinearProbingTrainer

    else:
        raise Exception(f"Model {cfg.model.name} not found")



