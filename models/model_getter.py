from pathlib import Path
from models.vicreg import VICReg
from models.linear_probing import LinearProbing



def get_model(cfg):
    if cfg.model.name == "vicreg":
        return VICReg(D=cfg.model.encoder_dim, proj_dim=cfg.model.proj_dim, device='cuda')
    if cfg.model.name == "linear_probing":
        return LinearProbing(D=cfg.model.encoder_dim, proj_dim=cfg.model.proj_dim, device='cuda')
    else:
        raise Exception(f"Model {cfg.model.name} not found")



def get_best_ckpt(checkpoint_dir: Path):
    ckpts = checkpoint_dir.glob('*.pt')
    ckpts = sorted(ckpts)
    return ckpts[-1] if ckpts else None


if __name__ == '__main__':
    print(get_best_ckpt("./../checkpoints/"))
