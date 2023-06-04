from torch.utils.data import DataLoader

from dataset.dataset import Clip2LatentDataset

def load_data(cfg):
    ds = Clip2LatentDataset(cfg)
    stats = ds.stats

    loader = DataLoader(ds, batch_size=cfg.bs, shuffle=True)
    
    return stats, loader