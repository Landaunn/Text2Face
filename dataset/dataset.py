import torch
import numpy as np
from torch.utils.data import Dataset


class Clip2LatentDataset(Dataset):
    
    def __init__(self, cfg):
        super().__init__()
    
        self.cfg = cfg
        self.data = np.load(cfg.path, allow_pickle=True)
        self.embed_noise_scale = cfg.embed_noise_scale
        
        self.stats = self.get_stats()
        
    def __getitem__(self, idx):
        clip_emb = torch.tensor(self.data[idx]['clip_emb'])
        
        if self.embed_noise_scale > 0:
            clip_emb = self.add_noise(clip_emb, self.embed_noise_scale)
            
        latent = torch.tensor(self.data[idx]['mapping'])
        latent = self.normalise_data(
            latent,
            self.stats['w'][0],
            self.stats['w'][1]
        )
        return clip_emb, latent
    
    def __len__(self,):
        return len(self.data)
    
    def get_stats(self,):
        clip_features = torch.stack([torch.tensor(self.data[i]['clip_emb']) for i in range(10_000)])
        w = torch.stack([torch.tensor(self.data[i]['mapping']) for i in range(10_000)])
        
        stats = {
            'clip_features': self.make_data_stats(clip_features),
            'w': self.make_data_stats(w)
        }
        return stats
        
    @staticmethod   
    def make_data_stats(w):
        w_mean = w.mean(dim=0)
        w_std = w.std(dim=0)
        return w_mean, w_std
    
    @staticmethod
    def add_noise(x, scale=0.75):
        orig_norm = x.norm(dim=-1, keepdim=True)
        x = x/orig_norm
        noise = torch.randn_like(x)
        noise /= noise.norm(dim=-1, keepdim=True)
        x += scale*noise
        x /= x.norm(dim=-1, keepdim=True)
        x *= orig_norm
        return x
    
    @staticmethod
    def normalise_data(w, w_mean, w_std):
        device = w.device
        w = w - w_mean.to(device)
        w = w / w_std.to(device)
        return w
        