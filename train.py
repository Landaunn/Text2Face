from dataset import load_data
from clip2latent.model import load_models
from clip2latent.train_utils import make_text_val_data, make_image_val_data

def main(cfg):
    
    device = cfg.device
    stats, loader = load_data(cfg.data)
    G, clip_model, trainer = load_models(cfg, device, stats)
    
    text_embed, text_samples = make_text_val_data(G, clip_model, cfg.data.val_text_samples)
    
    val_data = {
        "val_im": make_image_val_data(G, clip_model, cfg.data.val_im_samples, device),
        "val_text": text_embed,
        "val_caption": text_samples,
    }
    
    return val_data