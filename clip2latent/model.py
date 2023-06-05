import torch
from PIL import Image

from dalle2_pytorch import DiffusionPriorNetwork
from dalle2_pytorch.train import DiffusionPriorTrainer
import ruclip

from clip2latent.latent_prior import LatentPrior, WPlusPriorNetwork
from stylegan2 import load_sg


class Clipper(torch.nn.Module):
    def __init__(self, clip_variant):
        super().__init__()
        clip_model, processor = ruclip.load(clip_variant, device="cpu")
        self.clip = clip_model
        self.processor = processor

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        gen_image = (255*image.squeeze(0).clamp(-1, 1)/2 + 127.5).to(torch.uint8).permute(1,2,0).numpy()
        image_for_clip = self.processor.image_transform(Image.fromarray(gen_image))
        return self.clip.encode_image(image_for_clip.unsqueeze(0))

    def embed_text(self, text_samples):
        input_ids = self.processor(text=text_samples)[['input_ids']]
        return self.clip.encode_text(input_ids)

    def _get_device(self):
        for p in self.clip.parameters():
            return p.device


def load_models(cfg, device, stats=None):
    """Load the diffusion trainer and eval models based on a config

    If the model requires statistics for embed or latent normalisation
    then these should be passed into this function, unless the state of
    the model is to be loaded from a state_dict (which will contain these)
    statistics, in which case the stats will be filled with dummy values.
    """
    if cfg.data.n_latents > 1:
        prior_network = WPlusPriorNetwork(n_latents=cfg.data.n_latents, **cfg.model.network).to(device)
    else:
        prior_network = DiffusionPriorNetwork(**cfg.model.network).to(device)

    embed_stats = latent_stats = (None, None)
    if stats is None:
        # Make dummy stats assuming they will be loaded from the state dict
        clip_dummy_stat = torch.zeros(cfg.model.network.dim,1)
        w_dummy_stat = torch.zeros(cfg.model.network.dim)
        if cfg.data.n_latents > 1:
            w_dummy_stat = w_dummy_stat.unsqueeze(0).tile(1, cfg.data.n_latents)
        stats = {"clip_features": (clip_dummy_stat, clip_dummy_stat), "w": (w_dummy_stat, w_dummy_stat)}

    if cfg.train.znorm_embed:
        embed_stats = stats["clip_features"]
    if cfg.train.znorm_latent:
        latent_stats = stats["w"]

    diffusion_prior = LatentPrior(
        prior_network,
        num_latents=cfg.data.n_latents,
        latent_repeats=cfg.data.latent_repeats,
        latent_stats=latent_stats,
        embed_stats=embed_stats,
        **cfg.model.diffusion).to(device)
    diffusion_prior.cfg = cfg

    # Load eval models
    G = load_sg(cfg.data.sg_pkl).to(device)
    clip_model = Clipper(cfg.data.clip_variant).to(device)

    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=cfg.train.lr,
        wd=cfg.train.weight_decay,
        ema_beta=cfg.train.ema_beta,
        ema_update_every=cfg.train.ema_update_every,
    ).to(device)

    return G, clip_model, trainer 
