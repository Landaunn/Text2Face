from tqdm import tqdm
import torch

def normalise_data(w, w_mean, w_std):
    device = w.device
    w = w - w_mean.to(device)
    w = w / w_std.to(device)
    return w


def denormalise_data(w, w_mean, w_std):
    device = w.device
    w = w * w_std.to(device)
    w = w + w_mean.to(device)
    return w


@torch.no_grad()
def make_text_val_data(G, clip_model, text_samples_file):
    """Load text samples from file"""
    with open(text_samples_file, 'rt') as f:
        text_samples = f.read().splitlines()
    text_features = clip_model.embed_text(text_samples)
    val_data = {"clip_features": text_features,}
    return val_data, text_samples


@torch.no_grad()
def make_image_val_data(G, clip_model, n_im_val_samples, device, latent_dim=512):
    clip_features = []

    zs = torch.randn((n_im_val_samples, latent_dim), device='cpu')
    ws = G.mapping(zs, c=None)
    for w in tqdm(ws):
        out = G.synthesis(w.unsqueeze(0))
        image_features = clip_model.embed_image(out)
        clip_features.append(image_features)

    clip_features = torch.cat(clip_features, dim=0)
    val_data = {
        "clip_features": clip_features,
        "z": zs,
        "w": ws,
    }
    return val_data