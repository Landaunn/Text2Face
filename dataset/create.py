import os
os.environ["OMP_NUM_THREADS"] = "8"

from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import ruclip

from stylegan2 import load_sg


def create_dataset(config_path: str = 'configs/dataset/create_dataset.yaml'):
    config = OmegaConf.load(config_path)

    clip_device = config.clip.device
    gan_device = config.stylegan2.device

    # load clip model
    clip, processor = ruclip.load(**config.clip)

    #load stylegan2 model
    stylegan2 = load_sg(**config.stylegan2)

    dataset = []
    for _ in tqdm(range(config.num_samples)):
        tensor = stylegan2.mapping(torch.randn((1, 512)).to(gan_device), c=None)
        image = stylegan2.synthesis(tensor.to(gan_device),)
        gen_image = (255*image.squeeze(0).clamp(-1, 1)/2 + 127.5).to(torch.uint8).permute(1,2,0).numpy()

        pil_images = [Image.fromarray(gen_image)]
        image_for_clip = processor.image_transform(Image.fromarray(gen_image))
        clip_emb = clip.encode_image(image_for_clip.unsqueeze(0).to(clip_device))
        clip_emb = clip_emb.squeeze().detach().cpu().numpy().reshape(-1)
        mapping = tensor.squeeze().detach().cpu().numpy()[0, :]

        sample = {
            'clip_emb': clip_emb,
            'mapping': mapping
        }

        dataset.append(sample)
        np.save(config.dataset_save_path,  np.array(dataset))


if __name__ == '__main__':
    create_dataset()
    



