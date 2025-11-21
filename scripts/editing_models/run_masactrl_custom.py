import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
# from omegaconf import OmegaConf

from diffusers import DDIMScheduler

sys.path.append('.')
sys.path.append('enviroment/src/')
sys.path.append('enviroment/src/MasaCtrl/')

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

torch.cuda.set_device(0)  # set the GPU device

def load_masactrl_model():
    # Note that you may add your Hugging Face token to get access to the models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model_path = "xyn-ai/anything-v4.0"
    model_path = "CompVis/stable-diffusion-v1-4"
    # model_path = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    return model, device

from masactrl.masactrl import MutualSelfAttentionControl
from torchvision.io import read_image


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def run_masactrl_editing(model, device, image_path, target_prompt):
    import time
    start=time.time()
    seed = 42
    seed_everything(seed)

    source_image = load_image(image_path, device)

    source_prompt = ""
    # target_prompt = "a photo of a running corgi"
    prompts = [source_prompt, target_prompt]

    # invert the source image
    start_code, latents_list = model.invert(source_image,
                                            source_prompt,
                                            guidance_scale=7.5,
                                            num_inference_steps=50,
                                            return_intermediates=True)
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # # results of direct synthesis
    # editor = AttentionBase()
    # regiter_attention_editor_diffusers(model, editor)
    # image_fixed = model([target_prompt],
    #                     latents=start_code[-1:],
    #                     num_inference_steps=50,
    #                     guidance_scale=7.5)

    # inference the synthesized image with MasaCtrl
    STEP = 4
    LAYPER = 10

    # hijack the attention module
    editor = MutualSelfAttentionControl(STEP, LAYPER)
    regiter_attention_editor_diffusers(model, editor)

    # # inference the synthesized image
    # image_masactrl = model(prompts,
    #                     latents=start_code,
    #                     guidance_scale=7.5)
    # Note: querying the inversion intermediate features latents_list
    # may obtain better reconstruction and editing results
    image_masactrl = model(prompts,
                           latents=start_code,
                           guidance_scale=7.5,
                           ref_intermediate_latents=latents_list)
    end = time.time()
    print(end-start)
    return source_image * 0.5 + 0.5, image_masactrl[-1:]
    
    # # save the synthesized image
    # out_image = torch.cat([source_image * 0.5 + 0.5,
    #                     image_masactrl[0:1],
    #                     image_fixed,
    #                     image_masactrl[-1:]], dim=0)
    # save_image(out_image, os.path.join(out_dir, f"all_step{STEP}_layer{LAYPER}.png"))
    # save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"))
    # save_image(out_image[1], os.path.join(out_dir, f"reconstructed_source_step{STEP}_layer{LAYPER}.png"))
    # save_image(out_image[2], os.path.join(out_dir, f"without_step{STEP}_layer{LAYPER}.png"))
    # save_image(out_image[3], os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYPER}.png"))

    # print("Syntheiszed images are saved in", out_dir)