import os 
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDIMScheduler

sys.path.append('.')
sys.path.append('enviroment/src/')
sys.path.append('enviroment/src/EasyNLP/diffusion/FreePromptEditing/')

from Freeprompt.diffuser_utils import FreePromptPipeline
from Freeprompt.freeprompt_utils import register_attention_control_new
from torchvision.utils import save_image
from torchvision.io import read_image
from Freeprompt.freeprompt import SelfAttentionControlEdit,AttentionStore


def load_FPE_pipe():
    # Note that you may add your Hugging Face token to get access to the models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe = FreePromptPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

    # Note that you may add your Hugging Face token to get access to the models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return pipe, device

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


# out_dir = "examples/outputs"

def run_FPE_editing(pipe, device, source_image_path, target_prompt):
    import time
    start = time.time()
    self_replace_steps = .8
    NUM_DIFFUSION_STEPS = 50
    
    # SOURCE_IMAGE_PATH = "examples/img/000141.jpg"
    source_image = load_image(source_image_path, device)

    source_prompt = ""

    # invert the source image
    start_code, latents_list = pipe.invert(source_image,
                                            source_prompt,
                                            guidance_scale=7.5,
                                            num_inference_steps=50,
                                            return_intermediates=True)

    # target_prompt = 'a red car'

    latents = torch.randn(start_code.shape, device=device)
    prompts = [source_prompt, target_prompt]

    start_code = start_code.expand(len(prompts), -1, -1, -1)
    controller = SelfAttentionControlEdit(prompts, NUM_DIFFUSION_STEPS, self_replace_steps=self_replace_steps)

    register_attention_control_new(pipe, controller)

    # Note: querying the inversion intermediate features latents_list
    # may obtain better reconstruction and editing results
    results = pipe(prompts,
                        latents=start_code,
                        guidance_scale=7.5,
                        ref_intermediate_latents=latents_list)
    end = time.time()
    print(end-start)
    return source_image*0.5+0.5, results[1]
    # save_image(source_image, os.path.join(out_dir, 'source_image.jpg'))
    # save_image(results[1], os.path.join(out_dir, 'FPE_image'+'.jpg'))
