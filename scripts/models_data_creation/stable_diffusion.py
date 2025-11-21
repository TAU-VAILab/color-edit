import torch
import base64
import os
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    
def run_sd(prompt, model_id, output_path):
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
  pipe = pipe.to("cuda")
  image = pipe(prompt).images[0]
  image.save(f'{output_path}/{prompt}.png')
  return image
