# ColorEdit: Our Inference-Time Color Editing Method

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL
from tqdm import tqdm
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from torchvision import transforms
from diffusers import DDIMScheduler

from coloredit_code.coloredit_diffusion_pipeline import ColorEditDiffsuionPipeline
from coloredit_code.controllers import AttentionBase, AttentionStore, SelfAttentionControlEditCic, SelfCrossAttentionControlColorEdit, regiter_attention_editor_diffusers, register_attention_control_FPE, register_attention_control_ColorEdit
from coloredit_code.utils import show_cross_self_attention_maps

seed = 42
NUM_DIFFUSION_STEPS = 50
seed_everything(seed)

class ColorEdit():

    def __init__(self,
                 num_segments = 5,
                 background_segment_threshold=0.3,
                 background_blend_timestep=35,
                 **kwargs) -> None:
                 
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.num_segments = num_segments
        self.background_segment_threshold = background_segment_threshold
        self.background_blend_timestep=background_blend_timestep
        self.background_blend_timestep_start = kwargs.get('background_blend_timestep_start',background_blend_timestep)
        self.debug = kwargs.get('debug', False)
        self.load_model()
        
    def load_model(self):
        model_path = "runwayml/stable-diffusion-v1-5"
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.model = ColorEditDiffsuionPipeline.from_pretrained(model_path, scheduler=scheduler).to(self.device)

    def load_image(self,image_path, device):
        # Reads a file using pillow
        PIL_image = PIL.Image.open(image_path)
        image = transforms.PILToTensor()(PIL_image)
        image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
        image = F.interpolate(image, (512, 512))
        image = image.to(device)
        return image
    
    def set_editor(self, editor_type, prompts,low_resource=True,args ={}):
       
        if editor_type == 'AttentionStore':
            editor = AttentionStore()
            register = regiter_attention_editor_diffusers
        
        # editor for simplified prompt step
        elif editor_type == 'SelfAttentionControlEditCic':
            editor = SelfAttentionControlEditCic(prompts,
                                                 NUM_DIFFUSION_STEPS,
                                                 low_resource=low_resource,
                                                 self_replace_steps=0.8)
            print(f"{editor_type} low_resource: {editor.low_resource}")
            register = register_attention_control_FPE
        
        # editor for target prompt step
        elif editor_type == 'SelfCrossAttentionControlColorEdit':
            editor = SelfCrossAttentionControlColorEdit(prompts,
                                                           NUM_DIFFUSION_STEPS,
                                                           self_replace_steps=0.8,
                                                           low_resource=low_resource,
                                                           reference_attention_store=args['reference_attention_store'])
            print(f"{editor_type} low_resource: {editor.low_resource}")
            register = register_attention_control_ColorEdit
        else:
            raise(ValueError)

        return register, editor
    
    
    def synthesize_editing(self,
                           image_path,
                           simplified_prompt,
                           color_object_dict={},
                           target_prompt='',
                           simplified_prompt_token_indicies=[],
                           target_prompt_token_indicies=[],
                           editor_type_target_prompt='AttentionStore',
                           use_ref_intermediate_latents=False,
                           attention_loss_iters=10,
                           attention_loss_weight=20,
                           attnetion_loss_stoping_step=35,
                           attnetion_symmetric_kl_bottom_limit=0.1,
                           color_loss_weight=1.5,
                           color_loss_starting_step=25,
                           out_dir='',
                           SAM_mask_dict={},
                           use_attention_loss=True,
                           use_color_loss=True):
        
        # make outdirs
        os.makedirs(out_dir, exist_ok=True)
        sample_count = len(os.listdir(out_dir))
        self.out_dir = os.path.join(out_dir, f"sample_{sample_count}")
        os.makedirs(self.out_dir , exist_ok=True)
        
        # load image  
        source_image = self.load_image(image_path, self.device)
        
        # assuming empty source prompt
        source_prompt = ''
        print(f'source prompt: {source_prompt}')
        print(f'simplified prompt: {simplified_prompt}')
        print(f'target prompt: {target_prompt}')

        # prepearing the prompts for each step
        simplified_prompts = [source_prompt, simplified_prompt]
        target_prompts = [source_prompt, target_prompt]

        # 1: invert the source image
        regiter_attention_editor_diffusers(self.model, editor=AttentionBase())
        start_code, latents_list = self.model.invert(source_image,
                                                     source_prompt,
                                                     guidance_scale=7.5,
                                                     num_inference_steps=NUM_DIFFUSION_STEPS,
                                                     return_intermediates=True)
        start_code_base = start_code
        start_code = start_code.expand(len(target_prompts), -1, -1, -1)
        
        # allow the use of ref intermediate latents for better background preservation
        if use_ref_intermediate_latents :
            ref_intermediate_latents = latents_list
        else:
            ref_intermediate_latents = None
        

        # 2: simplified prompt forward process
        print("starting simplified prompt foward process")
        start_code = start_code_base.expand(len(simplified_prompts), -1, -1, -1)
        register_func,first_stage_editor = self.set_editor('SelfAttentionControlEditCic', 
                                                           simplified_prompts, 
                                                           low_resource=False)
        register_func(self.model, first_stage_editor)
        
        _, _ = self.model(prompt=simplified_prompts,
                          latents=start_code,
                          num_inference_steps=NUM_DIFFUSION_STEPS,
                          guidance_scale=7.5,
                          ref_intermediate_latents=ref_intermediate_latents)
        
        # save self and cross attention maps for debug 
        if self.debug:
            show_cross_self_attention_maps(first_stage_editor,
                                           res={'cross':16,'self':32},
                                           from_where=("up", "down"),
                                           prompts=simplified_prompts, 
                                           tokenizer=self.model.tokenizer,
                                           select=1,
                                           out_dir=self.out_dir,
                                           diff_step=50,
                                           prefix='a_previous_run')
        
        # 3: target prompt forward process
        print("starting ColorEditing")
        start_code = start_code_base.expand(len(target_prompts), -1, -1, -1)
        register_func, editor = self.set_editor(editor_type_target_prompt, 
                                                target_prompts, 
                                                low_resource=False,
                                                args={'reference_attention_store' : first_stage_editor.attention_store}) 
        # clear memory
        del first_stage_editor
        torch.cuda.empty_cache()      
        
        register_func(self.model, editor)    
        image_coloredit, mask = self.model(target_prompts,
                                           latents=start_code,
                                           guidance_scale=7.5,
                                           controller=editor,
                                           num_inference_steps=NUM_DIFFUSION_STEPS,
                                           ref_intermediate_latents=ref_intermediate_latents,
                                           simplified_prompt=simplified_prompt,
                                           simplified_prompt_token_indicies=simplified_prompt_token_indicies,
                                           target_prompt_token_indicies=target_prompt_token_indicies,
                                           color_object_dict=color_object_dict,
                                           num_segments=self.num_segments,
                                           background_segment_threshold=self.background_segment_threshold,
                                           background_blend_timestep=self.background_blend_timestep,
                                           background_blend_timestep_start=self.background_blend_timestep_start,
                                           attention_loss_iters=attention_loss_iters,
                                           attention_loss_weight=attention_loss_weight,
                                           attnetion_loss_stoping_step=attnetion_loss_stoping_step,
                                           attnetion_symmetric_kl_bottom_limit=attnetion_symmetric_kl_bottom_limit,
                                           color_loss_weight=color_loss_weight,
                                           color_loss_starting_step=color_loss_starting_step,
                                           debug_cross_self_maps=self.debug,
                                           out_dir=self.out_dir,
                                           SAM_mask_dict=SAM_mask_dict,
                                           use_attention_loss=use_attention_loss,
                                           use_color_loss=use_color_loss)
                
        mask_upsampled = torch.nn.Upsample(size=(source_image.shape[2], source_image.shape[3]), mode='nearest')(mask)
        mask_upsampled_3d = torch.cat([mask_upsampled, mask_upsampled, mask_upsampled], dim=1) 
        out_image = torch.cat([source_image * 0.5 + 0.5,
                               mask_upsampled_3d,
                               image_coloredit[1:2]],
                               dim=0)
        
        print("finished ColorEdit")

        save_image(out_image, os.path.join(self.out_dir, f"output_all_images.png"))
        save_image(out_image[0], os.path.join(self.out_dir, f"source_image.png"))
        save_image(out_image[1], os.path.join(self.out_dir, f"mask_of_{target_prompt.replace(' ','_')}.png"))
        save_image(out_image[2], os.path.join(self.out_dir, f"ColorEdit_image.png"))

        print("Syntheiszed images are saved in", self.out_dir)
        
        
        print("Delete memory from GPU stage1 and model")
        del mask_upsampled
        del mask_upsampled_3d
        del start_code
        del start_code_base
        del source_image
        del editor
        del latents_list
        print("Cleaning memory from GPU stage1 stage2 and model")    
        torch.cuda.empty_cache()

        return out_image