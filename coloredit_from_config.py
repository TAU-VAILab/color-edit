# ColorEdit: Our Inference-Time Color Editing Method

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time 
import shutil
from torchvision.utils import save_image

from coloredit_code.coloredit import ColorEdit

def normlized_color_dict(color_object_dict):
    new_dict = {}
    for k,v in color_object_dict.items():
        new_dict[k] = list(np.array(v)/255)
    return new_dict

def run_editing(run_config,prompt_images_path_config_dict,out_dir=None):
    # configuration
    if out_dir is None:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        run_name = run_config['run_name']
        base_out_dir = run_config['out_dir']
        base_out_dir_run = f'{base_out_dir}/{timestr}_{run_name}'
        os.makedirs(base_out_dir_run)
        # save configuration
        with open(f'{base_out_dir_run}/run_config.json', 'w') as f:
            json.dump(run_config, f, indent=4)
    else:
        base_out_dir_run = out_dir
    
    scic = ColorEdit(**run_config['scic_config'])
        
    for prompt_num, config in prompt_images_path_config_dict.items():
        model_name = config['model_name']
        original_generation_prompt = config['original_prompt']
        simplified_prompt = config['simplified_text_prompt']
        target_prompt = config['full_text_prompt']
        color_object_dict_with_hex = config['color_object_dict']
        
        color_object_dict = {}
        objects_sam_mask_dict = {}
        for object_name, color_hex_rgb in color_object_dict_with_hex.items():
            color_object_dict[object_name] = color_hex_rgb['RGB']
            if run_config.get('adjust_single_token_colors',False):
                # check if original color name is single token if so replace back to original color name
                original_color_name_tokens = scic.model.tokenizer(color_hex_rgb['color_name'])
                if len(original_color_name_tokens['input_ids'])==3: #start token + color name token + end token
                    target_prompt = target_prompt.replace(color_hex_rgb['base_color_name'] + f' {object_name}', color_hex_rgb['color_name'] + f' {object_name}')
            objects_sam_mask_dict[object_name] = config[f'mask_{object_name}']
             
        config['full_text_prompt'] = target_prompt
        color_object_dict = normlized_color_dict(color_object_dict) 

        image_path = config['image_path']
        print(f'original_generation_prompt: {original_generation_prompt}')
        print(f'simplified_prompt: {simplified_prompt}')
        print(f'target_prompt: {target_prompt}')
        print(f'color_object_dict: {color_object_dict}')
        print(f'image_path: {image_path}')
        
        # create token indices for each prompt
        if len(objects_sam_mask_dict.keys())==3:
            # a {object1} and a {object2} and a {object3}"   
            # a {color1} {object1} and a {color2} {object2} and a {color3} {object3}"
            simplified_prompt_token_indicies = [[2],[5],[8]]
            target_prompt_token_indicies = [[2,3],[6,7],[10,11]]
        elif len(objects_sam_mask_dict.keys())==2:
            # a {object1} and a {object2}"   
            # a {color1} {object1} and a {color2} {object2}"
            simplified_prompt_token_indicies = [[2],[5]]
            target_prompt_token_indicies = [[2,3],[6,7]]
        elif len(objects_sam_mask_dict.keys())==1:
            # a {object1}"   
            # a {color1} {object1}"
            simplified_prompt_token_indicies = [[2]]
            target_prompt_token_indicies = [[2,3]]
        else:
            raise ValueError("ColorEdit current implementation support up to 3 object")

        kwargs = {}
        kwargs['SAM_mask_dict'] = {}
        if run_config.get('use_SAM', False):
            kwargs['SAM_mask_dict'] = objects_sam_mask_dict  
        if not run_config.get('use_attention_loss', True):
            kwargs['use_attention_loss'] = False
        if not run_config.get('use_color_loss', True):
            kwargs['use_color_loss'] = False
        
        out_image = scic.synthesize_editing(image_path=image_path,
                                            simplified_prompt=simplified_prompt,
                                            color_object_dict=color_object_dict,
                                            target_prompt=target_prompt,
                                            simplified_prompt_token_indicies = simplified_prompt_token_indicies,
                                            target_prompt_token_indicies = target_prompt_token_indicies,
                                            editor_type_target_prompt=run_config['editor_type_target_prompt'],
                                            use_ref_intermediate_latents=run_config['use_ref_intermediate_latents'],
                                            attention_loss_iters=run_config['attention_loss_iters'],
                                            attention_loss_weight=run_config['attention_loss_weight'],
                                            attnetion_loss_stoping_step=run_config['attnetion_loss_stoping_step'],
                                            attnetion_symmetric_kl_bottom_limit=run_config['attnetion_symmetric_kl_bottom_limit'],
                                            color_loss_weight=run_config['color_loss_weight'],
                                            color_loss_starting_step=run_config['color_loss_starting_step'],
                                            out_dir=base_out_dir_run,
                                            **kwargs)
        
        for object_name, sam_mask_path in objects_sam_mask_dict.items():
            shutil.copy2(sam_mask_path,f'{scic.out_dir}/sam_mask_{object_name}.png')
        with open(f'{scic.out_dir}/{model_name}_run_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        # create model folder
        model_out_path = os.path.join(base_out_dir_run,f'{run_name}_edited_output',model_name)
        os.makedirs(model_out_path,exist_ok=True)
        
        #check curr idx folder in model dir
        next_img_idx = len(os.listdir(model_out_path))
        
        #create the next folder
        model_out_path_idx = os.path.join(model_out_path,str(next_img_idx))
        os.makedirs(model_out_path_idx)
        
        config['original_out_dir'] = scic.out_dir
        with open(f'{model_out_path_idx}/run_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        save_image(out_image[0], os.path.join(model_out_path_idx, f"source_image.png"))
        save_image(out_image[1], os.path.join(model_out_path_idx, f"mask_image.png"))
        save_image(out_image[2], os.path.join(model_out_path_idx, f"ColorEdit_image.png"))
        
        del out_image
        torch.cuda.empty_cache()
    
    return

def main(run_config,config_path):
    
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    if 'force_out_dir' in run_config.keys():
        out_dir = run_config['force_out_dir']
    else:
        out_dir = run_config['out_dir']
    
    os.makedirs(out_dir,exist_ok=True)
    shutil.copy2(config_path,out_dir)
    
    
    with open(run_config['source_images_path_config'] , 'r') as f:
        prompt_images_path_config_dict = json.loads(f.read()) 
    
    run_editing(run_config,prompt_images_path_config_dict)
    
if __name__ == '__main__':
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/storage/shayshomerchai/projects/color-edit/stable_diffusion_2_1_config.json')
    args = parser.parse_args()
    
    run_config_dict = {}
    with open(args.config, 'r') as f:
        run_config_dict = json.load(f)
    
    main(run_config_dict,args.config)

    