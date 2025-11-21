import os
import sys
import json
import random
import subprocess
import ast
import time
import numpy as np
import argparse
import shutil
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))


def fix_indices_for_bd(object_lst_indicies):
    curr_max_token = -1
    indicator_list_in_list = False
    curr_object_lst = []
    for ele in object_lst_indicies:
        if isinstance(ele,list):
            curr_object_lst.extend(ele)
            indicator_list_in_list = True
        else:
            if ele>curr_max_token:
                curr_max_token = ele
    
    if indicator_list_in_list:
        curr_object_lst.extend([curr_max_token])  
        return curr_object_lst
    else:
        object_lst_indicies_sorted = sorted(object_lst_indicies, key=int)
        return object_lst_indicies_sorted[:-2] + object_lst_indicies_sorted[-1:]  # remove the "colored" name

def run_model_on_prompts(model, prompt, seed, output_path, token_indices=[],objects_colors_dict={},subject_token_indices=None,pipe=None):
    import sys
    sys.path.append('.')
    from scripts.models_data_creation.stable_diffusion import run_sd
        
    print(f'running {model}')
    if model == 'sd_1_4':
        print(f'seed and token indices inputs are not in use in {model}')
        model_id = 'CompVis/stable-diffusion-v1-4'
        run_sd(prompt, model_id, output_path)
    
    elif model == 'sd_1_5':
        print(f'seed and token indices inputs are not in use in {model}')
        model_id = 'runwayml/stable-diffusion-v1-5'
        run_sd(prompt, model_id, output_path)    
        
    elif model == 'sd_2_1':
        print(f'seed and token indices inputs are not in use in {model}')
        model_id = 'stabilityai/stable-diffusion-2-1-base'
        run_sd(prompt, model_id, output_path)
    
    elif model == 'flux_dev' and pipe != None:
        import torch
        import sys
        sys.path.append('.')
        from scripts.models_data_creation.flux import pipe
        
        image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(0)).images[0]
        import os
        image_output_path = os.path.join(output_path,"flux_dev.png")
        image.save(image_output_path)
    
    elif model == 'ae_1_4':
        import os
        import sys
        from pathlib import Path

        sys.path.append('.')
        sys.path.insert(0,'/storage/shayshomerchai/projects/synthesize_colors_in_context/enviroment/src/')
        sys.path.insert(0,'/storage/shayshomerchai/projects/synthesize_colors_in_context/enviroment/src/attend_and_excite/')

        from scripts.models_data_creation.attend_and_excite import ae_main_custom
        from attend_and_excite.config import RunConfig 
        from scripts.syngen_prompt_parsing import extract_subject_indicies
        
        subject_token_indices = str(extract_subject_indicies(prompt))
        
        if not token_indices:
            raise ValueError('attend and excite model need token_indices')
                
        if subject_token_indices != None:
            syngen_token_indicies = ast.literal_eval(subject_token_indices)
            if len(syngen_token_indicies) ==1:
                first_object_lst = syngen_token_indicies[0]
                first_object_lst = fix_indices_for_bd(first_object_lst)
                token_indices = [first_object_lst[-1]]
            else:
                second_object_lst, first_object_lst = syngen_token_indicies
                first_object_lst = fix_indices_for_bd(first_object_lst)
                second_object_lst = fix_indices_for_bd(second_object_lst)
                token_indices = [first_object_lst[-1],second_object_lst[-1]]
        print(f'{model} prompt:{prompt} token_indices:{token_indices}')
        config = RunConfig(prompt=prompt, seeds=[seed], token_indices=token_indices, output_path=Path(output_path))
        ae_main_custom(config)
    
    elif model == 'syngen_1_4':
        import sys
        sys.path.append('.')
        sys.path.append('/storage/shayshomerchai/projects/synthesize_colors_in_context/enviroment/src/')
        sys.path.append('/storage/shayshomerchai/projects/synthesize_colors_in_context/enviroment/src/linguistic_binding_sd/')
        from linguistic_binding_sd.run import main as syngen_main
        
        model_id = 'CompVis/stable-diffusion-v1-4'
        subject_token_indices = syngen_main(prompt, seed, output_path, model_id, step_size=20.0, attn_res=256, include_entities=False)
        
        return  subject_token_indices
    
    elif model == 'ssd_1_4':
        import os
        import sys
        from pathlib import Path

        sys.path.append('.')
        sys.path.insert(0,'/storage/shayshomerchai/projects/synthesize_colors_in_context/enviroment/src/')
        sys.path.insert(0,'/storage/shayshomerchai/projects/synthesize_colors_in_context/enviroment/src/Structured-Diffusion-Guidance/')
        sys.path.insert(0,'/storage/shayshomerchai/projects/synthesize_colors_in_context/enviroment/src/Structured-Diffusion-Guidance/scripts/')

        from scripts.models_data_creation.structured_diffusion import ssd_main_custom
        
        if not token_indices:
            raise ValueError('attend and excite model need token_indices')

        
        print(f'{model} prompt:{prompt} token_indices:{token_indices}')
        if ' and ' in prompt:
            ssd_main_custom(prompt=prompt,outdir=output_path, plms=True, parser_type='constituency', conjunction=True) 
        else:
            ssd_main_custom(prompt=prompt,outdir=output_path, plms=True, parser_type='constituency', conjunction=False) 

    return subject_token_indices


def run_manager(models_seeds_dict,run_name,html_color_dict_path,prompts_file_path,base_out_dir,multiple_seeds=False,
                single_color_prompt=False,rich_text_html_name_mapping_dict=None,original_config_path='',start_from_middle=False):
    # seeds = [160, 1269] #30,0,,7
    os.makedirs(base_out_dir, exist_ok=True)
    shutil.copy2(original_config_path, base_out_dir)

    models = list(models_seeds_dict.keys())
        
    print(f'starting models_seeds_dict: {models_seeds_dict} , {prompts_file_path}, {base_out_dir}')
    timestr = time.strftime("%Y%m%d_%H%M%S")
    
    color_terms_rgb_dict = {}
    with open(html_color_dict_path, 'r') as f:
        color_terms_rgb_dict = json.load(f)

    prompts_dict = {}
    with open(prompts_file_path, 'r') as f:
        prompts_dict = json.load(f)

    prompt_count = -1
    finished_prompts = []
    if start_from_middle:
        for prompt in os.listdir(start_from_middle):
            if os.path.exists(os.path.join(start_from_middle,prompt,'0/flux_dev/flux_dev.png')):
                with open(os.path.join(start_from_middle,prompt,'0','run_config.json')) as f:
                    finished_prompts.append(json.load(f)['original_prompt'])
                    curr_prompt_count = int(prompt.split('prompt')[-1])
                    if curr_prompt_count> prompt_count:
                        prompt_count = curr_prompt_count
            else:
                print()
    
    pipe = None
    if models[0] == 'flux_dev':
        os.system(" huggingface-cli login --token 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")
        import sys
        sys.path.append('.')
        from running_scripts.run_flux import pipe

    ref_compare_color_format = False
    for benchmark_type, prompt_list in prompts_dict.items():
        base_out_dir_with_benchmark_type = os.path.join(base_out_dir,benchmark_type)
        out_dir = f'{base_out_dir_with_benchmark_type}/{timestr}_{run_name}'
        for i in range(len(prompt_list)):
            if start_from_middle and (prompt_list[i]['prompt'] in finished_prompts):
                out_dir = start_from_middle
                continue
            if i<=prompt_count:
                raise ValueError
            main_out_dir_run_prompt =os.path.join(out_dir,f'prompt{str(i)}')
            os.makedirs(main_out_dir_run_prompt, exist_ok=True)
            last_valid_subject_token_indices = None
            seeds = list(models_seeds_dict.values())
            if multiple_seeds:
                seeds.extend(list(range(10,100)))
            seeds = list(set(seeds))
            for seed in seeds:
                models_seeds_dict = {list(models_seeds_dict.keys())[0] : seed }
                for model_name,seed in models_seeds_dict.items():                            
                    prompt = prompt_list[i]
                    print(f'model:{model_name}, seed: {seed} , prompt: {prompt}')

                    if isinstance(prompt,dict):
                        ref_compare_color_format = True
                        ref_color = prompt['ref_color']
                        close_color = prompt.get('close_color', None)
                        distant_color = prompt.get('distant_color', None)
                        prompt = prompt['prompt']
                    
                    prompt_splitted = prompt.split(' ')
                    first_color = prompt_splitted[1]
                    first_object = prompt_splitted[3]
                    second_color = prompt_splitted[6]
                    second_object = prompt_splitted[8]
                    third_color = prompt_splitted[11]
                    third_object = prompt_splitted[13]
                    
                    first_base_color = color_terms_rgb_dict[first_color]['base_color_name']
                    second_base_color = color_terms_rgb_dict[second_color]['base_color_name']
                    third_base_color = color_terms_rgb_dict[third_color]['base_color_name']
                    
                    cic_prompt = f'a {first_base_color} {first_object}'\
                    + f' and a {second_base_color} {second_object}'\
                    + f' and a {third_base_color} {third_object}'
                                
                    objects_colors_dict = {first_object: color_terms_rgb_dict[first_color], 
                                           second_object: color_terms_rgb_dict[second_color],
                                           third_object: color_terms_rgb_dict[third_color]}

                    run_config = {
                        'seed' : seed,
                        'original_prompt' : prompt,
                        'simplified_text_prompt' : f'a {first_object} and a {second_object} and a {third_object}',
                        'full_text_prompt' : cic_prompt,
                        'objects_colors_dict' : objects_colors_dict
                    }
                        
                    if ref_compare_color_format:
                        run_config['ref_color'] = ref_color
                        run_config['close_color'] = close_color
                        run_config['distant_color'] = distant_color
                        
                    main_out_dir_run_prompt_seed =os.path.join(main_out_dir_run_prompt,str(seed))
                    os.makedirs(main_out_dir_run_prompt_seed,exist_ok=True)
                    # save configuration
                    with open(f'{main_out_dir_run_prompt_seed}/run_config.json', 'w') as f:
                        json.dump(run_config, f, indent=4)
                    
                    model_out_dir = os.path.join(main_out_dir_run_prompt_seed,model_name)
                    os.makedirs(model_out_dir, exist_ok=True)
                    print("Before running")
                    print(f'model:{model_name}, seed: {seed} , prompt: {prompt}')
                    subject_token_indices = run_model_on_prompts(model_name, prompt, seed, model_out_dir,token_indices=-1,
                                                                objects_colors_dict=objects_colors_dict,subject_token_indices=last_valid_subject_token_indices,pipe=pipe)
                    if subject_token_indices != None:
                        last_valid_subject_token_indices = subject_token_indices
            # if i==5:
            #     break
        
    print()   


def main(run_config_dict,config_path):
   
    run_manager(models_seeds_dict = run_config_dict['models_seeds_dict'],
                run_name = run_config_dict['run_name'],
                html_color_dict_path = run_config_dict['html_color_dict_path'],
                prompts_file_path = run_config_dict['prompts_file_path'],
                base_out_dir = run_config_dict['out_dir'],
                rich_text_html_name_mapping_dict = run_config_dict.get('rich_text_html_name_mapping_dict_path',None),
                multiple_seeds = run_config_dict.get('multiple_seeds', False),
                single_color_prompt =  run_config_dict.get('single_color_prompt',False),
                original_config_path=config_path,
                start_from_middle =  run_config_dict.get('start_from_middle',False))
    
if __name__ == '__main__':
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/storage/shayshomerchai/projects/synthesize_colors_in_context/algorithmic_experiments/benchmark/final_benchmark1110/3_colors_configs/flux_dev_close_distant_3_colors_start_from_middle.json')
    args = parser.parse_args()
    
    run_config_dict = {}
    with open(args.config, 'r') as f:
        run_config_dict = json.load(f)
    
    main(run_config_dict,args.config)

    