
import os
import sys
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import PIL

def get_object_mask(model, image_path, object_name1, object_name2, masks_multi=True):
    try:
        image_pil = PIL.Image.open(image_path).convert("RGB")
        results = model.predict([image_pil, image_pil], [object_name1, object_name2])
        # Check if masks exist
        if len(results[0]["masks"]) == 0 or len(results[1]["masks"]) == 0:
            return None, None  # Indicate missing masks
        # Convert masks to boolean
        if masks_multi:
            mask1 = np.bitwise_or.reduce(results[0]["masks"].astype(bool),axis=0)
            mask2 = np.bitwise_or.reduce(results[1]["masks"].astype(bool),axis=0)
        else:
            mask1 = results[0]["masks"][0].astype(bool).squeeze()
            mask2 = results[1]["masks"][0].astype(bool).squeeze()
        if len(mask1.shape) != 2:
            mask1 = mask1[0]
        if len(mask2.shape) != 2:
            mask2 = mask2[0]
        return mask1, mask2
    except Exception as e:
        print(f"Error during mask prediction: {e}")
        return None, None

def prepare_run_configs_for_editing_base_on_dir(source_images_path, model_name, masks_multi): 
    from lang_sam import LangSAM

    sam_model = LangSAM()  
        
    prompt_images_path_config_dict = {}
    for prompt_idx in tqdm(os.listdir(source_images_path)):
        base_prompt_path = os.path.join(source_images_path,prompt_idx)
        # if we have the json file skip 
        if prompt_idx.endswith('.json'):
            continue
        seeds = os.listdir(base_prompt_path)
        for seed in seeds:
            with open(os.path.join(base_prompt_path,str(seed),'run_config.json'), 'r') as f:
                models_run_config = json.loads(f.read())      
            image_path = ''
            filenames_list = os.listdir(os.path.join(base_prompt_path,str(seed),model_name))
            if filenames_list == []:
                continue
            else:
                for file_name in filenames_list:
                    if file_name.startswith('mask'):
                        continue
                    else:
                        filename = file_name
                # filename = filenames_list[0]
            if model_name == 'ba_1_5':
                filename = f'sample_0/{seed}_0.png'
            elif model_name == 'rich_text_1_5':
                filename = f'seed{seed}_rich.jpg'
            elif model_name == 'ae_1_4':
                for file in os.listdir(os.path.join(base_prompt_path,str(seed),model_name)):
                    if (file.endswith('png')) and (not file.startswith('mask')):
                        filename = file
            elif model_name == 'ssd_1_4':
                original_prompt = models_run_config['original_prompt']
                filename = f'samples/00001-1-{original_prompt}.jpg'
            if os.path.exists(os.path.join(base_prompt_path,str(seed),model_name,filename)):
                image_path = os.path.join(base_prompt_path,str(seed),model_name,filename)

            if image_path=='':
                continue
            else: 
                missing_objects = ''
                object_name1, object_name2 = list(models_run_config['objects_colors_dict'].keys())
                mask1, mask2 = get_object_mask(sam_model,image_path, object_name1, object_name2, masks_multi)
                if isinstance(mask1,np.ndarray): 
                    intersection = np.sum(mask1 & mask2) 
                    union = np.sum(mask1 | mask2)
                    iou = intersection / union
                    if iou>=0.5: # object are too overlapping
                        print(f"skipping {image_path} objects are too overlapping iou {iou}")
                        continue
                    
                else:
                    missing_objects = object_name1 + ' ' + object_name2
                    print(f'skipping {image_path} missing one of {missing_objects} on image')
                    continue
            
            if masks_multi:
                mask1_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name1}.png')
                mask2_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name2}.png')
            else:
                mask1_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name1}.png')
                mask2_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name2}.png')
            save_image(torch.Tensor(mask1),mask1_path)
            save_image(torch.Tensor(mask2),mask2_path)
            
            dict_key = prompt_idx
            if len(seeds)>1:
                dict_key = f'{prompt_idx}_{seed}'
            single_run_dict = {dict_key :
                                            {
                                                'model_name' : model_name,
                                                'original_prompt' : models_run_config['original_prompt'],
                                                'simplified_text_prompt' : models_run_config['simplified_text_prompt'],
                                                'full_text_prompt' : models_run_config['full_text_prompt'],
                                                'color_object_dict' : models_run_config['objects_colors_dict'],
                                                'image_path' : image_path,
                                                f'mask_{object_name1}' : mask1_path,
                                                f'mask_{object_name2}' : mask2_path
                                            }
                                            }
            
            if "ref_color" in list(models_run_config.keys()): #"compare_color"
                single_run_dict[dict_key]['ref_color'] = models_run_config["ref_color"]
                single_run_dict[dict_key]['compare_color'] = models_run_config["compare_color"]
                
            prompt_images_path_config_dict.update(single_run_dict)
    # del sam_model
    # del mask1
    # del mask2
    # torch.cuda.empty_cache()
    return prompt_images_path_config_dict


def real_images_editing(input_dir, masks_multi):
    from lang_sam import LangSAM

    sam_model = LangSAM()  
    
    color_terms_rgb_dict = {}
    with open('ColorPrompts_benchmark/html_color_dict.json', 'r') as f:
        color_terms_rgb_dict = json.load(f)
    
    prompt_images_path_config_dict_real = {}
    for image_folder in tqdm(os.listdir(input_dir)):
        if image_folder.endswith('.json') or image_folder in ['close', 'distant', 'tmp']: #ignore not desired files/folders
            continue
        base_prompt_path = os.path.join(input_dir,image_folder)
        with open(f'{base_prompt_path}/color_config.json', 'r') as f:
            color_config = json.load(f)
        
        two_objects = len(color_config['color_list'][0])==2
        image_path = f'{base_prompt_path}/{image_folder}.png'
        if not os.path.exists(image_path):
            print(f'{base_prompt_path} does not have image_source')
            continue
        
        for colors in color_config['color_list']:
            if two_objects: 
                first_color, second_color = colors
                missing_objects = ''
                        
                prompt_splitted = image_folder.split('_')
                original_prompt = " ".join(prompt_splitted)
                first_object = prompt_splitted[1]
                second_object = prompt_splitted[4]

                first_base_color = color_terms_rgb_dict[first_color]['base_color_name']
                second_base_color = color_terms_rgb_dict[second_color]['base_color_name']
                cic_prompt = f'a {first_base_color} {first_object} and a {second_base_color} {second_object}'
                objects_colors_dict = {first_object: color_terms_rgb_dict[first_color], second_object: color_terms_rgb_dict[second_color]}

                image_run_config = {
                                    'seed' : 0,
                                    'original_prompt' : original_prompt,
                                    'simplified_text_prompt' : f'a {first_object} and a {second_object}',
                                    'full_text_prompt' : cic_prompt,
                                    'objects_colors_dict' : objects_colors_dict,
                                }

                object_name1, object_name2 = list(image_run_config['objects_colors_dict'].keys())
                mask1, mask2 = get_object_mask(sam_model,image_path, object_name1, object_name2, masks_multi)
                if isinstance(mask1,np.ndarray): 
                    intersection = np.sum(mask1 & mask2) 
                    union = np.sum(mask1 | mask2)
                    iou = intersection / union
                    if iou>=0.5: # object are too overlapping
                        print(f"skipping {image_path} objects are too overlapping iou {iou}")
                        continue
                    
                else:
                    missing_objects = object_name1 + ' ' + object_name2
                    print(f'skipping {image_path} missing one of {missing_objects} on image')
                    continue
                
                if masks_multi:
                    mask1_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name1}.png')
                    mask2_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name2}.png')
                else:
                    mask1_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name1}.png')
                    mask2_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name2}.png')
                save_image(torch.Tensor(mask1),mask1_path)
                save_image(torch.Tensor(mask2),mask2_path)
                
                dict_key = '_'.join(image_folder.split(' '))
                single_run_dict = { dict_key :
                                                {
                                                    'model_name' : 'real',
                                                    'original_prompt' : image_run_config['original_prompt'],
                                                    'simplified_text_prompt' : image_run_config['simplified_text_prompt'],
                                                    'full_text_prompt' : image_run_config['full_text_prompt'],
                                                    'color_object_dict' : image_run_config['objects_colors_dict'],
                                                    'image_path' : image_path,
                                                    f'mask_{object_name1}' : mask1_path,
                                                    f'mask_{object_name2}' : mask2_path,
                                                    'simplified_text_prompt_idx' : [[2],[5]],
                                                    'full_text_prompt_idx' : [[2,3],[6,7]]
                                                    
                                                }
                                                }
            else:
                first_color = colors[0]
                missing_objects = ''
                        
                prompt_splitted = image_folder.split('_')
                original_prompt = " ".join(prompt_splitted)
                first_object = prompt_splitted[1]

                first_base_color = color_terms_rgb_dict[first_color]['base_color_name']
                cic_prompt = f'a {first_base_color} {first_object}'
                objects_colors_dict = {first_object: color_terms_rgb_dict[first_color]}

                image_run_config = {
                                    'seed' : 0,
                                    'original_prompt' : original_prompt,
                                    'simplified_text_prompt' : f'a {first_object}',
                                    'full_text_prompt' : cic_prompt,
                                    'objects_colors_dict' : objects_colors_dict,
                                }

                object_name1 = list(image_run_config['objects_colors_dict'].keys())[0]
                mask1 = get_object_mask(sam_model,image_path, object_name1, object_name1, masks_multi)
                mask1 = mask1[0]
                if isinstance(mask1,np.ndarray): 
                    pass
                else:
                    missing_objects = object_name1 
                    print(f'skipping {image_path} missing one of {missing_objects} on image')
                    continue
                
                if masks_multi:
                    mask1_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name1}.png')
                else:
                    mask1_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name1}.png')
                save_image(torch.Tensor(mask1),mask1_path)
                
                dict_key = '_'.join(image_folder.split(' '))
                single_run_dict = { f'{dict_key}_{first_color}' :
                                                {
                                                    'model_name' : 'real',
                                                    'original_prompt' : image_run_config['original_prompt'],
                                                    'simplified_text_prompt' : image_run_config['simplified_text_prompt'],
                                                    'full_text_prompt' : image_run_config['full_text_prompt'],
                                                    'color_object_dict' : image_run_config['objects_colors_dict'],
                                                    'image_path' : image_path,
                                                    f'mask_{object_name1}' : mask1_path,
                                                    'simplified_text_prompt_idx' : [[2]],
                                                    'full_text_prompt_idx' : [[2,3]]
                                                }
                                                }
            prompt_images_path_config_dict_real.update(single_run_dict)

    return prompt_images_path_config_dict_real

def FLUX_preprocessing_ver2(flux_dir,masks_multi):
    from lang_sam import LangSAM

    sam_model = LangSAM()  
    
    color_terms_rgb_dict = {}
    with open('colorPrompts_benchmark/html_color_dict.json', 'r') as f:
        color_terms_rgb_dict = json.load(f)
    
    prompt_images_path_config_dict_close = {}
    prompt_images_path_config_dict_distant = {}
    for image_folder in tqdm(os.listdir(flux_dir)):
        if image_folder.endswith('.json') or image_folder in ['close', 'distant', 'tmp']: #ignore not desired files/folders
            continue
        base_prompt_path = os.path.join(flux_dir,image_folder)
        with open(f'{base_prompt_path}/config.json', 'r') as f:
            image_config = json.load(f)
            
        image_path = f'{base_prompt_path}/image_source.png'
        if not os.path.exists(image_path):
            print(f'{base_prompt_path} does not have image_source')
            continue
        
        missing_objects = ''
        
        prompt_splitted = image_folder.split(' ')
        first_color = prompt_splitted[1]
        first_object = prompt_splitted[3]
        second_color = prompt_splitted[6]
        second_object = prompt_splitted[8]
        first_base_color = color_terms_rgb_dict[first_color]['base_color_name']
        second_base_color = color_terms_rgb_dict[second_color]['base_color_name']
        cic_prompt = f'a {first_base_color} {first_object} and a {second_base_color} {second_object}'
        objects_colors_dict = {first_object: color_terms_rgb_dict[first_color], second_object: color_terms_rgb_dict[second_color]}

        image_run_config = {
                            'seed' : 0,
                            'original_prompt' : image_folder,
                            'simplified_text_prompt' : f'a {first_object} and a {second_object}',
                            'full_text_prompt' : cic_prompt,
                            'objects_colors_dict' : objects_colors_dict,
                            'ref_color' : image_config['ref_color'],
                            'compare_color': image_config['compare_color']
                        }

        object_name1, object_name2 = list(image_run_config['objects_colors_dict'].keys())
        mask1, mask2 = get_object_mask(sam_model,image_path, object_name1, object_name2, masks_multi)
        if isinstance(mask1,np.ndarray): 
            intersection = np.sum(mask1 & mask2) 
            union = np.sum(mask1 | mask2)
            iou = intersection / union
            if iou>=0.5: # object are too overlapping
                print(f"skipping {image_path} objects are too overlapping iou {iou}")
                continue
            
        else:
            missing_objects = object_name1 + ' ' + object_name2
            print(f'skipping {image_path} missing one of {missing_objects} on image')
            continue
        
        if masks_multi:
            mask1_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name1}.png')
            mask2_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name2}.png')
        else:
            mask1_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name1}.png')
            mask2_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name2}.png')
        save_image(torch.Tensor(mask1),mask1_path)
        save_image(torch.Tensor(mask2),mask2_path)
        
        dict_key = '_'.join(image_folder.split(' '))
        single_run_dict = { dict_key :
                                        {
                                            'model_name' : model_name,
                                            'original_prompt' : image_run_config['original_prompt'],
                                            'simplified_text_prompt' : image_run_config['simplified_text_prompt'],
                                            'full_text_prompt' : image_run_config['full_text_prompt'],
                                            'color_object_dict' : image_run_config['objects_colors_dict'],
                                            'image_path' : image_path,
                                            f'mask_{object_name1}' : mask1_path,
                                            f'mask_{object_name2}' : mask2_path
                                        }
                                        }
        
        if "ref_color" in list(image_run_config.keys()): #"compare_color"
            single_run_dict[dict_key]['ref_color'] = image_run_config["ref_color"]
            single_run_dict[dict_key]['compare_color'] = image_run_config["compare_color"]
        
        if image_config['set'] == 'close':
            prompt_images_path_config_dict_close.update(single_run_dict)
        elif image_config['set'] == 'distant':
            prompt_images_path_config_dict_distant.update(single_run_dict)
        else:
            raise ValueError

    return prompt_images_path_config_dict_close, prompt_images_path_config_dict_distant

def FLUX_prepocessing(flux_dir_dict):
    html_color_dict_path = 'color_space_analyze/color_promot_benchmark/html_color_dict.json'
    color_terms_rgb_dict = {}
    with open(html_color_dict_path, 'r') as f:
        color_terms_rgb_dict = json.load(f)
    
    flux_dir = flux_dir_dict['flux_dir']
    prompt_images_path_config_list = []
    all_images_names = os.listdir(flux_dir)
    for image_name in all_images_names:
        image_path = os.path.join(flux_dir,image_name)
        original_generated_prompt = image_name.split('.png')[0].replace('_',' ')
        prompt_splitted = original_generated_prompt.split(' ')
        first_color = prompt_splitted[1]
        first_object = prompt_splitted[3]
        second_color = prompt_splitted[6]
        second_object = prompt_splitted[8]
        
        first_base_color = color_terms_rgb_dict[first_color]['base_color_name']
        second_base_color = color_terms_rgb_dict[second_color]['base_color_name']
        
        cic_prompt = f'a {first_base_color} {first_object} and a {second_base_color} {second_object}'
        objects_colors_dict = {first_object: color_terms_rgb_dict[first_color], second_object: color_terms_rgb_dict[second_color]}
        prompt_images_path_config_list.append(
        {
            'model_name' : 'FLUX',
            'original_prompt' : original_generated_prompt,
            'simplified_text_prompt' :  f'a {first_object} and a {second_object}',
            'full_text_prompt' : cic_prompt,
            'color_object_dict' : objects_colors_dict,
            'image_path' : image_path
        }
        )
    
    return flux_dir_dict, prompt_images_path_config_list, None


models_data_paths = \
{
    "sd_1_4" : [
        "/storage/shayshomerchai/projects/color-edit/outputs/repo/ColorEdit/color_promot_benchmark/prompts_our_color_with_ref/sd_1_4/close/close/20251121_093126_sd_1_4"
    ]
}

total_dict = {'ae_1_4':[], 'rich_text_1_5': [], 'sd_1_4' : [], 'sd_1_5' : [],
              'sd_2_1' : [], 'syngen_1_4' : [], 'ba_1_5' : [], 'FLUX': [], 'real_imgs': [],
              'ssd_1_4' : [], 'gemini' : [], 'grok' : [], 'openai_4o' : [], 'flux' : [], 'syngen' : []}
masks_multi = True
for model_name, data_paths in tqdm(models_data_paths.items()):
    for data_path in data_paths:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        if model_name=='FLUX':
            prompt_images_path_config_dict_close, prompt_images_path_config_dict_distant = FLUX_preprocessing_ver2(data_path, masks_multi)
            if masks_multi:
                json_file_name = 'prompt_images_path_with_masks_multi_config_close.json'
            else:
                json_file_name = 'prompt_images_path_with_masks_config_close.json'
            with open(f'{data_path}/{json_file_name}', 'w') as f:
                json.dump(prompt_images_path_config_dict_close, f, indent=4)
                
            if masks_multi:
                json_file_name = 'prompt_images_path_with_masks_multi_config_distant.json'
            else:
                json_file_name = 'prompt_images_path_with_masks_config_distant.json'
            with open(f'{data_path}/{json_file_name}', 'w') as f:
                json.dump(prompt_images_path_config_dict_distant, f, indent=4)    
            
            total_dict[model_name].extend([len(prompt_images_path_config_dict_close.keys()),len(prompt_images_path_config_dict_distant.keys())])
            timestrafter = time.strftime("%Y%m%d_%H%M%S")
            print(f'{timestr} , {timestrafter}')
        elif model_name=='real_imgs':
            prompt_images_path_config_dict_real = real_images_editing(data_path, masks_multi)
            if masks_multi:
                json_file_name = 'prompt_images_path_with_masks_multi_config_real.json'
            else:
                json_file_name = 'prompt_images_path_with_masks_config_real.json'
            with open(f'{data_path}/{json_file_name}', 'w') as f:
                json.dump(prompt_images_path_config_dict_real, f, indent=4)
            
            total_dict[model_name].extend([len(prompt_images_path_config_dict_real.keys())])
            timestrafter = time.strftime("%Y%m%d_%H%M%S")
            print(f'{timestr} , {timestrafter}')
        else:
            prompt_images_path_config_dict = prepare_run_configs_for_editing_base_on_dir(data_path, model_name, masks_multi)
            if masks_multi:
                json_file_name = 'prompt_images_path_with_masks_multi_config.json'
            else:
                json_file_name = 'prompt_images_path_with_masks_config.json'
            with open(f'{data_path}/{json_file_name}', 'w') as f:
                json.dump(prompt_images_path_config_dict, f, indent=4)
            total_dict[model_name].append(len(prompt_images_path_config_dict.keys()))
            timestrafter = time.strftime("%Y%m%d_%H%M%S")
            print(f'{timestr} , {timestrafter}')
print(total_dict)
