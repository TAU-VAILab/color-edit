
import os
import sys
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import PIL

def get_object_mask(model, image_path, object_name1, object_name2, object_name3, masks_multi=True):
    try:
        image_pil = PIL.Image.open(image_path).convert("RGB")
        results = model.predict([image_pil, image_pil, image_pil], [object_name1, object_name2, object_name3])
        # Check if masks exist
        if len(results[0]["masks"]) == 0 or len(results[1]["masks"]) == 0 or len(results[2]["masks"]) == 0:
            return None, None, None  # Indicate missing masks
        # Convert masks to boolean
        if masks_multi:
            mask1 = np.bitwise_or.reduce(results[0]["masks"].astype(bool),axis=0)
            mask2 = np.bitwise_or.reduce(results[1]["masks"].astype(bool),axis=0)
            mask3 = np.bitwise_or.reduce(results[2]["masks"].astype(bool),axis=0)
        else:
            mask1 = results[0]["masks"][0].astype(bool).squeeze()
            mask2 = results[1]["masks"][0].astype(bool).squeeze()
            mask3 = results[2]["masks"][0].astype(bool).squeeze()
        if len(mask1.shape) != 2:
            mask1 = mask1[0]
        if len(mask2.shape) != 2:
            mask2 = mask2[0]
        if len(mask3.shape) != 2:
            mask3 = mask3[0]
        return mask1, mask2, mask3
    except Exception as e:
        print(f"Error during mask prediction: {e}")
        return None, None, None

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
            elif model_name == 'flux_dev':
                filename = 'flux_dev.png'
            if os.path.exists(os.path.join(base_prompt_path,str(seed),model_name,filename)):
                image_path = os.path.join(base_prompt_path,str(seed),model_name,filename)

            if image_path=='':
                continue
            else: 
                missing_objects = ''
                object_name1, object_name2, object_name3 = list(models_run_config['objects_colors_dict'].keys())
                mask1, mask2, mask3 = get_object_mask(sam_model,image_path, object_name1, object_name2, object_name3, masks_multi)
                if isinstance(mask1,np.ndarray): 
                    intersection = np.sum(mask1 & mask2) 
                    union = np.sum(mask1 | mask2)
                    iou = intersection / union
                    if iou>=0.5: # object are too overlapping
                        print(f"skipping {image_path} objects are too overlapping iou {iou}")
                        continue
                    intersection = np.sum(mask1 & mask3) 
                    union = np.sum(mask1 | mask3)
                    iou = intersection / union
                    if iou>=0.5: # object are too overlapping
                        print(f"skipping {image_path} objects are too overlapping iou {iou}")
                        continue
                    intersection = np.sum(mask2 & mask3) 
                    union = np.sum(mask2 | mask3)
                    iou = intersection / union
                    if iou>=0.5: # object are too overlapping
                        print(f"skipping {image_path} objects are too overlapping iou {iou}")
                        continue
                else:
                    missing_objects = object_name1 + ' ' + object_name2 + ' ' + object_name3
                    print(f'skipping {image_path} missing one of {missing_objects} on image')
                    continue
            
            if masks_multi:
                mask1_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name1}.png')
                mask2_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name2}.png')
                mask3_path = os.path.join(os.path.dirname(image_path),f'mask_multi_{object_name3}.png')
            else:
                mask1_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name1}.png')
                mask2_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name2}.png')
                mask3_path = os.path.join(os.path.dirname(image_path),f'mask_{object_name3}.png')
            save_image(torch.Tensor(mask1),mask1_path)
            save_image(torch.Tensor(mask2),mask2_path)
            save_image(torch.Tensor(mask3),mask3_path)
            
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
                                                f'mask_{object_name2}' : mask2_path,
                                                f'mask_{object_name3}' : mask3_path
                                            }
                                            }
            
            if "ref_color" in list(models_run_config.keys()): #"compare_color"
                single_run_dict[dict_key]['ref_color'] = models_run_config["ref_color"]
                single_run_dict[dict_key]['close_color'] = models_run_config["close_color"]
                single_run_dict[dict_key]['distant_color'] = models_run_config["distant_color"]
                
            prompt_images_path_config_dict.update(single_run_dict)
    # del sam_model
    # del mask1
    # del mask2
    # torch.cuda.empty_cache()
    return prompt_images_path_config_dict

models_data_paths = \
{
    "sd_2_1" : [
        "/storage/shayshomerchai/projects/synthesize_colors_in_context/outputs/color_promot_benchmark/final_benchmark1110/prompts_our_color_with_ref/3_colors_close_distant/sd_2_1/close_distant/20250511_102708_sd_2_1"
    ]
}
total_dict = {'ae_1_4':[], 'rich_text_1_5': [], 'sd_1_4' : [], 'sd_1_5' : [],
              'sd_2_1' : [], 'syngen_1_4' : [], 'ba_1_5' : [], 'FLUX': [], 'real_imgs': [],
              'ssd_1_4' : [], 'flux_dev' : [], 'gemini': [], 'grok': [], 'openai_4o': [], 'flux': []}
masks_multi = False
for model_name, data_paths in tqdm(models_data_paths.items()):
    for data_path in data_paths:
        timestr = time.strftime("%Y%m%d_%H%M%S")
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
