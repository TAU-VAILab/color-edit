import os
from tqdm import tqdm
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
sys.path.append('enviroment/src/')
sys.path.append('enviroment/src/AnySD/')

from torchvision.io import read_image
from anysd.src.model import AnySDPipeline, choose_expert
from anysd.train.valid_log import download_image
from anysd.src.utils import choose_book, get_experts_dir

def load_AnySD_pipe():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    expert_file_path = get_experts_dir(repo_id="WeiChow/AnySD") 
    book_dim, book = choose_book('all')
    task_embs_checkpoints = expert_file_path + "task_embs.bin"
    adapter_checkpoints = {
        "global": expert_file_path + "global.bin",
        "viewpoint": expert_file_path + "viewpoint.bin",
        "visual_bbox": expert_file_path + "visual_bbox.bin",
        "visual_depth": expert_file_path + "visual_dep.bin",
        "visual_material_transfer": expert_file_path + "visual_mat.bin",
        "visual_reference": expert_file_path + "visual_ref.bin",
        "visual_scribble": expert_file_path + "visual_scr.bin",
        "visual_segment": expert_file_path + "visual_seg.bin",
        "visual_sketch": expert_file_path + "visual_ske.bin",
    }

    pipeline = AnySDPipeline(adapters_list=adapter_checkpoints, task_embs_checkpoints=task_embs_checkpoints)
    return pipeline, device

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def run_AnySD_1editing(pipeline, device, source_image_path, target_prompt, output_path):
    target_prompt
    prompt_splitted = target_prompt.split(' ')
    color1 = prompt_splitted[1]
    object1 = prompt_splitted[2]

    import time 
    start = time.time()
    source_image = load_image(source_image_path, device)
    case1 = [
        {
            "edit": f"Change the color of the {object1} to {color1}",
            "edit_type": 'general',
            "image_file": source_image_path
        }
    ]
    
    for index, item in enumerate(tqdm(case1)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(item['image_file']),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_1_path = os.path.join(output_path, f"AnySD_1_image.png")
        images.save(anysd_1_path)
    end = time.time()
    print(end-start)
    return source_image*0.5+0.5, images


def run_AnySD_editing(pipeline, device, source_image_path, target_prompt, output_path):
    target_prompt
    prompt_splitted = target_prompt.split(' ')
    if len(prompt_splitted) >=13:
        source_image, images = run_AnySD_3editing(pipeline, device, source_image_path, target_prompt, output_path)
        return source_image, images
    elif len(prompt_splitted) <=6:
        source_image, images = run_AnySD_1editing(pipeline, device, source_image_path, target_prompt, output_path)
        return source_image, images
    
    color1 = prompt_splitted[1]
    object1 = prompt_splitted[3]
    color2 = prompt_splitted[6]
    object2 = prompt_splitted[8]
    
    source_image = load_image(source_image_path, device)
    case1 = [
        {
            "edit": f"Change the color of the {object1} to {color1}",
            "edit_type": 'general',
            "image_file": source_image_path
        }
    ]
    case2 = [
        {
            "edit": f"Change the color of the {object2} to {color2}",
            "edit_type": 'general',
        }
    ]
    # case_full = [
    #     {
    #         "edit": f"Change the color of the {object1} to {color1} and the {object2} to {color2}",
    #         "edit_type": 'general',
    #         "image_file": source_image_path
    #     }
    # ]
    case_full2 = [
        {
            "edit": f"Change the color of the {object1} to {color1} and the color of the {object2} to {color2}",
            "edit_type": 'general',
            "image_file": source_image_path
        }
    ]
    
    for index, item in enumerate(tqdm(case1)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(item['image_file']),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_1_path = os.path.join(output_path, f"AnySD_1_image.png")
        images.save(anysd_1_path)
    
    for index, item in enumerate(tqdm(case2)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(anysd_1_path),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_1_2_path = os.path.join(output_path, f"AnySD_1_2_image.png")
        images.save(anysd_1_2_path)
    
    # for index, item in enumerate(tqdm(case_full)):
    #     images = pipeline(
    #         prompt=item['edit'],
    #         original_image=download_image(item['image_file']),
    #         guidance_scale=3,
    #         num_inference_steps=100,
    #         original_image_guidance_scale=3,
    #         adapter_name="general",
    #     )[0]
    #     anysd_full_path = os.path.join(output_path, f"AnySD_image_full.png")
    #     images.save(anysd_full_path)
    
    for index, item in enumerate(tqdm(case_full2)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(item['image_file']),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_full_path = os.path.join(output_path, f"AnySD_image_full_color.png")
        images.save(anysd_full_path)
    
    return source_image*0.5+0.5, images

def run_AnySD_3editing(pipeline, device, source_image_path, target_prompt, output_path):
    target_prompt
    prompt_splitted = target_prompt.split(' ')
    color1 = prompt_splitted[1]#'cornflower blue'# 'cornflower blue' #prompt_splitted[1]a klein blue gift box tied with a mint green ribbon and a beige rose
    object1 = prompt_splitted[3]#'gift box' #prompt_splitted[3]
    color2 = prompt_splitted[6]#'mint green' #prompt_splitted[6]
    object2 = prompt_splitted[8]#'ribbon' #prompt_splitted[8]
    color3 = prompt_splitted[11]#'burgundy'#burgundy' #prompt_splitted[11]
    object3 = prompt_splitted[13]#'rose' #prompt_splitted[13]
    
    
    source_image = load_image(source_image_path, device)
    case1 = [
        {
            "edit": f"Change the color of the {object1} to {color1}",
            "edit_type": 'general',
            "image_file": source_image_path
        }
    ]
    case2 = [
        {
            "edit": f"Change the color of the {object2} to {color2}",
            "edit_type": 'general',
        }
    ]
    case3 = [
        {
            "edit": f"Change the color of the {object3} to {color3}",
            "edit_type": 'general',
        }
    ]
    # case_full = [
    #     {
    #         "edit": f"Change the color of the {object1} to {color1} and the {object2} to {color2}",
    #         "edit_type": 'general',
    #         "image_file": source_image_path
    #     }
    # ]
    case_full3 = [
        {
            "edit": f"Change the color of the {object1} to {color1} and the color of the {object2} to {color2} and the color of the {object3} to {color3}",
            "edit_type": 'general',
            "image_file": source_image_path
        }
    ]
    
    for index, item in enumerate(tqdm(case1)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(item['image_file']),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_1_path = os.path.join(output_path, f"AnySD_1_image.png")
        images.save(anysd_1_path)
    
    for index, item in enumerate(tqdm(case2)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(anysd_1_path),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_1_2_path = os.path.join(output_path, f"AnySD_1_2_image.png")
        images.save(anysd_1_2_path)
    
    for index, item in enumerate(tqdm(case3)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(anysd_1_2_path),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_1_2_3_path = os.path.join(output_path, f"AnySD_1_2_3_image.png")
        images.save(anysd_1_2_3_path)
    
    # for index, item in enumerate(tqdm(case_full)):
    #     images = pipeline(
    #         prompt=item['edit'],
    #         original_image=download_image(item['image_file']),
    #         guidance_scale=3,
    #         num_inference_steps=100,
    #         original_image_guidance_scale=3,
    #         adapter_name="general",
    #     )[0]
    #     anysd_full_path = os.path.join(output_path, f"AnySD_image_full.png")
    #     images.save(anysd_full_path)
    
    for index, item in enumerate(tqdm(case_full3)):
        print(item['edit'])
        images = pipeline(
            prompt=item['edit'],
            original_image=download_image(item['image_file']),
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]
        anysd_full_path = os.path.join(output_path, f"AnySD_image_full_color.png")
        images.save(anysd_full_path)
    
    return source_image*0.5+0.5, images