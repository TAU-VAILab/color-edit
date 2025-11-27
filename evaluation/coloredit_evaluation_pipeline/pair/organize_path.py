import os
import json
import os.path as osp

def run_organize_pair(root_pair_dirs,output_path,reference_editing=False, editing_model_name=''):
    os.makedirs(output_path,exist_ok=True)

    def get_data(folder_name):
        with open(os.path.join(folder_name, "run_config.json")) as f:
            run_config = json.load(f)
        objects = list(run_config["color_object_dict"].keys())
        color1_rgb, color2_rgb = run_config["color_object_dict"][objects[0]]['RGB'], run_config["color_object_dict"][objects[1]]['RGB']
        color1_name, color2_name = run_config["color_object_dict"][objects[0]]['color_name'], run_config["color_object_dict"][objects[1]]['color_name']
        prompt = run_config["original_prompt"]
        ref_color = run_config["ref_color"]
        campare_color = run_config["compare_color"]
        return objects[0], objects[1], color1_rgb, color2_rgb, color1_name, color2_name, ref_color, campare_color, prompt



    results_organized = []
    for path in root_pair_dirs:
        print(f'running organized to {path}')
        if reference_editing:
            if 'close' in path.split('/')[-1]:
                dataset_type = 'close'
            elif 'distant' in path.split('/')[-1]:
                dataset_type = 'distant'
            else:
                raise ValueError
        else:
            dataset_type = path.split('/')[-1].split("_")[0]
        if osp.isdir(path):
            for method_folder in os.listdir(path):
                method = method_folder
                method_folder_path = osp.join(path, method_folder)
                if osp.isdir(method_folder_path):
                    for index_folder in os.listdir(method_folder_path)[:20]:
                        index = index_folder
                        index_folder_path = osp.join(method_folder_path, index_folder)
                        if osp.isdir(index_folder_path):
                            mask_img_dir = osp.join(index_folder_path, "mask_image.png")
                            source_img_dir = osp.join(index_folder_path, "source_image.png")
                            if editing_model_name != '' :
                                if editing_model_name == 'anysd_1_2':
                                    edited_img_dir = osp.join(index_folder_path, f'AnySD_1_2_image.png')
                                elif editing_model_name == 'anysd_full':
                                    edited_img_dir = osp.join(index_folder_path, f'AnySD_image_full_color.png')
                                else:
                                    edited_img_dir = osp.join(index_folder_path, f'{editing_model_name}_image.png')
                            else:
                                edited_img_dir = osp.join(index_folder_path, "ColorEdit_image.png")
                            object1, object2, color1_rgb, color2_rgb, color1_name, color2_name, ref_color, campare_color, prompt = get_data(index_folder_path)
                            results_organized.append({
                                "type": dataset_type,
                                "method": method,
                                "index": index,
                                "index_folder_path": index_folder_path,
                                "mask_img_dir": mask_img_dir,
                                "source_img_dir": source_img_dir,
                                "edited_img_dir": edited_img_dir,
                                "object1": object1,
                                "object2": object2,
                                "color1_rgb": color1_rgb,
                                "color2_rgb": color2_rgb,
                                "color1_name": color1_name,
                                "color2_name": color2_name,
                                "ref_color": ref_color,
                                "campare_color": campare_color,
                                "prompt": prompt
                            })
    organized_pair_results_path = osp.join(output_path, "pair_results_organized.json")
    with open(organized_pair_results_path, "w") as f:
        json.dump(results_organized, f)
    
    return organized_pair_results_path
