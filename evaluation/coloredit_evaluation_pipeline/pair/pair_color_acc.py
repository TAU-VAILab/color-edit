from PIL import Image
from lang_sam import LangSAM
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976
from colormath.color_diff import delta_e_cmc
import ast
import os
from tqdm import tqdm


model = LangSAM()

import numpy

def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)

def run_pair_color_acc(organized_path, output_path):

    with open(organized_path, "r") as f:
        data = json.load(f)

    def get_object_mask(image_path, object_name1, object_name2,masks_multi=True):
        try:
            image_pil = Image.open(image_path).convert("RGB")
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
            # mask1 = results[0]["masks"][0].astype(bool).squeeze()
            # mask2 = results[1]["masks"][0].astype(bool).squeeze()
            if len(mask1.shape) != 2:
                mask1 = mask1[0]
            if len(mask2.shape) != 2:
                mask2 = mask2[0]
            return mask1, mask2
        except Exception as e:
            print(f"Error during mask prediction: {e}")
            return None, None

    # Apply quantized colors to each masked area
    def apply_quantized_colors_to_image(masked_pixels, labels, mask, codebook, original_shape):
        quantized_img = np.zeros((original_shape[0] * original_shape[1], 3), dtype=np.uint8)
        quantized_img[mask.ravel()] = codebook[labels]
        return quantized_img.reshape(original_shape)

    # Quantize colors for each object
    def quantize_colors(n_colors, img, mask):
        w, h, d = original_shape = tuple(img.shape)
        assert d == 3

        image_array = np.reshape(img, (w * h, d))
        masked_pixels = image_array[mask.ravel()]

        image_array_sample = shuffle(masked_pixels, random_state=0, n_samples=1_000)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        labels = kmeans.predict(masked_pixels)
        quantized_img = apply_quantized_colors_to_image(masked_pixels, labels, mask, kmeans.cluster_centers_, original_shape)

        return quantized_img

    # Color representation class for LAB conversion and delta E distance calculation
    class ColorRepr:
        def __init__(self, rgb) -> None:
            self.rgb = rgb
            l, a, b = color.rgb2lab(np.array(rgb, dtype='float32') / 255)
            self.lab = LabColor(l, a, b)

        def compute_delta_e_cmc_distance(self, reference_color_rgb):
            l, a, b = color.rgb2lab(np.array(reference_color_rgb, dtype='float32') / 255)
            reference_color_lab = LabColor(l, a, b)
            delta_e_cmc_distance = delta_e_cmc(self.lab, reference_color_lab)
            # delta_e_cie1976(self.lab, reference_color_lab)
            return delta_e_cmc_distance

    # Main function to read, quantize, and find closest color for each object
    def read_image_quantize_and_find_best_match_to_colors(image_path, mask1, mask2, n_clusters_colors, color_tested_rgb_value1, color_tested_rgb_value2, threshold_percent, save_path="result.png", prompt="", method=""):
        image_arr = np.asarray(Image.open(image_path))
        
        # Quantize colors separately for each masked area
        image_arr_quantized1 = quantize_colors(n_clusters_colors, image_arr, mask1)
        image_arr_quantized2 = quantize_colors(n_clusters_colors, image_arr, mask2)
        
        # Process each mask with its respective color and threshold
        best_color_repr1, best_color_distance1, rgb_l2_diff1 = process_quantized_area(image_arr_quantized1, mask1, color_tested_rgb_value1, threshold_percent)
        best_color_repr2, best_color_distance2, rgb_l2_diff2 = process_quantized_area(image_arr_quantized2, mask2, color_tested_rgb_value2, threshold_percent)

        # Visualize results
        visualize_results(
            image_arr, 
            image_arr_quantized1, 
            image_arr_quantized2, 
            mask1, 
            mask2, 
            best_color_repr1, 
            best_color_repr2, 
            color_tested_rgb_value1, 
            color_tested_rgb_value2, 
            save_path, 
            prompt, 
            method,
            rgb_l2_diff1,
            rgb_l2_diff2
        )

        # Return results including RGB L2 differences
        return (
            (best_color_repr1, best_color_distance1, rgb_l2_diff1),
            (best_color_repr2, best_color_distance2, rgb_l2_diff2),
            image_arr,
            image_arr_quantized1,
            image_arr_quantized2
        )

    def process_quantized_area(image_arr_quantized, mask, color_tested_rgb_value, threshold_percent):
        try:
            image_hist_dict = {}
            colors_repr_list = []

            for i in range(image_arr_quantized.shape[0]):
                for j in range(image_arr_quantized.shape[1]):
                    if not mask[i, j]:
                        continue
                    rgb_value = image_arr_quantized[i, j].tolist()
                    rgb_string = str(rgb_value)
                    image_hist_dict[rgb_string] = image_hist_dict.get(rgb_string, 0) + 1

            masked_pixel_count = np.sum(mask)
            threshold_count = int(masked_pixel_count * threshold_percent)

            for rgb_string, rgb_count in image_hist_dict.items():
                if rgb_count < threshold_count:
                    continue
                colors_repr_list.append(ColorRepr(ast.literal_eval(rgb_string)))

            best_color_distance = float('inf')
            best_color_repr = None
            rgb_l2_diff = None

            for color_repr in colors_repr_list:
                curr_distance = color_repr.compute_delta_e_cmc_distance(color_tested_rgb_value)
                if curr_distance < best_color_distance:
                    best_color_distance = curr_distance
                    best_color_repr = color_repr
                    rgb_l2_diff = np.linalg.norm(np.array(color_repr.rgb) - np.array(color_tested_rgb_value))

            if best_color_repr is None:
                raise ValueError("No best-match color found.")

            return best_color_repr, best_color_distance, rgb_l2_diff

        except Exception as e:
            print(f"Error during quantized area processing: {e}")
            return None, None, None


    def generate_error_image(save_path, prompt, error_message):
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {error_message}\nPrompt: {prompt}", fontsize=14, ha='center', va='center', wrap=True)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


    def visualize_results(
        image_arr, 
        image_arr_quantized1, 
        image_arr_quantized2, 
        mask1, 
        mask2, 
        best_color_repr1, 
        best_color_repr2, 
        color_tested_rgb_value1, 
        color_tested_rgb_value2, 
        save_path="result.png", 
        prompt="", 
        method="", 
        rgb_l2_diff1=None, 
        rgb_l2_diff2=None
    ):
        matched_color1 = np.array(best_color_repr1.rgb)
        matched_color2 = np.array(best_color_repr2.rgb)
        
        # Create separate overlay images for each object
        overlay_image1 = image_arr.copy()
        overlay_image2 = image_arr.copy()
        
        # Highlight best-match color for object 1 in red
        mask_matched1 = (image_arr_quantized1 == matched_color1).all(axis=-1) & mask1
        overlay_image1[mask_matched1] = [255, 0, 0]  # Red for first object

        # Highlight best-match color for object 2 in blue
        mask_matched2 = (image_arr_quantized2 == matched_color2).all(axis=-1) & mask2
        overlay_image2[mask_matched2] = [255, 0, 0]  # Blue for second object

        # Create a single canvas with a grid layout for all images
        fig = plt.figure(figsize=(18, 18))
        fig.suptitle(f"Prompt: {prompt}\nMethod: {method}", fontsize=16, weight='bold', ha='center', va='top', y=0.95)
        
        # Original Image and Quantized Objects Overview
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.imshow(image_arr)
        ax1.set_title("Original Image")
        ax1.axis('off')

        ax2 = fig.add_subplot(3, 3, 2)
        ax2.imshow(image_arr_quantized1)
        ax2.set_title("Quantized Object 1")
        ax2.axis('off')

        ax3 = fig.add_subplot(3, 3, 3)
        ax3.imshow(image_arr_quantized2)
        ax3.set_title("Quantized Object 2")
        ax3.axis('off')

        # Detailed view for Object 1 with reference and best-match colors
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.imshow(image_arr_quantized1)
        ax4.set_title("Quantized Object 1")
        ax4.axis('off')

        ax5 = fig.add_subplot(3, 3, 5)
        ax5.imshow(overlay_image1)
        ax5.set_title(f"Object 1 Match\nDelta E: {best_color_repr1.compute_delta_e_cmc_distance(color_tested_rgb_value1):.2f}\nRGB L2: {rgb_l2_diff1:.2f}")
        ax5.axis('off')
        
        # Add reference and best-match color squares for Object 1
        color_card_ax1_ref = fig.add_axes([0.75, 0.55, 0.05, 0.05])  # Adjusted position for Ref Color 1
        color_card_ax1_ref.imshow(np.ones((10, 10, 3), dtype=np.uint8) * np.array(color_tested_rgb_value1, dtype=np.uint8))
        color_card_ax1_ref.axis('off')
        color_card_ax1_ref.set_title("Ref Color 1")

        color_card_ax1_match = fig.add_axes([0.81, 0.55, 0.05, 0.05])  # Adjusted position for Matched Color 1
        color_card_ax1_match.imshow(np.ones((10, 10, 3), dtype=np.uint8) * matched_color1)
        color_card_ax1_match.axis('off')
        color_card_ax1_match.set_title("Match Color 1")

        # Detailed view for Object 2 with reference and best-match colors
        ax6 = fig.add_subplot(3, 3, 7)
        ax6.imshow(image_arr_quantized2)
        ax6.set_title("Quantized Object 2")
        ax6.axis('off')

        ax7 = fig.add_subplot(3, 3, 8)
        ax7.imshow(overlay_image2)
        ax7.set_title(f"Object 2 Match\nDelta E: {best_color_repr2.compute_delta_e_cmc_distance(color_tested_rgb_value2):.2f}\nRGB L2: {rgb_l2_diff2:.2f}")
        ax7.axis('off')
        
        # Add reference and best-match color squares for Object 2
        color_card_ax2_ref = fig.add_axes([0.75, 0.27, 0.05, 0.05])  # Adjusted position for Ref Color 2
        color_card_ax2_ref.imshow(np.ones((10, 10, 3), dtype=np.uint8) * np.array(color_tested_rgb_value2, dtype=np.uint8))
        color_card_ax2_ref.axis('off')
        color_card_ax2_ref.set_title("Ref Color 2")

        color_card_ax2_match = fig.add_axes([0.81, 0.27, 0.05, 0.05])  # Adjusted position for Matched Color 2
        color_card_ax2_match.imshow(np.ones((10, 10, 3), dtype=np.uint8) * matched_color2)
        color_card_ax2_match.axis('off')
        color_card_ax2_match.set_title("Match Color 2")

        # Add RGB comparison text for Object 2
        ax2_rgb = fig.add_axes([0.75, 0.21, 0.1, 0.03])
        ax2_rgb.axis('off')
        ax2_rgb.text(0.5, 0.5, f"Ref: {color_tested_rgb_value2}\nMatch: {list(int(c) for c in matched_color2)}", ha='center', fontsize=10)

        # Save the final figure to a file
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()


    def main_inference(image_path, mask1, mask2, n_clusters_colors, color_tested_rgb_value1, color_tested_rgb_value2, threshold_percent, save_path="result.png", prompt="", method=""):
        """
        Runs the full pipeline for two masked objects with independent color quantization and matching.

        Returns:
        - Best matched colors and distances for each object, including RGB L2 differences.
        """
        try:
            # Check if masks are valid
            if mask1 is None or mask2 is None:
                generate_error_image(save_path, prompt, "No mask predicted for one or both objects.")
                return None

            # Execute the main function with the parameters provided
            (
                (best_color_repr1, best_color_distance1, rgb_l2_diff1),
                (best_color_repr2, best_color_distance2, rgb_l2_diff2),
                image_arr,
                image_arr_quantized1,
                image_arr_quantized2
            ) = read_image_quantize_and_find_best_match_to_colors(
                image_path=image_path,
                mask1=mask1,
                mask2=mask2,
                n_clusters_colors=n_clusters_colors,
                color_tested_rgb_value1=color_tested_rgb_value1,
                color_tested_rgb_value2=color_tested_rgb_value2,
                threshold_percent=threshold_percent,
                save_path=save_path,
                prompt=prompt,
                method=method,
            )

            # Check if best-match colors are found
            if best_color_repr1 is None or best_color_repr2 is None:
                generate_error_image(save_path, prompt, "No best-match color found for one or both objects.")
                return None

            return (best_color_repr1, best_color_distance1, rgb_l2_diff1), (best_color_repr2, best_color_distance2, rgb_l2_diff2)
        except Exception as e:
            print(f"Unexpected error during inference: {e}")
            generate_error_image(save_path, prompt, f"Unexpected error: {e}")
            return None



    def get_data(folder_name):
        with open(os.path.join(folder_name, "run_config.json")) as f:
            run_config = json.load(f)
        objects = list(run_config["color_object_dict"].keys())
        color1, color2 = run_config["color_object_dict"][objects[0]]['RGB'], run_config["color_object_dict"][objects[1]]['RGB']
        source_image_path = f"{index_folder_path}/source_image.png"
        edited_image_path = f"{index_folder_path}/ColorEdit_image.png"
        prompt = run_config["original_prompt"]
        return objects[0], objects[1], color1, color2, source_image_path, edited_image_path, prompt

    # Helper function to check mask overlap condition
    def is_significant_overlap(mask1, mask2):
        """
        Check if the smaller mask overlaps more than 90% with the larger mask.
        """
        intersection = np.sum(mask1 & mask2)  # Number of overlapping pixels
        smaller_mask_size = min(np.sum(mask1), np.sum(mask2))
        if smaller_mask_size == 0:  # Avoid division by zero
            return False
        overlap_ratio = intersection / smaller_mask_size
        return overlap_ratio > 0.9




    # root = "/share/elor/wp267/outputs/final_benchmark1108/results_1111"
    os.makedirs(output_path,exist_ok=True)

    n_clusters_colors = 5
    threshold_percent = 0.2 



    results = []
    # Initialize filter list
    filter_list = []


    for i in range(len(data)):#(1500, 1914): #1914 
        eval_data_item = data[i]
        updated_eval_data_item = eval_data_item
        ''' data_item be like
                                "type": type,
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
        '''
        method = eval_data_item["method"]
        print(method)
        index_folder_path = eval_data_item["index_folder_path"]
        object_name1, object_name2, color_tested_rgb_value1, \
        color_tested_rgb_value2, source_image_path, edited_image_path, prompt \
            = eval_data_item["object1"], eval_data_item["object2"], eval_data_item["color1_rgb"], eval_data_item["color2_rgb"], eval_data_item["source_img_dir"], eval_data_item["edited_img_dir"], eval_data_item["prompt"]


        ### Source
        # Get masks for the source image
        mask1, mask2 = get_object_mask(source_image_path, object_name1, object_name2,masks_multi=True)
        # Check for significant overlap in source image
        if mask1 is not None and mask2 is not None and is_significant_overlap(mask1, mask2):
            filter_list.append({"prompt": prompt, "objects": [object_name1, object_name2], "type": "source"})
            updated_eval_data_item["source_visual"] = "mask_overlap_err"
            continue  # Skip further processing for this folder

        # Run inference on source image
        source_results = main_inference(
            image_path=source_image_path,
            mask1=mask1,
            mask2=mask2,
            n_clusters_colors=n_clusters_colors,
            color_tested_rgb_value1=color_tested_rgb_value1,
            color_tested_rgb_value2=color_tested_rgb_value2,
            threshold_percent=threshold_percent,
            save_path=f"{index_folder_path}/result_multi_source.png", #result_source.png",
            prompt=prompt,
            method = method,
        )

        # Handle cases where source_results is None
        if source_results is None:
            updated_eval_data_item["source_visual"] = "no_best_match_err"
            continue  # Skip further processing for this folder

        (best_color_repr1, best_color_distance1, rgb_l2_diff1), (best_color_repr2, best_color_distance2, rgb_l2_diff2) = source_results
        updated_eval_data_item["source_visual"] = f"{index_folder_path}/result_multi_source.png" #result_source.png"
        updated_eval_data_item["source_best_color_repr1"] = best_color_repr1.rgb
        updated_eval_data_item["source_best_color_distance1"] = best_color_distance1
        updated_eval_data_item["source_rgb_l2_diff1"] = rgb_l2_diff1
        updated_eval_data_item["source_best_color_repr2"] = best_color_repr2.rgb
        updated_eval_data_item["source_best_color_distance2"] = best_color_distance2
        updated_eval_data_item["source_rgb_l2_diff2"] = rgb_l2_diff2


        ### Repeat for edited image
        # Get masks for the edited image
        mask1, mask2 = get_object_mask(edited_image_path, object_name1, object_name2,masks_multi=True)
        # Check for significant overlap in edited image
        if mask1 is not None and mask2 is not None and is_significant_overlap(mask1, mask2):
            filter_list.append({"prompt": prompt, "objects": [object_name1, object_name2], "type": "edited"})
            updated_eval_data_item["edited_visual"] = "mask_overlap_err"
            continue
        
        # Run inference on edited image
        edited_results = main_inference(
            image_path=edited_image_path,
            mask1=mask1,
            mask2=mask2,
            n_clusters_colors=n_clusters_colors,
            color_tested_rgb_value1=color_tested_rgb_value1,
            color_tested_rgb_value2=color_tested_rgb_value2,
            threshold_percent=threshold_percent,
            save_path=f"{index_folder_path}/result_multi_edited.png", #result_edited.png",
            prompt=prompt,
            method = f"{method} + Edit by ours",
        )

        # Handle cases where edited_results is None
        if edited_results is None:
            updated_eval_data_item["edited_visual"] = "no_best_match_err"
            continue

        (best_color_repr1, best_color_distance1, rgb_l2_diff1), (best_color_repr2, best_color_distance2, rgb_l2_diff2) = edited_results
        updated_eval_data_item["edited_visual"] = f"{index_folder_path}/result_multi_edited.png" #result_edited.png"
        updated_eval_data_item["edited_best_color_repr1"] = best_color_repr1.rgb
        updated_eval_data_item["edited_best_color_distance1"] = best_color_distance1
        updated_eval_data_item["edited_rgb_l2_diff1"] = rgb_l2_diff1
        updated_eval_data_item["edited_best_color_repr2"] = best_color_repr2.rgb
        updated_eval_data_item["edited_best_color_distance2"] = best_color_distance2
        updated_eval_data_item["edited_rgb_l2_diff2"] = rgb_l2_diff2

        # Append updated data item to results
        results.append(updated_eval_data_item)

    import pandas as pd    
    df_res = pd.DataFrame(results)   
    pickle_path = os.path.join(output_path, 'pair_color_results.pkl')
    df_res.to_pickle(pickle_path)

    # Save the dictionary to a file for later access if needed
    evaluation_results_path = os.path.join(output_path, 'evaluation_results.json')
    with open(evaluation_results_path, "w") as f:
        json.dump(results, f)

    # Save the filter list to a separate file
    filter_list_path = os.path.join(output_path, 'filter_list.json')
    with open(filter_list_path, "w") as f:
        json.dump(filter_list, f)

    return evaluation_results_path