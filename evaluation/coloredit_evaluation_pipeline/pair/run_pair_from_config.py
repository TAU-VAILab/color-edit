import os
import sys
import argparse
import json

from organize_path import run_organize_pair
from pair_color_acc_best_with_swap import run_pair_color_acc_best_with_swap
from pair_color_acc import run_pair_color_acc
from eval_summary import run_eval_summary_pair

def main(run_config,config_path):
        
    if 'force_out_dir' in run_config.keys():
        out_dir = run_config['force_out_dir']
    else:
        out_dir = run_config['out_dir']
    
    import shutil
    os.makedirs(out_dir,exist_ok=True)
    shutil.copy2(config_path,out_dir)
    
    root_pair_dirs = list(run_config['edited_root_pair_dirs'].values())
    if run_config.get('editing_model_name',False):
        organized_path = run_organize_pair(root_pair_dirs, out_dir, reference_editing=True, editing_model_name=run_config['editing_model_name'])
    else:
        organized_path = run_organize_pair(root_pair_dirs, out_dir)
    if run_config.get('with_swap',False):
        evaluation_path = run_pair_color_acc_best_with_swap(organized_path, out_dir)
    else:
        evaluation_path = run_pair_color_acc(organized_path, out_dir)
    run_eval_summary_pair(evaluation_path, out_dir, with_swap=run_config.get('with_swap',False))
    
if __name__ == '__main__':
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='evaluation/coloredit_evaluation_pipeline/pair_colors_sd_2_1.json')
    args = parser.parse_args()
    
    run_config_dict = {}
    with open(args.config, 'r') as f:
        run_config_dict = json.load(f)
    
    main(run_config_dict,args.config)