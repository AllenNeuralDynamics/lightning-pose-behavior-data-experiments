import argparse
import os, sys
from glob import glob
import subprocess
from pathlib import Path
import shutil
from datetime import datetime
from funcs import Logger
import yaml


############################# Han data #########################

scorer_name="Han_behavior_data"
project_folder_name="Foraging_Bot-Han_Lucas-2022-04-27"

video_name="bottom_face_1-0000"
video_type='.avi'
DLC_dir=f"/root/capsule/data/s3_video/DLC_projects/{project_folder_name}"
data_dir=f"/root/capsule/scratch/DLC_dataset_for_SLEAP/{scorer_name}/{project_folder_name}"

video_path=f"{DLC_dir}/videos/*{video_name}{video_type}"
print(video_path)
config_dir=f"/root/capsule/scratch/DLC_dataset_for_LP/{scorer_name}/{project_folder_name}/"

##########################################################
# for train_data_name in ['CollectedData_all_10perc', 
#                         'CollectedData_all_20perc', 
#                         'CollectedData_all_30perc', 
#                         'CollectedData_all_40perc', 
#                         'CollectedData_all_50perc', 
#                         'CollectedData_all_60perc', 
#                         'CollectedData_all_70perc',
#                         'CollectedData_all_80perc', 
#                         'CollectedData_all_90perc']:

for train_data_name in ['CollectedData_all_1perc', 
                        'CollectedData_all_2perc']:
    res_dir=f"/root/capsule/scratch/LP_outputs/"
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    res_dir=f"{res_dir}/{scorer_name}/"
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    res_dir=f"{res_dir}/{project_folder_name}"
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    res_dir=f"{res_dir}/{train_data_name}/"
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    ##########################################################

    '''
    Save training/testing log 
    '''
    log_dir =  os.path.join(res_dir,'logs')
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    testing_log = os.path.join(log_dir, "pipeline.log")
    testing_error_log = os.path.join(log_dir, "pipeline_error.log")

    if os.path.exists(testing_log):
        os.remove(testing_log)
    if os.path.exists(testing_error_log):
        os.remove(testing_error_log)

    sys.stdout = Logger(testing_log, sys.stdout)
    sys.stderr = Logger(testing_error_log, sys.stderr)     # redirect std err, if necessary
    print(testing_log)
    print(testing_error_log)

    start_time = datetime.now()

    print("--"*20)
    print("config directory:", config_dir)
    print("config file:", f"{train_data_name}.config.yaml")
    print("Result directory:", res_dir)

    print("--"*20)

    ############################################################################
    ########### copy testing video files  for the unsupervised part. ##########
    config_file = os.path.join(config_dir, f"{train_data_name}.config.yaml")
    print(config_file)
    with open(config_file, 'r') as file:
        param_updated = yaml.safe_load(file)
        # print(param_updated.keys())

    video_dir = param_updated["data"]["video_dir"]
    # Path(video_dir).mkdir(parents=True, exist_ok=True)

    save_dir = "/root/capsule/scratch/behavior_data/"
    out_file = save_dir + scorer_name + '.yaml'
    with open(out_file, 'r') as file:
        data_cfg = yaml.safe_load(file)

    out_file = save_dir + scorer_name + '_mice_inf.yaml'
    with open(out_file, 'r') as file:
        data_mice_cfg = yaml.safe_load(file)
        
    video_names = data_cfg["projects"][project_folder_name]['video_sets'].keys()
    print("video_names:", video_names, len(video_names))

    all_video_dir = param_updated["eval"]["test_videos_directory"]
    print(f"Testing video dir: {all_video_dir}")
    print(f"Training video dir: {video_dir}")

    ###########################################################################

    subprocess.run([
        "python", "/root/capsule/scratch/lightning-pose/scripts/train_hydra.py", 
            f"hydra.run.dir={res_dir}",
            f"--config-path={config_dir}",
            f"--config-name={train_data_name}.config.yaml"
        ])

    # if os.path.exists(video_dir):
    #     shutil.rmtree(video_dir)

    ##########################################################

    end_time = datetime.now()

    print('\nDuration: {}'.format(end_time - start_time))
        

# src = "/root/capsule/scratch/LP_outputs/"
# dst = "/root/capsule/results/LP_outputs/"
# if os.path.exists(dst):
#     shutil.rmtree(dst)
#     shutil.copytree(src, dst)


