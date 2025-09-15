import argparse
import os, sys
from glob import glob
import subprocess
from pathlib import Path
import shutil
from datetime import datetime
from funcs import Logger
import yaml

from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
    get_videos_in_dir,
)


############################# Marton data #########################

scorer_name="Marton_behavior_data_modified"
project_folder_name = "BCI_side&bottom_2023_Sept"
# /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data_modified/BCI_side&bottom_2023_Sept
DLC_dir=f"/root/capsule/data/{scorer_name}/{project_folder_name}"
config_dir=f"/root/capsule/scratch/DLC_dataset_for_LP/{scorer_name}/{project_folder_name}/"


##########################################################
# /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/BCI_side&bottom_2022_no_pole/CollectedData_all_pca_single_multiview.config.yaml
# /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/BCI_side&bottom_2022_no_pole/CollectedData_all_pca_singleview.config.yaml

# for train_data_name in ['CollectedData_all_pca_single_multiview', 'CollectedData_all_pca_singleview', 'CollectedData_all_supervised', ]:
# 'BCI29_excluded_pca_singleview', 'BCI26_excluded_pca_singleview'
for project_folder_name in ["BCI_side&bottom_2023_Sept"]:
    for train_data_name in ['BCI29_excluded_pca_single_multiview', 'BCI26_excluded_pca_single_multiview', 'BCI29_excluded_supervised', 'BCI26_excluded_supervised']:
        DLC_dir=f"/root/capsule/data/{scorer_name}/{project_folder_name}"
        config_dir=f"/root/capsule/scratch/DLC_dataset_for_LP/{scorer_name}/{project_folder_name}/"

        # /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/BCI_side&bottom_2022_no_pole/CollectedData_all_supervised.config.yaml
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

        train_csv_file = param_updated["data"]["csv_file"]
        print(f"Training csv file: {train_csv_file}")

        video_dir = param_updated["data"]["video_dir"]
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        print(f"Training video dir: {video_dir}")

        all_video_dir = param_updated["eval"]["test_videos_directory"]
        print(f"Testing video dir: {all_video_dir}")

        # ###########################################################################
        # check if the prediction generated successfulyy.
        # for video_file in filenames[:]:
        #     video_name = video_file.split("/")[-1]
        #     out_file = f"{res_dir}/video_preds/labeled_videos/{video_name}"
        #     if not os.path.exists(out_file):
        #         print(f"{out_file} NOT EXIST")


        ###########################################################################
        # start training models
        subprocess.run([
            "python", "/root/capsule/scratch/lightning-pose/scripts/train_hydra.py", 
                f"hydra.run.dir=\"{res_dir}\"",
                f"--config-path={config_dir}",
                f"--config-name={train_data_name}.config.yaml"
            ])

        # if os.path.exists(video_dir):
        #     shutil.rmtree(video_dir)

        ##########################################################

        end_time = datetime.now()

        print('\nDuration: {}'.format(end_time - start_time))



for project_folder_name in ["BCI_side&bottom_side_only_2023_Sept", "BCI_side&bottom_bottom_only_2023_Sept"]:
    for train_data_name in ['BCI29_excluded_supervised', 'BCI26_excluded_supervised']:
        DLC_dir=f"/root/capsule/data/{scorer_name}/{project_folder_name}"
        config_dir=f"/root/capsule/scratch/DLC_dataset_for_LP/{scorer_name}/{project_folder_name}/"

        # /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/BCI_side&bottom_2022_no_pole/CollectedData_all_supervised.config.yaml
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

        train_csv_file = param_updated["data"]["csv_file"]
        print(f"Training csv file: {train_csv_file}")

        video_dir = param_updated["data"]["video_dir"]
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        print(f"Training video dir: {video_dir}")

        all_video_dir = param_updated["eval"]["test_videos_directory"]
        print(f"Testing video dir: {all_video_dir}")

        # ###########################################################################
        # check if the prediction generated successfulyy.
        # for video_file in filenames[:]:
        #     video_name = video_file.split("/")[-1]
        #     out_file = f"{res_dir}/video_preds/labeled_videos/{video_name}"
        #     if not os.path.exists(out_file):
        #         print(f"{out_file} NOT EXIST")


        ###########################################################################
        # start training models
        subprocess.run([
            "python", "/root/capsule/scratch/lightning-pose/scripts/train_hydra.py", 
                f"hydra.run.dir=\"{res_dir}\"",
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


