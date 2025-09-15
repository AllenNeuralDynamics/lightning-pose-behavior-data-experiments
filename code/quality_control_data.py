import yaml
from pathlib import Path
from glob import glob
import os, sys
import collections
from pprint import pprint
import json
from omegaconf import DictConfig, OmegaConf
from organize_data_funcs import edit_cfg_file_subset_traindata, edit_cfg_file_leave_mouse_out, get_project_org, generate_collected_data_csv, edit_cfg_file, edit_cfg_file_leave_mouse_out_inlcude_oneframe
import numpy as np
from funcs import Logger
import shutil

import pandas as pd
]

# save project organization to yaml file
save_pred_dir_tp = "/root/capsule/scratch/"
save_dir = save_pred_dir_tp + '/behavior_data/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'
LP_config_template = "/root/capsule/scratch/lightning-pose/scripts/configs/config_toy-Handataset.yaml"

################################################################################
# Marton two-view dataset, generate training file, combine multiple cvs files
################################################################################


scorer_name = "Marton_behavior_data"


################################################################################
# Marton single-view dataset extracted from two-view dataset.
################################################################################

def get_bodypart(df):
    df.head()
    print(df.shape)
    bodyparts = [b[0] for b in df.columns if b[1] == "x"]
    print("bodyparts:", bodyparts)
    n_bodyparts = len(bodyparts)
    print("n_bodyparts:", n_bodyparts)

# proj_name = "BCI_bottom_2022"
# LP_proj_folder = f"/root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/{proj_name}/"
# label_file_name = "CollectedData_all_bottom_view.csv"
# file_list = [f"{LP_proj_folder}/{label_file_name}"]
# DLC_folder = "/root/capsule/data/Marton_behavior_data/BCI_bottom_2022_10_06/"

# proj_name = "BCI_side_2022"
# LP_proj_folder = f"/root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/{proj_name}/"
# label_file_name = "CollectedData_all_side_view.csv"
# file_list = [f"{LP_proj_folder}/{label_file_name}"]
# DLC_folder = "/root/capsule/data/Marton_behavior_data/BCI_side_2022_08_12/"


# # /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/BCI_side&bottom_2022_no_pole/CollectedData_all.csv
# proj_name = "BCI_side&bottom_2022_no_pole"
# LP_proj_folder = f"/root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/{proj_name}/"
# label_file_name = "CollectedData_all.csv"
# file_list = [f"{LP_proj_folder}/{label_file_name}"]
# DLC_folder = f"/root/capsule/data/Marton_behavior_data/{proj_name}/"


# /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/BCI_side&bottom_2023_Sept/CollectedData_all.csv

file_list = [f"{LP_proj_folder}/{label_file_name}"]
DLC_folder = f"/root/capsule/data/Marton_behavior_data/{proj_name}/"
project = "BCI_side&bottom_2023_Sept"
# "BCI_side&bottom_bottom_only_2023_Sept"
# "BCI_side&bottom_side_only_2023_Sept"


for proj_name in ["BCI_side&bottom_2023_Sept"]:
    LP_proj_folder = f"/root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data/{proj_name}/"
    label_file_name = "CollectedData_all.csv"
    file = f"{LP_proj_folder}/{label_file_name}"

    print("\n")
    print(file)
    df = pd.read_csv(file, header=[0, 1], index_col=0)
    get_bodypart(df)

    df = pd.read_csv(file, )
    print(df.columns)

    selected_frames = df['Unnamed: 0'].tolist()
    selected_frames = [x for x in selected_frames if str(x) != 'nan']
    # print(f"selected_frames: {selected_frames}, {len(selected_frames)}")
    print(f"selected_frames: {len(selected_frames)}")
    
    # find the videos that contains training data.
    video_names = []
    for cur_frame in selected_frames:
        video_names.append(cur_frame.split('/')[1])
    video_names = list(set(video_names))
    print("------"*30)
    print("Find the videos that contains training data .....", video_names, len(video_names))
    print(f"The number of the selected labeled frames: {len(selected_frames)}")
    print("------"*30)
    print("")


