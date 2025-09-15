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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def generate_collected_data_csv(data_dir, save_data_dir):
    ''' generate training/testing file
        convert DLC annotation file to LP training package '''
    
    print("Start generating collected data csv files ...... ")
    print("find all videos under ", data_dir)
    labeled_data_files = data_dir + '/labeled-data/*/CollectedData_*.csv'
    print(labeled_data_files)
    csv_files = glob(labeled_data_files)

    print("Total num of videos =", len(csv_files))

    # combine all the annotation data
    df1 = pd.read_csv(csv_files[0], sep = ',')
    # print(df1.head())
    if len(csv_files) > 1:
        combined_csv = [pd.read_csv(f, sep = ',').iloc[1:] for f in csv_files[1:] ]
        df = pd.concat( [df1] + combined_csv )
    else:
        df = df1

    if "Unnamed: 1" in df.columns and "Unnamed: 2" in df.columns:
        # drop "Unnamed: 1", "Unnamed: 2" columns for Han's behavior dataset, 
        df["scorer"] = df["scorer"].astype(str) + "/" + df["Unnamed: 1"].astype(str) +"/"+ df[ "Unnamed: 2"].astype(str)
        df = df.drop(["Unnamed: 1", "Unnamed: 2"], axis=1)
        df.at[0,'scorer'] = "bodyparts"
        df.at[1,'scorer'] = "coords"
    print(df.head())
    print("**"*20)
    TRAINING_LP_FILE = save_data_dir + '/CollectedData_all.csv'
    print("TRAINING_LP_FILE:", TRAINING_LP_FILE)
    print(df.shape)
    print("**"*20)
    df.to_csv(TRAINING_LP_FILE, index=False)


def make_dlc_pandas_index(model_name, keypoint_names):
    # xyl_labels = ["x", "y", "likelihood"]
    xyl_labels = ["x", "y"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % model_name], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex

def convert_header_to_dlc(df):
    df_arry = df.to_numpy()
    keypoint_names = [b[0] for b in df.columns if b[1] == "x"]
    pdindex = make_dlc_pandas_index('base', keypoint_names)
    df_dlc_index = pd.DataFrame(df_arry, columns=pdindex, index =  df.index)

    print(df.shape, df_arry.shape, df_dlc_index.shape)
    return df_dlc_index

# save project organization to yaml file
save_pred_dir_tp = "/root/capsule/scratch/"
save_dir = save_pred_dir_tp + '/behavior_data/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'
LP_config_template = "/root/capsule/scratch/lightning-pose/scripts/configs/config_toy-Handataset.yaml"
# /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data_modified
################################################################################
# Marton two-view dataset, generate training file, combine multiple cvs files
################################################################################



# out_file = save_dir + scorer_name + '.yaml'
# print("config file based on DLC project:", out_file)
# get_project_org(data_dir, out_file, scorer_name, save_data_dir)

# /root/capsule/data/Marton_behavior_data/BCI_side&bottom_2022_no_pole
scorer_name = "Marton_behavior_data"

project = "BCI_side&bottom_2022_no_pole"
project = "BCI_side&bottom_2023_Sept"
# for project in ["BCI_side&bottom_bottom_only_2023_Sept", "BCI_side&bottom_side_only_2023_Sept"]:
for project in []:
    print("\n", project)
    data_dir = f"/root/capsule/data/{scorer_name}/{project}/"

    Path(save_data_dir).mkdir(parents=True, exist_ok=True)
    save_data_dir_tp = save_data_dir + scorer_name + "/"
    Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)
    save_data_dir_tp = save_data_dir_tp + project
    Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)

    print("Save collected_data_csv to:", save_data_dir_tp)

    generate_collected_data_csv(data_dir, save_data_dir_tp)


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


# /root/capsule/scratch/DLC_dataset_for_LP/Marton_behavior_data_modified
scorer_name = "Marton_behavior_data_modified"
# proj_name = "BCI_side&bottom_2023_Sept"
# proj_name = "BCI_side&bottom_bottom_only_2023_Sept"
proj_name = "BCI_side&bottom_side_only_2023_Sept"
LP_proj_folder = f"/root/capsule/scratch/DLC_dataset_for_LP/{scorer_name}/{proj_name}/"
label_file_name = "CollectedData_all.csv"
file_list = [f"{LP_proj_folder}/{label_file_name}"]
# file_list = []

DLC_folder = f"/root/capsule/data/Marton_behavior_data/{proj_name}/"

for file in file_list:
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

    ###################################
    # copy labeled framess
    ###################################
    for cur_video in video_names[:]:
        print("\n", cur_video)
        # labels_folder = f"{DLC_folder}/labeled-data/{cur_video[6:]}" # for bottom and side views
        labels_folder = f"{DLC_folder}/labeled-data/{cur_video}" # for two-view dataset
        print(labels_folder)
        tgt_labels_folder = f"{LP_proj_folder}/labeled-data/"
        Path(tgt_labels_folder).mkdir(parents=True, exist_ok=True)

        tgt_labels_folder = f"{LP_proj_folder}/labeled-data/{cur_video}/"
        if not os.path.exists(tgt_labels_folder):
            shutil.copytree(labels_folder, tgt_labels_folder)

        print(tgt_labels_folder)


        ###################################
        # generate ground truth csv file for labeled frames
        ###################################
        # save new gt file to
        labels_gt_csv_file = f"{tgt_labels_folder}/CollectedData_Kayvon_Marton_Mohit_Lucas.csv"
        print("labels_gt_csv_file:", labels_gt_csv_file)

        # extract labeled frames based on "CollectedData_all_side_view.csv" "CollectedData_all_bottom_view.csv"
        # since not all labled frames in DLC project used for training.
        print(file)
        df = pd.read_csv(file, sep = ',')

        # extract the labels for current video
        selected_rows = df.loc[df['Unnamed: 0'].str.contains(cur_video, na=False)]
        out_df = pd.concat([df.iloc[[0]], selected_rows], ignore_index=True)
        out_df.to_csv(labels_gt_csv_file, index=False) 

        df = pd.read_csv(labels_gt_csv_file, header=[0, 1], index_col=0)
        df_dlc_index = convert_header_to_dlc(df)
        # df_dlc_index.to_csv(labels_gt_csv_file, index=False) 
        df_dlc_index.to_csv(labels_gt_csv_file) 


    ###################################
    # copy videos
    ###################################
    for cur_video in video_names[:]:
        print(cur_video)
        src_video = f"{DLC_folder}/videos/{cur_video}.mp4"
        print(src_video)
        tgt_labels_folder = f"{LP_proj_folder}/videos/"
        Path(tgt_labels_folder).mkdir(parents=True, exist_ok=True)

        tgt_video = f"{LP_proj_folder}/videos/{cur_video}.mp4"
        if not os.path.exists(tgt_video):
            shutil.copyfile(src_video, tgt_video)

        print(tgt_labels_folder)


#     mice_IDs = list(set([i[:5] for i in video_names]))
#     print(mice_IDs, len(mice_IDs))
#     mice_video = {'BCI26':[],
#     'BCI29':[]}
#     for i in mice_IDs:
#         for j in video_names:
#             if j[:5] == i:
#                 mice_video[i].append(j)
#         print(i, len(mice_video[i]))


#     print(selected_frames)

#     mice_video = {'BCI26':[],
#     'BCI29':[]}
#     for i in mice_IDs:
#         for j in selected_frames:
#             if i in j:
#                 mice_video[i].append(j)
#         print(i, len(mice_video[i]))

# # /root/capsule/data/Marton_behavior_data/BCI_side&bottom_2022_no_pole/videos
# # /root/capsule/data/Marton_behavior_data/BCI_bottom_2022_10_06/labeled-data

# # for folder in labeled_folders:
# # 		tgt_path=os.path.join(res_labeled_dir, folder)
# # 		print("scr_path:", os.path.join(DLC_labeled_data_dir, folder))
# # 		print("tgt_path:", tgt_path)
# # 		print("")
# # 		if not os.path.exists(tgt_path):
# # 			shutil.copytree( os.path.join(DLC_labeled_data_dir, folder), tgt_path)
# # 	print("File Copied Successfully")



