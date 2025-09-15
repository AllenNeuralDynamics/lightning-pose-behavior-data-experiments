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

# save project organization to yaml file
save_pred_dir_tp = "/root/capsule/scratch/"
save_dir = save_pred_dir_tp + '/behavior_data/'
Path(save_dir).mkdir(parents=True, exist_ok=True)

save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'

LP_config_template = "/root/capsule/scratch/lightning-pose/scripts/configs/config_toy-Handataset.yaml"

############################################

# scorer_name = "Corbett_behavior_data"
# data_dir = "/root/capsule/data/" + scorer_name + "/"
# out_file = save_dir + scorer_name + '.yaml'
# print("config file based on DLC project:", out_file)
# # get_project_org(data_dir, out_file, scorer_name, save_data_dir)

# # load Lightening pose config file and modify it.
# with open(out_file, 'r') as file:
#     data_cfg = yaml.safe_load(file)
#     print(data_cfg.keys())
# edit_cfg_file(data_cfg, LP_config_template, scorer_name, save_data_dir)

############################################

# scorer_name = "Marton_behavior_data"
# data_dir = "/root/capsule/data/" + scorer_name + "/"
# out_file = save_dir + scorer_name + '.yaml'
# print("config file based on DLC project:", out_file)
# get_project_org(data_dir, out_file, scorer_name, save_data_dir)

# # load Lightening pose config file and modify it.
# with open(out_file, 'r') as file:
#     data_cfg = yaml.safe_load(file)
#     print(data_cfg.keys())
# edit_cfg_file(data_cfg, LP_config_template, scorer_name, save_data_dir)

# ############################################

scorer_name = "Han_behavior_data"
data_dir = "/root/capsule/data/s3_video/DLC_projects/"
out_file = save_dir + scorer_name + '.yaml'
print("config file based on DLC project:", out_file)
# get_project_org(data_dir, out_file, scorer_name, save_data_dir)

# load Lightening pose config file and modify it.
with open(out_file, 'r') as file:
    data_cfg = yaml.safe_load(file)
    print(data_cfg.keys())

# # ---------- leave-one-video-out----------# 
# edit_cfg_file(data_cfg, LP_config_template, scorer_name, save_data_dir)

# # ----------leave-one-mouse-out----------#
# out_file = save_dir + scorer_name + '_mice_inf.yaml'
# with open(out_file, 'r') as file:
#     data_mice_cfg = yaml.safe_load(file)
#     print(data_mice_cfg.keys())
# edit_cfg_file_leave_mouse_out(data_cfg, data_mice_cfg, LP_config_template, scorer_name, save_data_dir)

# # ----------leave-one-mouse-out exclude frames but include videos for unsupervised training ----------#
# out_file = save_dir + scorer_name + '_mice_inf.yaml'
# with open(out_file, 'r') as file:
#     data_mice_cfg = yaml.safe_load(file)
#     print(data_mice_cfg.keys())
# edit_cfg_file_leave_mouse_out(data_cfg, data_mice_cfg, LP_config_template, scorer_name, save_data_dir)


# # ----------leave-one-mouse-out exclude frames but only include one labeled frame and include left-out mouse videos for unsupervised training ----------#
# out_file = save_dir + scorer_name + '_mice_inf.yaml'
# with open(out_file, 'r') as file:
#     data_mice_cfg = yaml.safe_load(file)
#     print(data_mice_cfg.keys())
# edit_cfg_file_leave_mouse_out_inlcude_oneframe(data_cfg, data_mice_cfg, LP_config_template, scorer_name, save_data_dir)

# ---------- train model on the subset of training data ----------#
out_file = save_dir + scorer_name + '_mice_inf.yaml'
with open(out_file, 'r') as file:
    data_mice_cfg = yaml.safe_load(file)
    print(data_mice_cfg.keys())
edit_cfg_file_subset_traindata(data_cfg, data_mice_cfg, LP_config_template, scorer_name, save_data_dir)



############### TODO summarize behavior data organization #############################
for p in range(0):
    res_dir = "/root/capsule/scratch/behavior_data/"
    log_dir =  os.path.join(res_dir,'logs')
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    testing_log = os.path.join(log_dir, f"{scorer_name}_info.log")
    testing_error_log = os.path.join(log_dir, f"{scorer_name}_info_error.log")

    if os.path.exists(testing_log):
        os.remove(testing_log)
    if os.path.exists(testing_error_log):
        os.remove(testing_error_log)

    sys.stdout = Logger(testing_log, sys.stdout)
    sys.stderr = Logger(testing_error_log, sys.stderr)     # redirect std err, if necessary

    with open(out_file, 'r') as file:
        data_cfg = yaml.safe_load(file)
        print(data_cfg.keys())

        print(
            "There are {} DLC projects \n".format(
                data_cfg["num_projects"]
            )
        )
    # from termcolor import colored
    for project in list(data_cfg["projects"]):
        print(f"{bcolors.BOLD}Project: {project}{bcolors.ENDC}")
        # print(colored(f"Project: {project}", 'red'))
        # print(f"Project: {project}")
        # print(f"\033[38;5;4mProject: {project}")


        # print(data_cfg["projects"][project]['data_dir'])
        # print(f"{bcolors.UNDERLINE}   {data_cfg["projects"][project]['num_videos']} videos, {data_cfg["projects"][project]['num_labeled_frames']} labeled frames. {bcolors.ENDC}")

        a = (data_cfg["projects"][project]['num_videos'])
        b = (data_cfg["projects"][project]['num_labeled_frames'])
        print(f"{bcolors.UNDERLINE} {a} videos, {b} labeled frames.{bcolors.ENDC}")
        print(
            "    {} bodyparts, {}\n \
            ".format(
                data_cfg["projects"][project]['num_bodyparts'], data_cfg["projects"][project]['bodyparts']
            )
        )
        video_sets = data_cfg["projects"][project]['video_sets']
        video_names = [i for i in  data_cfg["projects"][project]['video_sets'].keys()]
        # print("video_names:", video_names, len(video_names))

        for index, video in enumerate(video_names[:1]):
            print("    The summary of the video {}:".format(video))
            # print(data_cfg["projects"][project]['data_dir'])
            print(
                "        Duration of video [s]: {}, recorded with {} fps. \n        Overall # of frames: {} with the frame dimension {} X {} (WIDTH X HEIGHT). \n        {} labeled frames generated by DLC GUI. \n \
                ".format(
                    video_sets[video]["video_duration"], video_sets[video]["video_fps"],
                    video_sets[video]["video_num_frames"], video_sets[video]["image_orig_width"], video_sets[video]["image_orig_height"],
                    video_sets[video]["video_num_labeled_frames"]
                )
            )



# han_meta_file = "/root/capsule/scratch/behavior_data/UPDATED_BCI_annotated_videos_notes_LucasKinsey_20220420 (version 2) (version 3).xlsx"
# meta = pd.read_excel(han_meta_file)
# print(meta.head())