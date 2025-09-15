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

# train_data_name = "CollectedData_all"
#  ['bottom_face_1-0000_excluded', 'bottom_face_5-0000_excluded', 'bottom_face_6-0000_excluded', 'bottom_face_20-0000_excluded', 'bottom_face_37-0000_excluded', 'bottom_face_41-0000_excluded']
# for train_data_name in ['bottom_face_1-0000_excluded', 'bottom_face_6-0000_excluded', 'bottom_face_37-0000_excluded']:
# for train_data_name in ['bottom_face_41-0000_excluded']:
# for train_data_name in ['CollectedData_all_include_nan_keypoint']:


# train_data_type = 'video'
# for train_data_name in ['bottom_face_1-0000_excluded', 'bottom_face_6-0000_excluded', 'bottom_face_37-0000_excluded', 'bottom_face_49-0000_excluded', 'bottom_face_110-0000_excluded']:
# for train_data_name in ['bottom_face_1-0000', 'bottom_face_6-0000', 'bottom_face_37-0000', 'bottom_face_49-0000', 'bottom_face_110-0000']:


# train_data_type = 'mouse'
# for train_data_name in ['HH08_excluded', 'HH09_excluded', 'HH13_excluded', 'HH14_excluded', 'HH15_excluded', 'HH16_excluded', 'HH18_excluded', 
# 'HH08', 'HH09', 'HH13', 'HH14', 'HH15', 'HH16', 'HH18']:
# for train_data_name in ['HH15_excluded', 'HH16_excluded', 'HH15', 'HH16']:

# # 'HH08_exclude_frames_include_video', 
train_data_type = "mouse_exclude_frames_include_video"
# for train_data_name in ['HH08_exclude_frames_include_video', 'HH09_exclude_frames_include_video', 'HH13_exclude_frames_include_video', 'HH14_exclude_frames_include_video', 
#                         'HH15_exclude_frames_include_video', 'HH16_exclude_frames_include_video', 'HH18_exclude_frames_include_video']:
# for train_data_name in ['HH08_exclude_frames_include_video']:
    

# train_data_type = "mouse_exclude_frames_include_leftoutmouse_video"
# for train_data_name in ['HH08_exclude_frames_include_leftoutmouse_video', 
#                         'HH09_exclude_frames_include_leftoutmouse_video', 
#                         'HH13_exclude_frames_include_leftoutmouse_video', 
#                         'HH14_exclude_frames_include_leftoutmouse_video', 
#                         'HH15_exclude_frames_include_leftoutmouse_video', 
#                         'HH16_exclude_frames_include_leftoutmouse_video', 
#                         'HH18_exclude_frames_include_leftoutmouse_video']:

train_data_type = "mouse_exclude_frames_include_leftoutmouse_video"
# for train_data_name in ['HH08_exclude_frames_include_leftoutmouse_oneframeandvideo', 
#                         'HH09_exclude_frames_include_leftoutmouse_oneframeandvideo', 
#                         'HH13_exclude_frames_include_leftoutmouse_oneframeandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_oneframeandvideo', 
#                         'HH15_exclude_frames_include_leftoutmouse_oneframeandvideo', 
#                         'HH16_exclude_frames_include_leftoutmouse_oneframeandvideo', 
#                         'HH18_exclude_frames_include_leftoutmouse_oneframeandvideo']:

# for train_data_name in ['HH08_exclude_frames_include_leftoutmouse_1frameandvideo', 
#                         'HH08_exclude_frames_include_leftoutmouse_2frameandvideo', 
#                         'HH08_exclude_frames_include_leftoutmouse_4frameandvideo', 
#                         'HH08_exclude_frames_include_leftoutmouse_8frameandvideo', 
#                         'HH08_exclude_frames_include_leftoutmouse_16frameandvideo', 
#                         'HH08_exclude_frames_include_leftoutmouse_32frameandvideo', 
#                         'HH08_exclude_frames_include_leftoutmouse_64frameandvideo',
#                         'HH08_exclude_frames_include_leftoutmouse_128frameandvideo',  
#                         ]:

# for train_data_name in ['HH09_exclude_frames_include_leftoutmouse_1frameandvideo', 
#                         'HH09_exclude_frames_include_leftoutmouse_2frameandvideo', 
#                         'HH09_exclude_frames_include_leftoutmouse_4frameandvideo', 
#                         'HH09_exclude_frames_include_leftoutmouse_8frameandvideo', 
#                         'HH09_exclude_frames_include_leftoutmouse_16frameandvideo', 
#                         'HH09_exclude_frames_include_leftoutmouse_32frameandvideo', 
#                         'HH09_exclude_frames_include_leftoutmouse_64frameandvideo',
#                         'HH09_exclude_frames_include_leftoutmouse_128frameandvideo',  
#                         ]:
# for train_data_name in ['HH13_exclude_frames_include_leftoutmouse_1frameandvideo', 
#                         'HH13_exclude_frames_include_leftoutmouse_2frameandvideo', 
#                         'HH13_exclude_frames_include_leftoutmouse_4frameandvideo', 
#                         'HH13_exclude_frames_include_leftoutmouse_8frameandvideo', 
#                         'HH13_exclude_frames_include_leftoutmouse_16frameandvideo', 
#                         'HH13_exclude_frames_include_leftoutmouse_32frameandvideo', 
#                         'HH13_exclude_frames_include_leftoutmouse_64frameandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_1frameandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_2frameandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_4frameandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_8frameandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_16frameandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_32frameandvideo', 
#                         'HH14_exclude_frames_include_leftoutmouse_64frameandvideo', 
#                         ]:
# 'HH18_exclude_frames_include_leftoutmouse_1frameandvideo', 
#                         'HH18_exclude_frames_include_leftoutmouse_2frameandvideo', 
#                         'HH18_exclude_frames_include_leftoutmouse_4frameandvideo', 
#                         'HH18_exclude_frames_include_leftoutmouse_8frameandvideo', 
#                         'HH18_exclude_frames_include_leftoutmouse_16frameandvideo', 
#                         'HH18_exclude_frames_include_leftoutmouse_32frameandvideo', 

#                         'HH16_exclude_frames_include_leftoutmouse_1frameandvideo', 
#                         'HH16_exclude_frames_include_leftoutmouse_2frameandvideo', 
#                         'HH16_exclude_frames_include_leftoutmouse_4frameandvideo', 
#                         'HH16_exclude_frames_include_leftoutmouse_8frameandvideo',
                        # 'HH16_exclude_frames_include_leftoutmouse_16frameandvideo', 
                        # 'HH15_exclude_frames_include_leftoutmouse_1frameandvideo', 
                        # 'HH15_exclude_frames_include_leftoutmouse_2frameandvideo', 
for train_data_name in [  
                        'HH15_exclude_frames_include_leftoutmouse_4frameandvideo', 
                        ]:

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

    ##########################################################

    # TODO
    # array=("0" "1")
    # datacolumns_for_singleview_pca="${array[@]}"
    # losses.pca_singleview.components_to_keep=0.7
    # eval.test_videos_directory=${test_videos_dir}
    # training.max_epochs=${max_epochs}
    # test_videos_dir="/root/capsule/scratch/LP_outputs/Han_behavior_data/Foraging_Bot-Han_Lucas-2022-04-27/CollectedData_all/videos/"
    # f"eval.test_videos_directory={test_videos_dir}", \
    # f"data.video_dir={test_videos_dir}", \

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


    if train_data_type == "mouse_exclude_frames_include_leftoutmouse_video":
        mice = train_data_name.replace("_exclude_frames_include_leftoutmouse_video", "")
        print("^^^^"*20)
        print(f"Train {mice}_exclude_frames_include_leftoutmouse_video")
        print("^^^^"*20)

    elif train_data_type == "mouse_exclude_frames_include_video":
        mice = train_data_name.replace("_exclude_frames_include_video", "")
        print(f"Train {mice}_exclude_frames_include_video")
    else:
        print(f"leave-video-out or leave-mouse-mout......")
        if train_data_type == "video":
            if "_excluded" in train_data_name:
                _video_name = train_data_name.replace('_excluded', '')
                print("Exclude one video:", _video_name)
                train_video_folder = [ i for i in video_names if i != _video_name]
            else:
                print("Include one video:", train_data_name)
                # only one video from training data		
                train_video_folder = [train_data_name]
        elif train_data_type == "mouse":
            mice = train_data_name.replace("_excluded", "")
            videos_cur_mice = data_mice_cfg[project_folder_name][mice]['videos']
            print(f"videos_cur_mice: {videos_cur_mice},{len(videos_cur_mice)}")
            if "_excluded" in train_data_name: 
                # exlude the videos of this mice from video_names
                videos_exclude_cur_mice = [ i for i in video_names if i not in videos_cur_mice]
                print(f"videos_exclude_cur_mice: {videos_exclude_cur_mice},{len(videos_exclude_cur_mice)}")
                train_video_folder = videos_exclude_cur_mice
            else:
                train_video_folder = videos_cur_mice

        for _video in train_video_folder:
            print(_video)
            _video_path = glob(all_video_dir + '/' + _video + "*.*")[0]
            print(_video_path)
            shutil.copy(_video_path, video_dir)


        # TODO: 



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


