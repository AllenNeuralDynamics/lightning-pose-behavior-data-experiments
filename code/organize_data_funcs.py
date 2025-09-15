import yaml
from pathlib import Path
from glob import glob
import os, sys
import collections
from pprint import pprint
import json
import pandas as pd
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import moviepy.editor as moviepy
import shutil
import random

def get_project_org(data_dir, out_file, scorer_name, save_data_dir):
    ''' 
        generate behavior data organization: projects, video sets, # videos, # key points,  bodyparts, 
        data_dir: the path to DLC project
        out_file: output yaml file
        scorer_name: scorer_name
        save_data_dir: path to save avi to mp4
    '''
    print("Start getting project organization .....")
    csv_files = glob(data_dir + '/*/')
    project_names = []
    for labeled_file in csv_files:
        # print(labeled_file)
        path=os.path.dirname(labeled_file)
        project_names.append(os.path.basename(path))
    print(project_names, len(project_names))

    # creat data organization for each project
    # data_dict = collections.defaultdict(dict)
    data_dict = {}


    for project in project_names:
        print("---------- {}---------------".format(project))
        DLC_dir = data_dir + project 
        print(DLC_dir)

        TRAINING_DLC_FILE = glob(DLC_dir + "/training-datasets/iteration-0/*/CollectedData_*.csv")
        # print(TRAINING_DLC_FILE)

        DLC_FILEs = DLC_dir + "/labeled-data/*/CollectedData_*.csv"
        csv_files = glob(DLC_FILEs)

        # get videos which contain labeled frames
        video_names = []
        for labeled_file in csv_files:
            # print(labeled_file)
            path=os.path.dirname(labeled_file)
            video_names.append(os.path.basename(path))
        # print(video_names, len(video_names))

        # get DLC project config file 
        config_file = os.path.join(DLC_dir, 'config.yaml')
        print(config_file)

        if os.path.exists(config_file):
            # create dictionary for current project 
            # project_dict = collections.defaultdict(dict)
            project_dict = {}
            project_dict['config_file']   = config_file
            project_dict['num_videos']    = len(video_names)

            with open(config_file, 'r') as config_file1:
                DLC_config_param = yaml.safe_load(config_file1)

                # creat dictionary for video sets, specify the original video width and height
                # video_sets = collections.defaultdict(dict)
                video_sets = { j: {                
                                    "image_orig_width": None, 
                                    "image_orig_height": None,
                                    "image_crop_width": None,
                                    "image_crop_height": None,
                                    "1st_labeled_img_path": None,
                                    "video_num_labeled_frames": None,
                                    "video_path": None,
                                    "video_path_mp4": None,
                                    "video_duration": None,
                                    "video_fps": None,
                                    "video_num_frames": None
                                } for j in video_names}

                # print(video_sets)

                total_labeled_frames = 0
                for cur_video_name in video_names:
                    print("cur_video_name:", cur_video_name)

                    # get the crop image dimension
                    for cur_video in list(DLC_config_param["video_sets"]):
                        # print("cur_video", cur_video)
                        if cur_video_name in cur_video:
                            tp = DLC_config_param["video_sets"][cur_video]['crop']
                            tp = tp.split(',')
                            width  = int(tp[1]) - int(tp[0])
                            height = int(tp[3]) - int(tp[2])
                            # print(width, height)
                            video_sets[cur_video_name]['image_crop_width']  = width
                            video_sets[cur_video_name]['image_crop_height'] = height
                            break

                    # find the path of first labeled frame in current video
                    # get the original image dimension
                    labeled_img_dir = os.path.join(DLC_dir, 'labeled-data/', cur_video_name)
                    image_name = glob(labeled_img_dir + '/*.png')[0]
                    image = Image.open(image_name).convert("RGB")
                    video_sets[cur_video_name]['image_orig_width']  = image.size[0]
                    video_sets[cur_video_name]['image_orig_height'] = image.size[1]
                    print(f"The image_orig_dims for {image_name} is (width={image.size[0]}, height={image.size[1]}) ")
                    video_sets[cur_video_name]['1st_labeled_img_path'] = image_name

                    # find the number of labeled frames in current video based on CollectedData_*.csv
                    # DLC_FILE = DLC_dir + f"/labeled-data/{cur_video_name}/CollectedData_*.csv"
                    DLC_FILE = DLC_dir + f"/labeled-data/{cur_video_name}/CollectedData_*.csv"

                    csv_file = glob(DLC_FILE)
                    # print("curr_DLC_FILE:", DLC_FILE)
                    # print('curr_csv_file:', csv_file)
                    if len(csv_file) > 0:
                        df1 = pd.read_csv(csv_file[0], sep = ',')
                        video_sets[cur_video_name]['video_num_labeled_frames'] = df1.shape[0] - 2
                    else:
                        video_sets[cur_video_name]['video_num_labeled_frames'] = 0

                    total_labeled_frames += video_sets[cur_video_name]['video_num_labeled_frames']

                    # find the path of current video and get the video info 
                    curr_video_dir = os.path.join(DLC_dir, 'videos/')
                    curr_video_path = glob(curr_video_dir + '*' + cur_video_name + "*")
                    # print("curr_video_dir:", curr_video_dir)
                    # print('curr_video_path:', curr_video_path)
                    if len(curr_video_path) > 0:
                        video_file = curr_video_path[0]
                        video_sets[cur_video_name]['video_path'] = video_file
                        # analyze the current video  
                        print(video_file)
                        clip = VideoFileClip(video_file)
                        nframes = int(np.ceil(clip.duration * 1.0 * clip.fps))
                        video_sets[cur_video_name]['video_duration'] = clip.duration
                        video_sets[cur_video_name]['video_fps'] = clip.fps
                        video_sets[cur_video_name]['video_num_frames'] = nframes

                        # TODO convert avi to mp4 for LP
                        if video_file.endswith(".avi"):
                            video_type = ".avi"
                            Path(save_data_dir).mkdir(parents=True, exist_ok=True)
                            
                            save_data_dir_tp = os.path.join(save_data_dir, scorer_name)
                            Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)

                            save_data_dir_tp = os.path.join(save_data_dir_tp, project)
                            Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)

                            mp4_video_save_path = os.path.join(save_data_dir_tp, 'video/')
                            Path(mp4_video_save_path).mkdir(parents=True, exist_ok=True)

                            print("Converting video files to be mp4 format!")

                            inputfile  = video_file[:-4] + '.avi'
                            outputfile = video_file[:-4] + '.mp4'

                            outputfile =  os.path.join(mp4_video_save_path, outputfile.split('/')[-1])
                            print("outputfile:", outputfile) 

                            if not os.path.exists(outputfile):
                                clip = moviepy.VideoFileClip(inputfile)
                                clip.write_videofile(outputfile)
                            video_sets[cur_video_name]['video_path_mp4'] = outputfile
                        else:
                            video_type = ".mp4"
                            video_sets[cur_video_name]['video_path_mp4'] = video_file


                    print(video_sets[cur_video_name])
                    print("**"*30)
                    print()
            project_dict['video_sets'] = video_sets
            project_dict['video_dir']  = os.path.join(DLC_dir, 'videos')

            if video_type == '.mp4':
                project_dict['video_dir_mp4'] = os.path.join(DLC_dir, 'videos')
            elif video_type == '.avi':
                project_dict['video_dir_mp4'] = mp4_video_save_path

            project_dict['data_dir']      = DLC_dir
            project_dict['bodyparts']     = DLC_config_param["bodyparts"]
            project_dict['num_bodyparts'] = len(DLC_config_param["bodyparts"])
            project_dict['num_labeled_frames'] = total_labeled_frames

            if TRAINING_DLC_FILE:
                project_dict['TRAINING_DLC_FILE'] = TRAINING_DLC_FILE[0]
            else:
                project_dict['TRAINING_DLC_FILE'] = None
            
            data_dict[project] = project_dict
 
    print("")
    # print("-------------------------------")
    # # final_data_dict = collections.defaultdict(dict)
    # final_data_dict = {}
    # final_data_dict['data_dir'] = data_dir
    # final_data_dict['projects'] = data_dict
    # final_data_dict['num_projects']  = len(project_names)

    # print(json.dumps(final_data_dict, indent=4, sort_keys=True))
    # print("-------------------------------")
    # print("")
    # print(config_file)
    # print(out_file)

    # with open(out_file, 'w') as yaml_file:
    #     # yaml.dump(final_data_dict, yaml_file, default_flow_style=False)
    #     yaml.dump(final_data_dict, yaml_file, default_flow_style=False)
    # print()


def generate_collected_data_csv(data_dir, video_names, save_data_dir):
    ''' generate training/testing file with leave-one-out, 
        convert DLC annotation file to LP training package '''
    
    print("Start generating collected data csv files with leave-one-out ...... ")
    print("find all videos under ", save_data_dir)
    labeled_data_files = data_dir + '/labeled-data/*/CollectedData_*.csv'
    csv_files = glob(labeled_data_files)

    print(len(csv_files))

    # combine all the annotation data
    df1 = pd.read_csv(csv_files[0], sep = ',')
    if len(csv_files) > 1:
        combined_csv = [pd.read_csv(f, sep = ',').iloc[2:] for f in csv_files[1:] ]
        df = pd.concat( [df1] + combined_csv )
    else:
        df = df1
    # print(df)
    if "Unnamed: 1" in df.columns and "Unnamed: 2" in df.columns:
        # drop "Unnamed: 1", "Unnamed: 2" columns for Han's behavior dataset, 
        df["scorer"] = df["scorer"].astype(str) + "/" + df["Unnamed: 1"].astype(str) +"/"+ df[ "Unnamed: 2"].astype(str)
        df = df.drop(["Unnamed: 1", "Unnamed: 2"], axis=1)
        df.at[0,'scorer'] = "bodyparts"
        df.at[1,'scorer'] = "coords"

    print("**"*20)
    TRAINING_LP_FILE = save_data_dir + '/CollectedData_all.csv'
    print("TRAINING_LP_FILE:", TRAINING_LP_FILE)
    print("**"*20)
    df.to_csv(TRAINING_LP_FILE, index=False)

    if len(csv_files) > 1:
        for cur_name in video_names:
            # print(cur_name)
            # find the index of labeled frames from a specific video
            selected_rows = df.loc[df['scorer'].str.contains(cur_name)] 
            # print(selected_rows, len(selected_rows))
            # selected_rows = df.loc[(df['scorer'] == cur_name)] 
            selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows], ignore_index=True)
            selected_rows.to_csv(f"{save_data_dir}/{cur_name}.csv", index=False) # for LP

        for cur_name in video_names:
            # print(cur_name)
            # find the index of labeled frames from a specific video
            selected_rows = df.loc[~df['scorer'].str.contains(cur_name)] 
            # print(selected_rows, len(selected_rows))
            # selected_rows = df.loc[(df['scorer'] == cur_name)] 
            selected_rows = pd.concat([selected_rows], ignore_index=True)
            selected_rows.to_csv(f"{save_data_dir}/{cur_name}_excluded.csv", index=False) # for LP



def closest(K):
    # resize dimensions: we do this internally to accelerate training. make sure its a multiple of 128.
    lst = []
    for i in range(1, 10):
        if K > i*128:
            lst.append(i*128)
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-int(K)))]
     
def edit_cfg_file(data_cfg, LP_config_template, scorer_name, save_data_dir):
    ''' generate cfg file for LP training  with leave-one-out, 

        data_cfg: config file generated from DLC config file, containing the organization of the project.
        LP_config_template: update the template cfg file for lightening-pose training and testing
    '''
    Path(save_data_dir).mkdir(parents=True, exist_ok=True)
    save_data_dir_tp = save_data_dir + scorer_name + "/"
    Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)
    
    for project in list(data_cfg["projects"]):
        print("Current project:", project)
        with open(LP_config_template, 'r') as file:
            param_updated = yaml.safe_load(file)
            # print(param_updated.keys())

            video_sets = data_cfg["projects"][project]['video_sets']
            video_names = data_cfg["projects"][project]['video_sets'].keys()
            print("video_names:", video_names, len(video_names))

            data_dir = data_cfg["projects"][project]['data_dir']

            # save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'
            
            save_data_dir = save_data_dir_tp + project
            Path(save_data_dir).mkdir(parents=True, exist_ok=True)
            print("Save collected_data_csv to:", save_data_dir)

            # generate .csv training file for leave-one-out cross validation
            generate_collected_data_csv(data_dir, video_names, save_data_dir)

            for index, video in enumerate(video_names):
                print(index, video)
                print(data_cfg["projects"][project]['data_dir'])

                # find the original image dimensions 
                param_updated['data']["image_orig_dims"]["width"]  = video_sets[video]["image_orig_width"]
                param_updated['data']["image_orig_dims"]["height"] = video_sets[video]["image_orig_height"]

                if video_sets[video]["image_crop_width"] and video_sets[video]["image_crop_height"]:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[video]["image_crop_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[video]["image_crop_height"])
                else:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[video]["image_orig_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[video]["image_orig_height"])
                
                # TODO: resize image to smaller size to avoid OOM
                if param_updated['data']["image_resize_dims"]["width"] > 640:
                    param_updated['data']["image_resize_dims"]["width"] = 640
                if param_updated['data']["image_resize_dims"]["height"] > 640:
                    param_updated['data']["image_resize_dims"]["height"] = 640

                param_updated['data']["data_dir"]  = data_cfg["projects"][project]['data_dir']
                param_updated['eval']['test_videos_directory'] =  data_cfg["projects"][project]['video_dir_mp4']

                num_bodyparts = data_cfg["projects"][project]['num_bodyparts']
                param_updated['data']["num_keypoints"]  = num_bodyparts
                param_updated['data']["columns_for_singleview_pca"] = [ i for i in range(num_bodyparts) ] 

                param_updated['training']["train_batch_size"] = 8
                param_updated['training']["val_batch_size"] = 8
                param_updated['training']["test_batch_size"] = 8
                param_updated['training']["train_prob"] = 0.9
                param_updated['training']["val_prob"] = 0.05

                # save config file for lightening-pose training
                if index == 0:
                    param_updated['data']["csv_file"]  = os.path.join(save_data_dir, 'CollectedData_all.csv') 
                    out_file = os.path.join(save_data_dir,  'CollectedData_all.config.yaml') 
                    with open(out_file, 'w') as yaml_file:
                        yaml.dump(param_updated, yaml_file)

                if len(video_names) > 1:
                    param_updated['data']["csv_file"]  = os.path.join(save_data_dir, video + '.csv') 
                    param_updated['data']["video_dir"] = os.path.join(data_cfg["projects"][project]['video_dir_mp4'], video + '/')
                    out_file = os.path.join(save_data_dir, video + '.config.yaml') 
                    with open(out_file, 'w') as yaml_file:
                        # yaml.dump(final_data_dict, yaml_file, default_flow_style=False)
                        yaml.dump(param_updated, yaml_file)

                    param_updated['data']["csv_file"]  = os.path.join(save_data_dir, video + '_excluded.csv') 
                    param_updated['data']["video_dir"] = os.path.join(data_cfg["projects"][project]['video_dir_mp4'], video + '_excluded/')
                    out_file = os.path.join(save_data_dir, video + '_excluded.config.yaml') 
                    with open(out_file, 'w') as yaml_file:
                        # yaml.dump(final_data_dict, yaml_file, default_flow_style=False)
                        yaml.dump(param_updated, yaml_file)

        # print("***************")
        # print(param_updated)


def generate_collected_data_csv_leave_mouse_out(data_dir, mice_name, videos_exclude_cur_mice, videos_cur_mice, save_data_dir):
    ''' generate training/testing file with leave-one-out, 
        convert DLC annotation file to LP training package '''

    print("Start generating collected data csv files with leave-one-out ...... ")
    print("find all videos under ", save_data_dir)
    labeled_data_files = data_dir + '/labeled-data/*/CollectedData_*.csv'
    csv_files = glob(labeled_data_files)

    print(len(csv_files))

    # combine all the annotation data
    df1 = pd.read_csv(csv_files[0], sep = ',')
    if len(csv_files) > 1:
        combined_csv = [pd.read_csv(f, sep = ',').iloc[2:] for f in csv_files[1:] ]
        df = pd.concat( [df1] + combined_csv )
    else:
        df = df1
    # print(df)
    if "Unnamed: 1" in df.columns and "Unnamed: 2" in df.columns:
        # drop "Unnamed: 1", "Unnamed: 2" columns for Han's behavior dataset, 
        df["scorer"] = df["scorer"].astype(str) + "/" + df["Unnamed: 1"].astype(str) +"/"+ df[ "Unnamed: 2"].astype(str)
        df = df.drop(["Unnamed: 1", "Unnamed: 2"], axis=1)
        df.at[0,'scorer'] = "bodyparts"
        df.at[1,'scorer'] = "coords"

    print("**"*20)
    # TRAINING_LP_FILE = save_data_dir + '/CollectedData_all.csv'
    # print("TRAINING_LP_FILE:", TRAINING_LP_FILE)
    # print("**"*20)
    # df.to_csv(TRAINING_LP_FILE, index=False)

    if len(csv_files) > 1:
        cur_name = "|".join(videos_cur_mice)
        print(cur_name)
        # find the index of labeled frames from a specific video
        selected_rows = df.loc[df['scorer'].str.contains(cur_name)] 
        selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows], ignore_index=True)
        selected_rows.to_csv(f"{save_data_dir}/{mice_name}.csv", index=False) # for LP

        cur_name = "|".join(videos_exclude_cur_mice)
        print(cur_name)
        # find the index of labeled frames from a specific video
        selected_rows = df.loc[df['scorer'].str.contains(cur_name)] 
        selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows], ignore_index=True)
        selected_rows.to_csv(f"{save_data_dir}/{mice_name}_excluded.csv", index=False) # for LP

def generate_collected_data_csv_leave_mouse_out_includeoneframe(data_dir, mice_name, videos_exclude_cur_mice, videos_cur_mice, save_data_dir):
    ''' generate training/testing file with leave-one-out, 
        supervised parts: keep one frame of each the left-out mouse video and all frames of other mice:
        left-mouse Mouse HH08:
        bottom_face_41-0000: 32 labeled frames ---> only keep one frams
        bottom_face_593-0000: 17 labeled frames --> only keep one frame
        bottom_face_115-0000: 17 labeled frames --> only keep one frame
        ......
        convert DLC annotation file to LP training package '''

    print("Start generating collected data csv files with leave-one-out ...... ")
    print("find all videos under ", save_data_dir)
    labeled_data_files = data_dir + '/labeled-data/*/CollectedData_*.csv'
    csv_files = glob(labeled_data_files)

    print(len(csv_files))

    # combine all the annotation data
    df1 = pd.read_csv(csv_files[0], sep = ',')
    if len(csv_files) > 1:
        combined_csv = [pd.read_csv(f, sep = ',').iloc[2:] for f in csv_files[1:] ]
        df = pd.concat( [df1] + combined_csv )
    else:
        df = df1
    # print(df)
    if "Unnamed: 1" in df.columns and "Unnamed: 2" in df.columns:
        # drop "Unnamed: 1", "Unnamed: 2" columns for Han's behavior dataset, 
        df["scorer"] = df["scorer"].astype(str) + "/" + df["Unnamed: 1"].astype(str) +"/"+ df[ "Unnamed: 2"].astype(str)
        df = df.drop(["Unnamed: 1", "Unnamed: 2"], axis=1)
        df.at[0,'scorer'] = "bodyparts"
        df.at[1,'scorer'] = "coords"

    print("**"*20)

    if len(csv_files) > 1:
        cur_name = "|".join(videos_cur_mice)
        print(cur_name)
        # find the index of labeled frames from a specific video
        _oneframe_pervideo = []
        for cur_name in videos_cur_mice:
            # only keep the first labeled frame
            # selected_rows1 = df.loc[df['scorer'].str.contains(cur_name)].head(1)
           
            # randonly select one labeled frame include in training
            selected_rows_tp = df.loc[df['scorer'].str.contains(cur_name)]
            print(selected_rows_tp.shape)
            pick = random.randint(0,selected_rows_tp.shape[0]-1)
            print("pick one number randomly:", pick)
            selected_rows1 = selected_rows_tp.iloc[[pick]]
            # print(selected_rows1)
            _oneframe_pervideo.append(selected_rows1)
        selected_rows1 = pd.concat(_oneframe_pervideo, ignore_index=True)

        # print("########"*10)
        # print(selected_rows1)
        # print("########"*10)

        cur_name = "|".join(videos_exclude_cur_mice)
        print(cur_name)
        # find the index of labeled frames from a specific video
        selected_rows = df.loc[df['scorer'].str.contains(cur_name)] 
        selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows, selected_rows1], ignore_index=True)
        selected_rows.to_csv(f"{save_data_dir}/{mice_name}_oneframepervideo_keepothermice.csv", index=False) # for LP

        
def generate_collected_data_csv_leave_mouse_out_includeNframe(data_dir, mice_name, videos_exclude_cur_mice, videos_cur_mice, save_data_dir, num_picks):
    ''' generate training/testing file with leave-one-out, 
        supervised parts: keep N frame of the left-out mouse video and all frames of other mice:
        ......
        convert DLC annotation file to LP training package '''

    print("Start generating collected data csv files with leave-one-out ...... ")
    print("find all videos under ", save_data_dir)
    labeled_data_files = data_dir + '/labeled-data/*/CollectedData_*.csv'
    csv_files = glob(labeled_data_files)

    print(len(csv_files))

    # combine all the annotation data
    df1 = pd.read_csv(csv_files[0], sep = ',')
    if len(csv_files) > 1:
        combined_csv = [pd.read_csv(f, sep = ',').iloc[2:] for f in csv_files[1:] ]
        df = pd.concat( [df1] + combined_csv )
    else:
        df = df1
    # print(df)
    if "Unnamed: 1" in df.columns and "Unnamed: 2" in df.columns:
        # drop "Unnamed: 1", "Unnamed: 2" columns for Han's behavior dataset, 
        df["scorer"] = df["scorer"].astype(str) + "/" + df["Unnamed: 1"].astype(str) +"/"+ df[ "Unnamed: 2"].astype(str)
        df = df.drop(["Unnamed: 1", "Unnamed: 2"], axis=1)
        df.at[0,'scorer'] = "bodyparts"
        df.at[1,'scorer'] = "coords"

    print("**"*20)

    if len(csv_files) > 1:
        for num_pick in num_picks:
            print("^^^^^"*20)
            print(f"num_pick = {num_pick}")
            print("^^^^^"*20)

            '''
            include frames of other mice
            '''
            print("Include frames of other mice ... ...")
            cur_name = "|".join(videos_exclude_cur_mice)
            print(cur_name)
            selected_rows_othermice = df.loc[df['scorer'].str.contains(cur_name)] 

            '''
            combine frames of the left-out mouse and randomly pick N frames
            '''
            print("\nInclude frames of the left-out mice ... ...")
            cur_name = "|".join(videos_cur_mice)
            print(cur_name)
            selected_rows_tp = df.loc[df['scorer'].str.contains(cur_name)] 
            pick = random.sample(range(0,selected_rows_tp.shape[0]-1), num_pick)
            print("########"*10)
            print(f"Pick {num_pick} number randomly:", pick,len(pick))
            selected_rows_leftoutmouse = selected_rows_tp.iloc[pick]
            print(selected_rows_leftoutmouse)
            print("########"*10)
            
            '''
            combine frames other mice and the left-out mouse
            '''
            selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows_othermice, selected_rows_leftoutmouse], ignore_index=True)
            print(selected_rows.shape)
            print(f"{save_data_dir}/{mice_name}_{num_pick}frame_keepothermice.csv")
            selected_rows.to_csv(f"{save_data_dir}/{mice_name}_{num_pick}frame_keepothermice.csv", index=False) # for LP


def edit_cfg_file_leave_mouse_out(data_cfg, data_mice_cfg, LP_config_template, scorer_name, save_data_dir):
    ''' generate cfg file for LP training  with leave-one-out, 

        data_cfg: config file generated from DLC config file, containing the organization of the project.
        LP_config_template: update the template cfg file for lightening-pose training and testing
    '''
    Path(save_data_dir).mkdir(parents=True, exist_ok=True)
    save_data_dir_tp = save_data_dir + scorer_name + "/"
    Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)
    
    for project in list(data_cfg["projects"])[:1]:
        print("Current project:", project)
        with open(LP_config_template, 'r') as file:
            param_updated = yaml.safe_load(file)
            # print(param_updated.keys())

            video_sets = data_cfg["projects"][project]['video_sets']
            video_names = data_cfg["projects"][project]['video_sets'].keys()
            print("video_names:", video_names, len(video_names))

            data_dir = data_cfg["projects"][project]['data_dir']

            # save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'
            
            save_data_dir = save_data_dir_tp + project
            Path(save_data_dir).mkdir(parents=True, exist_ok=True)
            print("Save collected_data_csv to:", save_data_dir)

            mice_names = list( data_mice_cfg[project].keys() )
            print("mice_names:", mice_names)

            for index, mice in enumerate(mice_names[:]):
                print(f"\n{index}, {mice}")
                videos_cur_mice = data_mice_cfg[project][mice]['videos']
                print(f"videos_cur_mice: {videos_cur_mice},{len(videos_cur_mice)}\n")

                # exlude the videos of this mice from video_names
                videos_exclude_cur_mice = [ i for i in video_names if i not in videos_cur_mice]
                print(f"videos_exclude_cur_mice: {videos_exclude_cur_mice},{len(videos_exclude_cur_mice)}")

                # # generate .csv leave-one-mouse-out training and testing dataset
                # generate_collected_data_csv_leave_mouse_out(data_dir, mice, videos_exclude_cur_mice, videos_cur_mice, save_data_dir)

                print(data_cfg["projects"][project]['data_dir'])

                cur_video = videos_exclude_cur_mice[0]

                # find the original image dimensions 
                param_updated['data']["image_orig_dims"]["width"]  = video_sets[cur_video]["image_orig_width"]
                param_updated['data']["image_orig_dims"]["height"] = video_sets[cur_video]["image_orig_height"]

                if video_sets[cur_video]["image_crop_width"] and video_sets[cur_video]["image_crop_height"]:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[cur_video]["image_crop_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[cur_video]["image_crop_height"])
                else:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[cur_video]["image_orig_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[cur_video]["image_orig_height"])
                
                # TODO: resize image to smaller size to avoid OOM
                if param_updated['data']["image_resize_dims"]["width"] > 640:
                    param_updated['data']["image_resize_dims"]["width"] = 640
                if param_updated['data']["image_resize_dims"]["height"] > 640:
                    param_updated['data']["image_resize_dims"]["height"] = 640

                param_updated['data']["data_dir"]  = data_cfg["projects"][project]['data_dir']
                param_updated['eval']['test_videos_directory'] =  data_cfg["projects"][project]['video_dir_mp4']

                num_bodyparts = data_cfg["projects"][project]['num_bodyparts']
                param_updated['data']["num_keypoints"]  = num_bodyparts
                param_updated['data']["columns_for_singleview_pca"] = [ i for i in range(num_bodyparts) ] 

                param_updated['training']["train_batch_size"] = 8
                param_updated['training']["val_batch_size"] = 8
                param_updated['training']["test_batch_size"] = 8
                param_updated['training']["train_prob"] = 0.9
                param_updated['training']["val_prob"] = 0.05

                # save config file for lightening-pose training
                # if index == 0:
                #     param_updated['data']["csv_file"]  = os.path.join(save_data_dir, 'CollectedData_all.csv') 
                #     out_file = os.path.join(save_data_dir,  'CollectedData_all.config.yaml') 
                #     with open(out_file, 'w') as yaml_file:
                #         yaml.dump(param_updated, yaml_file)
                
                if len(mice_names) > 1:
                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '.csv') 
                    # param_updated['data']["video_dir"] = data_cfg["projects"][project]['video_dir_mp4'] + mice + '/'
                    # out_file = os.path.join(save_data_dir, mice + '.config.yaml') 
                    # with open(out_file, 'w') as yaml_file:
                    #     # yaml.dump(final_data_dict, yaml_file, default_flow_style=False)
                    #     yaml.dump(param_updated, yaml_file)


                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '_excluded.csv') 
                    # param_updated['data']["video_dir"] = data_cfg["projects"][project]['video_dir_mp4'] + mice + '_excluded/'
                    # out_file = os.path.join(save_data_dir, mice + '_excluded.config.yaml') 
                    # with open(out_file, 'w') as yaml_file:
                    #     yaml.dump(param_updated, yaml_file)


                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '_excluded.csv') 
                    # param_updated['data']["video_dir"] = data_cfg["projects"][project]['video_dir_mp4']
                    # out_file = os.path.join(save_data_dir, mice + '_exclude_frames_include_video.config.yaml') 
                    # with open(out_file, 'w') as yaml_file:
                    #     yaml.dump(param_updated, yaml_file)


                    # unsupervised parts only contains left-out mouse videos to train LP models 
                    param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '_excluded.csv') 
                    all_video_dir = data_cfg["projects"][project]['video_dir_mp4']

                    mice_video_path = data_cfg["projects"][project]['video_dir_mp4'] + mice + '/'
                    Path(mice_video_path).mkdir(parents=True, exist_ok=True)
                    for _video in videos_cur_mice:
                        print(_video)
                        _video_path = glob(all_video_dir + '/' + _video + "*.*")[0]
                        print(_video_path)
                        shutil.copy(_video_path, mice_video_path)
                    
                    param_updated['data']["video_dir"] = mice_video_path
                    print("^^^^^^^^^^^"*8)
                    print(param_updated['data']["video_dir"])
                    print(f"mice_video_path: {mice_video_path}")
                    print("^^^^^^^^^^^"*8)

                    out_file = os.path.join(save_data_dir, mice + '_exclude_frames_include_leftoutmouse_video.config.yaml') 
                    with open(out_file, 'w') as yaml_file:
                        yaml.dump(param_updated, yaml_file)

     
        print("***************")
        print(param_updated)


def edit_cfg_file_leave_mouse_out_inlcude_oneframe(data_cfg, data_mice_cfg, LP_config_template, scorer_name, save_data_dir):
    ''' generate cfg file for LP training  with leave-one-out, 
        
        # supervised parts contains N frame of left-out mouse video and frames of other mice to train LP models 
        # unsupervised parts only contains left-out mouse videos to train LP models 
        ......
        data_cfg: config file generated from DLC config file, containing the organization of the project.
        LP_config_template: update the template cfg file for lightening-pose training and testing
    '''
    Path(save_data_dir).mkdir(parents=True, exist_ok=True)
    save_data_dir_tp = save_data_dir + scorer_name + "/"
    Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)
    
    for project in list(data_cfg["projects"])[:1]:
        print("Current project:", project)
        with open(LP_config_template, 'r') as file:
            param_updated = yaml.safe_load(file)
            # print(param_updated.keys())

            video_sets = data_cfg["projects"][project]['video_sets']
            video_names = data_cfg["projects"][project]['video_sets'].keys()
            print("video_names:", video_names, len(video_names))

            data_dir = data_cfg["projects"][project]['data_dir']

            # save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'
            
            save_data_dir = save_data_dir_tp + project
            Path(save_data_dir).mkdir(parents=True, exist_ok=True)
            print("Save collected_data_csv to:", save_data_dir)

            mice_names = list( data_mice_cfg[project].keys() )
            print("mice_names:", mice_names, len(mice_names))

            for index, mice in enumerate(mice_names[3:4]):
                mice = "HH15"
                print(f"\n{index}, {mice}")
                videos_cur_mice = data_mice_cfg[project][mice]['videos']
                print(f"videos_cur_mice: {videos_cur_mice},{len(videos_cur_mice)}")

                # exlude the videos of this mice from video_names
                videos_exclude_cur_mice = [ i for i in video_names if i not in videos_cur_mice]
                print(f"videos_exclude_cur_mice: {videos_exclude_cur_mice},{len(videos_exclude_cur_mice)}")

                # # generate .csv leave-one-mouse-out training and testing dataset
                # generate_collected_data_csv_leave_mouse_out(data_dir, mice, videos_exclude_cur_mice, videos_cur_mice, save_data_dir)
                # generate_collected_data_csv_leave_mouse_out_includeoneframe(data_dir, mice, videos_exclude_cur_mice, videos_cur_mice, save_data_dir)
                # num_picks = [1, 2, 4, 8, 16, 32, 64, 128]
                # num_picks = [1, 2, 4, 8, 16, 32, 64]
                # num_picks = [1, 2, 4, 8, 16, 32]
                num_picks = [1, 2, 4]


                generate_collected_data_csv_leave_mouse_out_includeNframe(data_dir, mice, videos_exclude_cur_mice, videos_cur_mice, save_data_dir, num_picks)
                print(data_cfg["projects"][project]['data_dir'])


                ################################################
                ################################################

                cur_video = videos_exclude_cur_mice[0]
                # find the original image dimensions 
                param_updated['data']["image_orig_dims"]["width"]  = video_sets[cur_video]["image_orig_width"]
                param_updated['data']["image_orig_dims"]["height"] = video_sets[cur_video]["image_orig_height"]

                if video_sets[cur_video]["image_crop_width"] and video_sets[cur_video]["image_crop_height"]:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[cur_video]["image_crop_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[cur_video]["image_crop_height"])
                else:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[cur_video]["image_orig_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[cur_video]["image_orig_height"])
                
                # TODO: resize image to smaller size to avoid OOM
                if param_updated['data']["image_resize_dims"]["width"] > 640:
                    param_updated['data']["image_resize_dims"]["width"] = 640
                if param_updated['data']["image_resize_dims"]["height"] > 640:
                    param_updated['data']["image_resize_dims"]["height"] = 640

                param_updated['data']["data_dir"]  = data_cfg["projects"][project]['data_dir']

                num_bodyparts = data_cfg["projects"][project]['num_bodyparts']
                param_updated['data']["num_keypoints"]  = num_bodyparts
                param_updated['data']["columns_for_singleview_pca"] = [ i for i in range(num_bodyparts) ] 

                param_updated['training']["train_batch_size"] = 8
                param_updated['training']["val_batch_size"] = 8
                param_updated['training']["test_batch_size"] = 8
                param_updated['training']["train_prob"] = 0.9
                param_updated['training']["val_prob"] = 0.05

                # save config file for lightening-pose training
                # if index == 0:
                #     param_updated['data']["csv_file"]  = os.path.join(save_data_dir, 'CollectedData_all.csv') 
                #     out_file = os.path.join(save_data_dir,  'CollectedData_all.config.yaml') 
                #     with open(out_file, 'w') as yaml_file:
                #         yaml.dump(param_updated, yaml_file)
                
                if len(mice_names) > 1:
                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '.csv') 
                    # param_updated['data']["video_dir"] = data_cfg["projects"][project]['video_dir_mp4'] + mice + '/'
                    # out_file = os.path.join(save_data_dir, mice + '.config.yaml') 
                    # with open(out_file, 'w') as yaml_file:
                    #     # yaml.dump(final_data_dict, yaml_file, default_flow_style=False)
                    #     yaml.dump(param_updated, yaml_file)


                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '_excluded.csv') 
                    # param_updated['data']["video_dir"] = data_cfg["projects"][project]['video_dir_mp4'] + mice + '_excluded/'
                    # out_file = os.path.join(save_data_dir, mice + '_excluded.config.yaml') 
                    # with open(out_file, 'w') as yaml_file:
                    #     yaml.dump(param_updated, yaml_file)


                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '_excluded.csv') 
                    # param_updated['data']["video_dir"] = data_cfg["projects"][project]['video_dir_mp4']
                    # out_file = os.path.join(save_data_dir, mice + '_exclude_frames_include_video.config.yaml') 
                    # with open(out_file, 'w') as yaml_file:
                    #     yaml.dump(param_updated, yaml_file)


                    # unsupervised parts only contains left-out mouse videos to train LP models 
                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '_excluded.csv') 

                    # # supervised parts contains one frame of each left-out mouse videos (
                    # left-out Mouse HH08:
                        # bottom_face_41-0000: 32 labeled frames ---> only keep one frams
                        # bottom_face_593-0000: 17 labeled frames --> only keep one frame
                        # bottom_face_115-0000: 17 labeled frames --> only keep one frame
                    # and framse of other mice to train LP models 
                    # # unsupervised parts only contains left-out mouse videos to train LP models 
                    # param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + '_oneframepervideo_keepothermice.csv') 


                    # supervised parts contains N frames randomly picked from the left-out mouse videos 
                    # and framse of other mice to train LP models 
                    # unsupervised parts only contains left-out mouse videos to train LP models 

                    all_video_dir = data_cfg["projects"][project]['video_dir_mp4']
                    mice_video_path = data_cfg["projects"][project]['video_dir_mp4'] + mice + '/'
                    param_updated['data']["video_dir"] = mice_video_path
                    param_updated['eval']['test_videos_directory'] = mice_video_path

                    # Path(mice_video_path).mkdir(parents=True, exist_ok=True)
                    # for _video in videos_cur_mice:
                    #     print(_video)
                    #     _video_path = glob(all_video_dir + '/' + _video + "*.*")[0]
                    #     print(_video_path)
                    #     shutil.copy(_video_path, mice_video_path)
   
                    for num_pick in num_picks:
                        param_updated['data']["csv_file"]  = os.path.join(save_data_dir, mice + f'_{num_pick}frame_keepothermice.csv') 

                        out_file = os.path.join(save_data_dir, mice + f'_exclude_frames_include_leftoutmouse_{num_pick}frameandvideo.config.yaml') 
                        with open(out_file, 'w') as yaml_file:
                            yaml.dump(param_updated, yaml_file)
                        
                        print("^^^^^^^^^^^"*8)
                        print("param_updated[data][video_dir]:", param_updated['data']["video_dir"])
                        print("param_updated[data][csv_file] :", param_updated['data']["csv_file"])
                        print(out_file)
                        print("^^^^^^^^^^^"*8)
     
        print("***************")
        print(param_updated)


def generate_collected_data_csv_includeNframe(data_dir, save_data_dir, perc_picks):
    ''' generate training/testing file with all collected data, 
        supervised parts: keep a subset of training dataset
        ......
        convert DLC annotation file to LP training package '''

    print("Start generating collected data csv files with the subset of training data ...... ")
    print("find all videos under ", save_data_dir)
    labeled_data_files = data_dir + '/labeled-data/*/CollectedData_*.csv'
    csv_files = glob(labeled_data_files)

    print(len(csv_files))

    # combine all the annotation data
    df1 = pd.read_csv(csv_files[0], sep = ',')
    if len(csv_files) > 1:
        combined_csv = [pd.read_csv(f, sep = ',').iloc[2:] for f in csv_files[1:] ]
        df = pd.concat( [df1] + combined_csv )
    else:
        df = df1
    # print(df)
    if "Unnamed: 1" in df.columns and "Unnamed: 2" in df.columns:
        # drop "Unnamed: 1", "Unnamed: 2" columns for Han's behavior dataset, 
        df["scorer"] = df["scorer"].astype(str) + "/" + df["Unnamed: 1"].astype(str) +"/"+ df[ "Unnamed: 2"].astype(str)
        df = df.drop(["Unnamed: 1", "Unnamed: 2"], axis=1)
        df.at[0,'scorer'] = "bodyparts"
        df.at[1,'scorer'] = "coords"

    print("**"*20)
    video_name_list = []
    for perc_pick in perc_picks:
        num_pick = int(perc_pick / 100 * (df.shape[0]-2))
        print("^^^^^"*20)
        print(f"all data has {df.shape} ")
        print(f"perc_pick = {perc_pick}, num_pick = {num_pick}")
        print("^^^^^"*20)
        '''
        select a subset of frames
        '''
        selected_rows = df.iloc[2:].sample(n = num_pick)
        print(df.iloc[2:])

        # return the name of the first video
        print(selected_rows.head(1))
        print(selected_rows.head(1)['scorer'].values[0])
        video_name = selected_rows.head(1)['scorer'].values[0].split('/')[1]
        print(video_name)
        video_name_list.append(video_name)

        '''
        combine frames with headers
        '''
        selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows], ignore_index=True)
        print(selected_rows.shape)
        print(f"{save_data_dir}/CollectedData_all_{perc_pick}perc.csv")
        selected_rows.to_csv(f"{save_data_dir}/CollectedData_all_{perc_pick}perc.csv", index=False) # for LP
    return video_name_list


def edit_cfg_file_subset_traindata(data_cfg, data_mice_cfg, LP_config_template, scorer_name, save_data_dir):
    ''' generate cfg file for LP training with leave-one-out, 
        # supervised parts contains the subset of training data
        ......
        data_cfg: config file generated from DLC config file, containing the organization of the project.
        LP_config_template: update the template cfg file for lightening-pose training and testing
    '''
    Path(save_data_dir).mkdir(parents=True, exist_ok=True)
    save_data_dir_tp = save_data_dir + scorer_name + "/"
    Path(save_data_dir_tp).mkdir(parents=True, exist_ok=True)
    
    for project in list(data_cfg["projects"])[:1]:
        print("Current project:", project)
        with open(LP_config_template, 'r') as file:
            param_updated = yaml.safe_load(file)
            # print(param_updated.keys())

            video_sets  = data_cfg["projects"][project]['video_sets']
            video_names = data_cfg["projects"][project]['video_sets'].keys()
            print("video_names:", video_names, len(video_names))

            data_dir = data_cfg["projects"][project]['data_dir']

            # save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'
            
            save_data_dir = save_data_dir_tp + project
            Path(save_data_dir).mkdir(parents=True, exist_ok=True)
            print("Save collected_data_csv to:", save_data_dir)

            mice_names = list( data_mice_cfg[project].keys() )
            print("mice_names:", mice_names, len(mice_names))

            # perc_picks = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            perc_picks = [1, 2, 5]

            video_name_list = generate_collected_data_csv_includeNframe(data_dir, save_data_dir, perc_picks)
            print(data_cfg["projects"][project]['data_dir'])
            print(video_name_list)

            for index, perc_pick in enumerate(perc_picks):
                # need to find the first selected video
                cur_video = video_name_list[index]
                print(f"\n{index}, {perc_pick}, the first selected video is {cur_video}")
                # find the original image dimensions 
                param_updated['data']["image_orig_dims"]["width"]  = video_sets[cur_video]["image_orig_width"]
                param_updated['data']["image_orig_dims"]["height"] = video_sets[cur_video]["image_orig_height"]

                if video_sets[cur_video]["image_crop_width"] and video_sets[cur_video]["image_crop_height"]:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[cur_video]["image_crop_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[cur_video]["image_crop_height"])
                else:
                    param_updated['data']["image_resize_dims"]["width"] = closest(video_sets[cur_video]["image_orig_width"])
                    param_updated['data']["image_resize_dims"]["height"] = closest(video_sets[cur_video]["image_orig_height"])
                
                # TODO: resize image to smaller size to avoid OOM
                if param_updated['data']["image_resize_dims"]["width"] > 640:
                    param_updated['data']["image_resize_dims"]["width"] = 640
                if param_updated['data']["image_resize_dims"]["height"] > 640:
                    param_updated['data']["image_resize_dims"]["height"] = 640

                param_updated['data']["data_dir"]  = data_cfg["projects"][project]['data_dir']

                num_bodyparts = data_cfg["projects"][project]['num_bodyparts']
                param_updated['data']["num_keypoints"]  = num_bodyparts
                param_updated['data']["columns_for_singleview_pca"] = [ i for i in range(num_bodyparts) ] 

                param_updated['training']["train_batch_size"] = 8
                param_updated['training']["val_batch_size"] = 8
                param_updated['training']["test_batch_size"] = 8
                param_updated['training']["train_prob"] = 0.9
                param_updated['training']["val_prob"] = 0.05

                all_video_dir = data_cfg["projects"][project]['video_dir_mp4']
                param_updated['data']["video_dir"] = all_video_dir
                param_updated['eval']['test_videos_directory'] = all_video_dir
                param_updated['data']["csv_file"]  = os.path.join(save_data_dir, f'CollectedData_all_{perc_pick}perc.csv') 

                out_file = os.path.join(save_data_dir, f'CollectedData_all_{perc_pick}perc.config.yaml') 
                with open(out_file, 'w') as yaml_file:
                    yaml.dump(param_updated, yaml_file)
                
                print("^^^^^^^^^^^"*8)
                print("param_updated[data][video_dir]:", param_updated['data']["video_dir"])
                print("param_updated[data][csv_file] :", param_updated['data']["csv_file"])
                print("param_updated[eval][test_videos_directory] :", param_updated['eval']["test_videos_directory"])
                print(out_file)
                print("^^^^^^^^^^^"*8)
    
        print("***************")
        print(param_updated)
