# convert DLC format to lightning-pose
import os, sys
from glob import glob
from pathlib import Path
import pandas as pd

def convert_DLC_to_LP(TRAINING_DLC_FILE, save_data_dir, video_names):
    ''' convert DLC annotation file to lightning-pose training package '''
    print("Converting DeepLabCut datasets into LP file.")
    df = pd.read_csv(TRAINING_DLC_FILE, sep = ',')
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

    for cur_name in video_names:
        print(cur_name)
        # find the index of labeled frames from a specific video
        selected_rows = df.loc[df['scorer'].str.contains(cur_name)] 
        print(selected_rows, len(selected_rows))
        # selected_rows = df.loc[(df['scorer'] == cur_name)] 
        selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows], ignore_index=True)
        selected_rows.to_csv(f"{save_data_dir}/{cur_name}.csv", index=False) # for LP

    for cur_name in video_names:
        print(cur_name)
        # find the index of labeled frames from a specific video
        selected_rows = df.loc[~df['scorer'].str.contains(cur_name)] 
        print(selected_rows, len(selected_rows))
        # selected_rows = df.loc[(df['scorer'] == cur_name)] 
        selected_rows = pd.concat([df.iloc[[0, 1]], selected_rows], ignore_index=True)
        selected_rows.to_csv(f"{save_data_dir}/{cur_name}_excluded.csv", index=False) # for LP

project_folder_name="Foraging_Bot-Han_Lucas-2022-04-27"
DLC_dir = "/root/capsule/data/s3_video/DLC_projects/" + project_folder_name

save_data_dir = '/root/capsule/scratch/DLC_dataset_for_LP/'
Path(save_data_dir).mkdir(parents=True, exist_ok=True)

save_data_dir = save_data_dir + project_folder_name
Path(save_data_dir).mkdir(parents=True, exist_ok=True)

'''
create lightening pose annotation data file that contains all the videos
'''
# TRAINING_DLC_FILE = DLC_dir + "/labeled-data/bottom_face_*/CollectedData_*.csv"
TRAINING_DLC_FILE = DLC_dir + "/training-datasets/iteration-0/UnaugmentedDataSet_Foraging_BotApr27/CollectedData_Han_Lucas.csv"

DLC_FILEs = DLC_dir + "/labeled-data/bottom_face_*/CollectedData_*.csv"
csv_files = glob(DLC_FILEs)
video_names = []
for labeled_file in csv_files:
    print(labeled_file)
    path=os.path.dirname(labeled_file)
    video_names.append(os.path.basename(path))
print(video_names, len(video_names))

convert_DLC_to_LP(TRAINING_DLC_FILE, save_data_dir, video_names)
