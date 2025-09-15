from collections import defaultdict
import pandas as pd
from pathlib import Path
import argparse
import os, sys
import numpy as np
from tqdm import tqdm
import h5py   
import csv

from lightning_pose.apps.utils import build_precomputed_metrics_df, get_col_names, concat_dfs
from lightning_pose.apps.plots import make_seaborn_catplot, make_plotly_catplot
from lightning_pose.metrics import (
    pixel_error,
    temporal_norm,
    pca_singleview_reprojection_error,
    pca_multiview_reprojection_error,
)
from lightning_pose.utils.pca import KeypointPCA

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



def plot_precomputed_traces(df_metrics, df_traces, cols, video_name):
    pix_error_key = "pixel error"
    conf_error_key = "confidence"
    temp_norm_error_key = "temporal norm"
    pcamv_error_key = "pca multiview"
    pcasv_error_key = "pca singleview"

    # -------------------------------------------------------------
    # setup
    # -------------------------------------------------------------
    coordinate = "x"  # placeholder
    keypoint = cols[0].split("_%s_" % coordinate)[0]
    colors = px.colors.qualitative.Plotly

    rows = 3
    row_heights = [2, 2, 0.75]
    if temp_norm_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)
    if pcamv_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)
    if pcasv_error_key in df_metrics.keys():
        rows += 1
        row_heights.insert(0, 0.75)

    fig_traces = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        x_title="Frame number",
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    yaxis_labels = {}
    row = 1

    # -------------------------------------------------------------
    # plot temporal norms, pcamv reproj errors, pcasv reproj errors
    # -------------------------------------------------------------
    for error_key in [temp_norm_error_key, pcamv_error_key, pcasv_error_key]:
        if error_key in df_metrics.keys():
            for c, col in enumerate(cols):
                # col = <keypoint>_<coord>_<model_name>.csv
                pieces = col.split("_%s_" % coordinate)
                if len(pieces) != 2:
                    # otherwise "_[x/y]_" appears in keypoint or model name :(
                    raise ValueError("invalid column name %s" % col)
                kp = pieces[0]
                model = pieces[1]
                fig_traces.add_trace(
                    go.Scatter(
                        name=col,
                        x=np.arange(df_traces.shape[0]),
                        y=df_metrics[error_key][kp][df_metrics[error_key].model_name == model],
                        mode='lines',
                        line=dict(color=colors[c]),
                        showlegend=False,
                    ),
                    row=row, col=1
                )
            if error_key == temp_norm_error_key:
                yaxis_labels['yaxis%i' % row] = "temporal<br>norm"
            elif error_key == pcamv_error_key:
                yaxis_labels['yaxis%i' % row] = "pca multi<br>error"
            elif error_key == pcasv_error_key:
                yaxis_labels['yaxis%i' % row] = "pca single<br>error"
            row += 1

    # -------------------------------------------------------------
    # plot traces
    # -------------------------------------------------------------
    for coord in ["x", "y"]:
        for c, col in enumerate(cols):
            pieces = col.split("_%s_" % coordinate)
            assert len(pieces) == 2  # otherwise "_[x/y]_" appears in keypoint or model name :(
            kp = pieces[0]
            model = pieces[1]
            new_col = col.replace("_%s_" % coordinate, "_%s_" % coord)
            fig_traces.add_trace(
                go.Scatter(
                    name=model,
                    x=np.arange(df_traces.shape[0]),
                    y=df_traces[new_col],
                    mode='lines',
                    line=dict(color=colors[c]),
                    showlegend=False if coord == "x" else True,
                ),
                row=row, col=1
            )
        yaxis_labels['yaxis%i' % row] = "%s coordinate" % coord
        row += 1

    # -------------------------------------------------------------
    # plot likelihoods
    # -------------------------------------------------------------
    for c, col in enumerate(cols):
        col_l = col.replace("_%s_" % coordinate, "_likelihood_")
        fig_traces.add_trace(
            go.Scatter(
                name=col_l,
                x=np.arange(df_traces.shape[0]),
                y=df_traces[col_l],
                mode='lines',
                line=dict(color=colors[c]),
                showlegend=False,
            ),
            row=row, col=1
        )
    yaxis_labels['yaxis%i' % row] = "confidence"
    row += 1

    # -------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------
    for k, v in yaxis_labels.items():
        fig_traces["layout"][k]["title"] = v
    fig_traces.update_layout(
        width=800, height=np.sum(row_heights) * 125,
        title_text="%s: Timeseries of %s" % (video_name, keypoint)
    )

    return fig_traces



def make_dlc_pandas_index(slp_model_name, keypoint_names) -> pd.MultiIndex:
    xyl_labels = ["x", "y", "likelihood"]
    pdindex = pd.MultiIndex.from_product(
        [["%s_tracker" % slp_model_name], keypoint_names, xyl_labels],
        names=["scorer", "bodyparts", "coords"],
    )
    return pdindex

# https://github.com/talmolab/sleap/issues/443
# https://github.com/talmolab/sleap/blob/1a0404c0ffae7b248eb360562b0bb95a42a287b6/sleap/io/dataset.py#L2691
def convert_slp_to_csv(fname_h5, fname_csv):    
    # if os.path.exists(fname_h5)==False:
    #     text = 'sleap-convert --format analysis ' + fname_slp # "/home/cat/Downloads/sleap_video.mp4.tracker_flow.slp"
    #     print ("Converting to .h5: ", text)
    #     os.system(text)

    print("fname_h5:", fname_h5)
    print("fname_csv:", fname_csv)

    if os.path.exists(fname_csv)==False:
    # if True:
        print ("")
        print ("... converting h5 to csv")
        # Open the HDF5 file using h5py.
        with h5py.File(fname_h5, "r") as f:
            # Print a list of the keys available.
            print("Keys in the HDF5 file:", list(f.keys()))

            keys = ['edge_inds', 'edge_names', 'instance_scores', 'labels_path', 'node_names', 'point_scores', 'provenance', 'track_names', 'track_occupancy', 'tracking_scores', 'tracks', 'video_ind', 'video_path']
            keys = ['instance_scores', 'node_names', 'point_scores',  'track_occupancy', 'tracking_scores', 'tracks']

            # Load all the datasets into a dictionary.
            # data = {k: v for k, v in f.items()}
            data = {k: f[k][:] for k in keys}

            # converting string arrays into regular Python strings.
            data["node_names"] = [s.decode() for s in data["node_names"].tolist()]
            # data["track_names"] = [s.decode() for s in data["track_names"].tolist()]

            # flip the order of the tracks axes for convenience.
            data["tracks"] = np.transpose(data["tracks"])
            data["point_scores"] = np.transpose(data["point_scores"])

            locations  = f["tracks"][:].T

            # convert the data type of the track occupancy array to boolean.
            data["track_occupancy"] = data["track_occupancy"].astype(bool)

        # Describe the values in the data dictionary we just created.
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: {value.dtype} array of shape {value.shape}")
            else:
                print(f"{key}: {value}")
    
        print("===locations data shape===")
        print(locations.shape) # (3274, 14, 2, 1)
        print()

        frame_count, node_count, _, instance_count = locations.shape
        print("frame count:", frame_count)
        print("node count:", node_count)
        print("instance count:", instance_count)

        # creat tracks
        num_frames = data['point_scores'].shape[0]
        pdindex = make_dlc_pandas_index('slp_UNet', data["node_names"])
        print("Total number of frames:", num_frames)

        # Get valid frame IDs
        valid_frame_idxs = np.argwhere(data["track_occupancy"].any(axis=1)).flatten()
        print("valid_frame_idxs:", len(valid_frame_idxs))
        print("tqdm(valid_frame_idxs)", len(tqdm(valid_frame_idxs)))

        new_index = pd.RangeIndex(num_frames)
        tracks = pd.DataFrame(np.nan, index=new_index, columns=pdindex)
        # for frame_idx in tqdm(valid_frame_idxs):
        for frame_idx in range(num_frames):
            # print("********************")
            # print(frame_idx)    
            # Get the tracking data for the current frame.
            frame_tracks = data["tracks"][frame_idx]
            frame_point_scores = data["point_scores"][frame_idx]
            # print("frame_tracks:", frame_tracks.shape) # (num_frames, num_nodes, 2, num_tracks)
            # print("frame_point_scores:", frame_point_scores.shape) # (num_frames, num_nodes, num_tracks)

            # Loop over the animals in the current frame.
            for i in range(frame_tracks.shape[-1]):
                pts = frame_tracks[..., i]
                scores = frame_point_scores[..., i]

            if np.isnan(pts).all():
                # Skip this animal if all of its points are missing (i.e., it wasn't
                # detected in the current frame).
                continue
        
            # fill in the coordinates for each body part.
            detection = []
            for node_name, (x, y), score in zip(data["node_names"], pts, scores):
                detection.append(x)
                detection.append(y)
                detection.append(score)
            # Add the row to the list and move on to the next detection.
            # print(len(detection))
            tracks.loc[frame_idx] = detection

        print(tracks.head())
        print(tracks.shape)

        # tracks.to_csv(fname_csv, index=False)
        tracks.to_csv(fname_csv)

    # # final step convert to .npy file
    # fname_npy = fname_slp[:-4]+'.analysis.npy'
    # if os.path.exists(fname_npy)==False:
    #     # csv loader to .npy
    #     traces =[]
    #     with open(fname_csv, mode='r') as infile:
    #         reader = csv.reader(infile)
    #         for row in reader:
    #             traces.append(row)
    #     traces.pop(0)
    #     np.save(fname_npy, traces)



from lightning_pose.utils.scripts import create_labeled_video
from moviepy.editor import VideoFileClip
# https://github.com/danbider/lightning-pose/blob/cbfd7128f0d959ff7a77f478c0f0d08006cbb9de/scripts/predict_new_vids.py#L98
# https://github.com/danbider/lightning-pose/blob/cbfd7128f0d959ff7a77f478c0f0d08006cbb9de/lightning_pose/utils/scripts.py#L652
def export_labeled_video(prediction_csv_file, video_file, labeled_mp4_file):
    """Export a labeled video for a single video file based on the predictions csv."""
    # labeled_mp4_file = prediction_csv_file.replace(".csv", "_labeled.mp4")

    preds_df = pd.read_csv(prediction_csv_file)

    print(preds_df.head())

    preds_df = preds_df.iloc[2: , 1:]

    print(preds_df.head())
    preds_df = preds_df.applymap(float)
    print(preds_df.dtypes)

    confidence_thresh_for_vid = 0.0
    # create labeled video
    if labeled_mp4_file is not None:
        print(labeled_mp4_file)
        os.makedirs(os.path.dirname(labeled_mp4_file), exist_ok=True)
        video_clip = VideoFileClip(video_file)
        # transform df to numpy array

        print(preds_df.head())
        print(preds_df.shape)

        keypoints_arr = np.reshape(preds_df.to_numpy(), [preds_df.shape[0], -1, 3])
        xs_arr = keypoints_arr[:, :, 0]
        ys_arr = keypoints_arr[:, :, 1]
        mask_array = keypoints_arr[:, :, 2] > confidence_thresh_for_vid

        # video generation
        create_labeled_video(
            clip=video_clip,
            xs_arr=xs_arr,
            ys_arr=ys_arr,
            mask_array=mask_array,
            filename=labeled_mp4_file,
        )



def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""
    # Store initial shape.
    initial_shape = Y.shape
    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        # Save slice
        Y[:, i] = y
    # Restore to initial shape.
    Y = Y.reshape(initial_shape)
    return Y


def plot_loc(bodypart_loc, bodypart, save_path):
    '''plot trajectory'''
    # sns.set('notebook', 'ticks', font_scale=1.2)
    plt.figure(figsize=(15,6), dpi=150, facecolor="w")
    plt.plot(bodypart_loc[:,0], 'y',label=bodypart)
    plt.plot(-1*bodypart_loc[:,1], 'y')

    plt.legend(loc="center right")
    plt.title(bodypart + ' locations')
    plt.savefig(save_path + '/' + bodypart + '_locations.png', bbox_inches = 'tight', pad_inches = 0.01)
    plt.close()

    plt.figure(figsize=(7,7))
    plt.plot(bodypart_loc[:,0],bodypart_loc[:,1], 'y',label=bodypart)
    plt.legend()

    plt.xlim(0,1024)
    plt.xticks([])
    plt.ylim(0,1024)
    plt.yticks([])
    plt.title(bodypart + ' tracks')
    plt.savefig(save_path + '/' + bodypart + '_tracks.png', bbox_inches = 'tight', pad_inches = 0.01)
    plt.close()

def plot_trajectory2(prediction_file):
    dframe = pd.read_csv(prediction_file, header=[1, 2], index_col=0)
    dframes_traces[model_name] = dframe
    dframes_metrics[model_name]["confidence"] = dframe

    locations  = f["tracks"][:].T
    
    frame_count, node_count, _, instance_count = locations.shape
    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)

    print("===nodes: trajectory===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}") 
        bodypart_loc = locations[:, i, :, 0] # the 1st instance
        if np.count_nonzero(~np.isnan(bodypart_loc)):
            bodypart_loc = fill_missing(bodypart_loc) # 8: jaw, 0: instance
            plot_loc(bodypart_loc, name, save_path)
        else:
            print(f"No trajectory for {name}")

def plot_trajectory(prediction_files, temporal_norm_files, save_dir, video_name):
    print("========="*3)
    print("prediction_files:", prediction_files)
    model_names  = ["SLEAP", 'DLC', "LP"]
    model_folders = ['/root/capsule/code/'] * 3
    # load data
    dframes_metrics = defaultdict(dict)
    dframes_traces = {}
    for p, model_pred_file, temp_norm_file in zip([0, 1, 2], prediction_files, temporal_norm_files):
        model_name = model_names[p]
        model_folder = model_folders[p]

        # model_pred_file_path = os.path.join(model_folder, "video_preds", model_pred_file)
        # model_pred_file_path = os.path.join(model_folder, model_pred_file)
        model_pred_file_path = model_pred_file
        print("model_pred_file:", model_pred_file_path)
        dframe = pd.read_csv(model_pred_file_path, header=[1, 2], index_col=0)
        dframes_traces[model_name] = dframe
        dframes_metrics[model_name]["confidence"] = dframe

        model_pred_file_path = temp_norm_file 
        print("model_pred_file:", model_pred_file_path)
        if "pca" in str(model_pred_file_path) or "temporal" in str(model_pred_file_path) or "pixel" in str(model_pred_file_path):
            dframe = pd.read_csv(model_pred_file_path, index_col=None)
            # dframes_metrics[model_name][str(model_pred_file)] = dframe
            dframes_metrics[model_name]["temporal norm"] = dframe
        data_types = dframe.iloc[:, -1].unique()

    # compute metrics
    # concat dataframes, collapsing hierarchy and making df fatter.
    print("-==============")
    print("dframes_metrics:", dframes_metrics)
    print("dframes_traces:", dframes_traces)
    print("-==============")

    df_concat, keypoint_names = concat_dfs(dframes_traces)
    df_metrics = build_precomputed_metrics_df(
        dframes=dframes_metrics, keypoint_names=keypoint_names)
    metric_options = list(df_metrics.keys())

    # print keypoint names; select one of these to plot below
    print("keypoint_names:", keypoint_names)
    print("metric_options:", metric_options)
    print("dframes_metrics.keys():", dframes_metrics.keys())

    for keypoint_name in keypoint_names:
        cols = get_col_names(keypoint_name, "x", dframes_metrics.keys())
        print("cols:", cols)
        fig_traces = plot_precomputed_traces(df_metrics, df_concat, cols, video_name)
        # save figures
        # fig_traces.savefig('/root/capsule/scratch/lightning-pose/testets.png', bbox_inches = 'tight', pad_inches = 0.01)
        # fig_traces.write_image(f"{keypoint_name}_TimeSeries.png", format='png')
        fig_traces.write_image(f"{save_dir}/{keypoint_name}_TimeSeries.png", format='png')

def get_keypoint_names(csv_file, header_rows):
    if os.path.exists(csv_file):
        csv_data = pd.read_csv(csv_file, header=header_rows)
        # collect marker names from multiindex header
        if header_rows == [1, 2] or header_rows == [0, 1]:
            # self.keypoint_names = csv_data.columns.levels[0]
            # ^this returns a sorted list for some reason, don't want that
            keypoint_names = [b[0] for b in csv_data.columns if b[1] == "x"]
        elif header_rows == [0, 1, 2]:
            # self.keypoint_names = csv_data.columns.levels[1]
            keypoint_names = [b[1] for b in csv_data.columns if b[2] == "x"]
    else:
        # keypoint_names = ["bp_%i" % n for n in range(cfg.data.num_targets // 2)]
        keypoint_names = []
        print("keypoint_names do not exist!!!")
    return keypoint_names


def compute_metrics_pixel_error(preds_file, labels_file):
    # assume dlc format, get keypoint names
    header_rows = [0, 1, 2]
    keypoint_names = get_keypoint_names(preds_file, header_rows)
    print("Number of keypoint_names:", len(keypoint_names))
    print("keypoint_names:", keypoint_names)

    # load ground truth
    labels_df = pd.read_csv(labels_file, header=[1, 2], index_col=0)
    # print("---"*5, "labels_df", "---"*5, )
    # print(labels_df.head())

    # ground truth
    keypoints_true = labels_df.to_numpy().reshape(labels_df.shape[0], -1, 2)
    # find the index of lebeled frame  within the corresponding video, trace every frame which have ground truth back to its origin,
    labels_df_1 = pd.read_csv(labels_file)
    selected_frames = labels_df_1['scorer'].to_list()[2:]
    index = []
    for frame_path in selected_frames:
        tp_index = int(frame_path.split('/')[-1][3:-4])
        index.append(tp_index)
    # print(index, len(index))

    # load video predictions
    pred_df_all_frames = pd.read_csv(preds_file , header=header_rows)
    # print("---"*5, "pred_df_all_frames", "---"*5, )
    # print(pred_df_all_frames.head())

    # prediction for frames with ground truth
    preds_file_only_frames_withGT = preds_file.replace(".csv", ".frames_withGT.csv")
    # print("---"*5, "preds_file_only_frames_withGT", "---"*5, )
    # print(preds_file_only_frames_withGT)

    pred_df_all_frames.iloc[index].to_csv(f"{preds_file_only_frames_withGT}", index=False) 


    # load frame predictions
    pred_df = pd.read_csv(preds_file_only_frames_withGT, header=header_rows, index_col=0)
    # print(pred_df.head())
    tmp = pred_df.to_numpy().reshape(pred_df.shape[0], -1, 3)
    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)

    index = labels_df.index
    # compute metrics; csv files will be saved to the same directory the prdictions are stored in
    error_per_keypoint = pixel_error(keypoints_true, keypoints_pred)
    error_df = pd.DataFrame(error_per_keypoint, index=index, columns=keypoint_names)

    save_file = preds_file.replace(".csv", ".pixel_error.csv")
    error_df.to_csv(save_file)



    #########################
    # # https://github.com/DeepLabCut/DeepLabCut/blob/6b7687e7cc0e52b100ff2abf26d24b87dd11fae8/deeplabcut/generate_training_dataset/frame_extraction.py#L373
    # video_file = 
    # from moviepy.editor import VideoFileClip
    # print(video_file)
    # clip = VideoFileClip(video_file)
    # fps = clip.fps
    # nframes = int(np.ceil(clip.duration * 1.0 * fps))
    # indexlength = int(np.ceil(np.log10(nframes)))

    # print(clip)
    # print("fps:", fps)
    # print("clip.duration:", clip.duration)
    # print("nframes:", nframes)
    # print("indexlength:", indexlength)
    # fps: 30.0
    # nframes: 3276
    # indexlength: 4
    # frame index: 0-3275
    #########################



def compute_metrics_temporal(preds_file, save_file):
    # # get keypoint names
    # labels_file = return_absolute_path(
    #     os.path.join(cfg["data"]["data_dir"], cfg["data"]["csv_file"]))
    # labels_df = pd.read_csv(labels_file, header=list(cfg["data"]["header_rows"]), index_col=0)
    # keypoint_names = get_keypoint_names(
    #     cfg, csv_file=labels_file, header_rows=list(cfg["data"]["header_rows"]))

    # assume dlc format
    header_rows = [0, 1, 2]
    keypoint_names = get_keypoint_names(preds_file, header_rows)

    # load predictions
    pred_df = pd.read_csv(preds_file, header=header_rows, index_col=0)
    print(pred_df.head())
    print("Number of keypoint_names:", len(keypoint_names))
    print("keypoint_names:", keypoint_names)

    if pred_df.keys()[-1][0] == "set":
        # these are predictions on labeled data
        # get rid of last column that contains info about train/val/test set
        is_video = False
        tmp = pred_df.iloc[:, :-1].to_numpy().reshape(pred_df.shape[0], -1, 3)
        index = labels_df.index
        set = pred_df.iloc[:, -1].to_numpy()
    else:
        # these are predictions on video data
        is_video = True
        tmp = pred_df.to_numpy().reshape(pred_df.shape[0], -1, 3)
        index = pred_df.index
        set = None

    keypoints_pred = tmp[:, :, :2]  # shape (samples, n_keypoints, 2)
    # confidences = tmp[:, :, -1]  # shape (samples, n_keypoints)

   # hard-code metrics for now
    if is_video:
        metrics_to_compute = ["temporal"]
    else:  # labeled data
        metrics_to_compute = ["pixel_error"]
    # for either labeled and unlabeled data, if a pca loss is specified in config, we compute the
    # associated metric

    columns_for_singleview_pca = False
    if columns_for_singleview_pca:
        metrics_to_compute += ["pca_singleview"]
    mirrored_column_matches = False
    if mirrored_column_matches:
        metrics_to_compute += ["pca_multiview"]

    print("metrics_to_compute:", metrics_to_compute)

    # compute metrics; csv files will be saved to the same directory the prdictions are stored in
    if "temporal" in metrics_to_compute:
        temporal_norm_per_keypoint = temporal_norm(keypoints_pred)
        temporal_norm_df = pd.DataFrame(
            temporal_norm_per_keypoint, index=index, columns=keypoint_names
        )
        # add train/val/test split
        if set is not None:
            temporal_norm_df["set"] = set
        temporal_norm_df.to_csv(save_file)


def compute_mean_error(pixel_error_files, alg_names, out_file):
    ''' compute mean pixel error and temporal loss'''
    # get keypoint names
    df = pd.read_csv(pixel_error_files[0])
    keypoint_names = list(df.columns[1:])

    with open(out_file, 'a') as f:
        fieldnames = ['Algorithm'] + keypoint_names + ['Average']
        write = csv.writer(f) 
        write.writerows([fieldnames]) 

    print(fieldnames)
    res = []
    for cur_file, alg_name in zip(pixel_error_files, alg_names):
        print(cur_file)
        df = pd.read_csv(cur_file)
        # print(df.head())
        # print()
        
        temp_list = []
        temp_list.append(alg_name)
        for col in df.columns[1:]:
            mean  = float( format(np.nanmean(df[col], axis=0), '.3f'))
            stdev = float( format(np.nanstd(df[col], axis=0), '.3f'))
            # print(f'{col}: {mean} ({stdev})')
            temp_list.append("{:.2f} ± {:.2f}".format(mean, stdev))
            # print("{}: {:.2f} ± {:.2f} pixel".format(col, mean, stdev))

        df = df.iloc[: , 1:]
        mean  = float( format(np.nanmean(df), '.3f'))
        stdev = float( format(np.nanstd(df), '.3f'))
        temp_list.append("{:.2f} ± {:.2f}".format(mean, stdev))
        print("Average pixel error: {:.2f} ± {:.2f} pixel".format(mean, stdev))

        print(temp_list)
        res.append(temp_list)

    with open(out_file, 'a') as f:
        write = csv.writer(f) 
        write.writerows(res)