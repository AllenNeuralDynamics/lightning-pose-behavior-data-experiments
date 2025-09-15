#!/usr/bin/env python
# coding: utf-8

# # ⚡ Train and visualize a Lightning Pose model ⚡
# 
# Using a toy dataset (a.k.a. "mirror-mouse") with 90 labeled images from Warren et al., 2022 (eLife).
# * [Environment setup](#Environment-setup)
# * [Train (via PyTorch Lightning)](#Training)
# * [Monitor optimization in real time (via TensorBoard UI)](#Monitor-training)
# * [Compute diagnostics on labeled and video data](#Compute-diagnostics)
# * [Compare train / val / test images (via FiftyOne UI)](#FiftyOne)
# * [Video predictions and diagnostics](#Plot-video-predictions-and-unsupervised-losses)
# 
# 
# **Make sure to use a GPU runtime!**
# 
# To do so, in the upper right corner of this notebook:
# * click the "Connect" button (or select "Connect to a hosted runtime" from the drop-down)
# * ensure you are connected to a GPU by clicking the down arrow, selecting "View resources" from the menu, and make sure you see "Python 3 Google Compute Engine backend (GPU)"

# ## Environment setup
import os, sys

# working_dir = "/root/capsule/scratch/lightning-pose/"
working_dir = "../scratch/lightning-pose/"

# optional: run our unit tests (takes 4-5 minutes on a T4)

# NOTE: you may see the following error:
#    RuntimeError: timed out waiting for adapter to connect
# As long as you see this on the following line:
#    ============================= test session starts ==============================
# then the notebook is working properly

# ## Trainingdf -h
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import lightning.pytorch as pl

from lightning_pose.utils import pretty_print_str, pretty_print_cfg
from lightning_pose.utils.io import (
    check_video_paths,
    return_absolute_data_paths,
    return_absolute_path,
)
from lightning_pose.utils.predictions import predict_dataset
from lightning_pose.utils.scripts import (
    export_predictions_and_labeled_video,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
    get_callbacks,
    calculate_train_batches,
    compute_metrics,
)


# read hydra configuration file from lightning-pose/scripts/configs/config_toy-dataset.yaml
cfg = OmegaConf.load(working_dir + "scripts/configs/config_toy-dataset.yaml")

# get absolute data and video directories for toy dataset
data_dir = os.path.join(working_dir, cfg.data.data_dir)
video_dir = os.path.join(working_dir, cfg.data.data_dir, cfg.data.video_dir)
cfg.data.data_dir = data_dir
cfg.data.video_dir = video_dir

assert os.path.isdir(cfg.data.data_dir), "data_dir not a valid directory"
assert os.path.isdir(cfg.data.video_dir), "video_dir not a valid directory"


# In[9]:


# build dataset, model, and trainer

# make training short for a demo (we usually do 300)
# cfg.training.min_epochs = 100
# cfg.training.max_epochs = 150
cfg.training.min_epochs = 10
cfg.training.max_epochs = 15
cfg.training.batch_size = 8

# build imgaug transform
imgaug_transform = get_imgaug_transform(cfg=cfg)

# build dataset
dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)

# build datamodule; breaks up dataset into train/val/test
data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

# build loss factory which orchestrates different losses
loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

# build model
model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)

# logger
logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)

# early stopping, learning rate monitoring, model checkpointing, backbone unfreezing
callbacks = get_callbacks(cfg)

# calculate number of batches for both labeled and unlabeled data per epoch
limit_train_batches = calculate_train_batches(cfg, dataset)

# set up trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=cfg.training.max_epochs,
    min_epochs=cfg.training.min_epochs,
    check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
    log_every_n_steps=cfg.training.log_every_n_steps,
    callbacks=callbacks,
    logger=logger,
    limit_train_batches=limit_train_batches,
)


# ## Monitor training

# get_ipython().run_line_magic('matplotlib', 'inline')

# # Load the TensorBoard notebook extension
# get_ipython().run_line_magic('load_ext', 'tensorboard')

# Launch tensorboard before launching training (happens in next cell).
# If you receive a 403 error, be sure to enable all cookies for this site in your browser.
# To see the losses during training, select TIME SERIES and hit the refresh button (circle arrow) on the top right.

# The two most important diagnostics are:
# - `train_supervised_rmse`: root mean square error (rmse) of predictions on training data
# - `val_supervised_rmse`: rmse on validation data

# get_ipython().run_line_magic('tensorboard', '--logdir tb_logs')


# Train the model (approx 15-20 mins on this T4 GPU machine)
trainer.fit(model=model, datamodule=data_module)


# ## Compute diagnostics
# * Get model predictions for train, validation, and test sets; use these to compute per-keypoint pixel errors and unsupervised losses
# * Get model predictions on a test video, and compute unsupervised losses

# ### Predictions/diagnostics for labeled data

# In[ ]:


from datetime import datetime

# Get the current date and time
now = datetime.now()

# Format the date and time as a string in the desired format
formatted_now = now.strftime("%Y-%m-%d/%H-%M-%S")

output_directory = os.path.join("/lightning-pose/outputs", formatted_now)
os.makedirs(output_directory)
print(f"Created an output directory at: {output_directory}")

# get best ckpt
best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)

# check if best_ckpt is a file
if not os.path.isfile(best_ckpt):
    raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")

# make unaugmented data_loader if necessary
if cfg.training.imgaug != "default":
    cfg_pred = cfg.copy()
    cfg_pred.training.imgaug = "default"
    imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
    dataset_pred = get_dataset(
        cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred
    )
    data_module_pred = get_data_module(cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir)
    data_module_pred.setup()
else:
    data_module_pred = data_module

# compute and save frame-wise predictions
pretty_print_str("Predicting train/val/test images...")
preds_file = os.path.join(output_directory, "predictions.csv")
predict_dataset(
    cfg=cfg,
    trainer=trainer,
    model=model,
    data_module=data_module_pred,
    ckpt_file=best_ckpt,
    preds_file=preds_file,
)

# compute and save various metrics
try:
    compute_metrics(cfg=cfg, preds_file=preds_file, data_module=data_module_pred)
except Exception as e:
    print(f"Error computing metrics\n{e}")


# In[ ]:


artifacts = os.listdir(output_directory)
print("Generated the following diagnostic csv files:")
print(artifacts)


# ### Predictions/diagnostics for example video

# In[ ]:


# for this demo data, we define
cfg.eval.test_videos_directory = video_dir
# feel free to change this according to the folder you want to predict
assert os.path.isdir(cfg.eval.test_videos_directory)

if cfg.eval.test_videos_directory is None:
    filenames = []
else:
    filenames = check_video_paths(return_absolute_path(cfg.eval.test_videos_directory))
    vidstr = "video" if (len(filenames) == 1) else "videos"
    pretty_print_str(
        f"Found {len(filenames)} {vidstr} to predict on (in cfg.eval.test_videos_directory)")

for video_file in filenames:
    assert os.path.isfile(video_file)
    pretty_print_str(f"Predicting video: {video_file}...")
    # get save name for prediction csv file
    video_pred_dir = os.path.join(output_directory, "video_preds")
    video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
    prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
    # get save name labeled video csv
    if cfg.eval.save_vids_after_training:
        labeled_vid_dir = os.path.join(video_pred_dir, "labeled_videos")
        labeled_mp4_file = os.path.join(labeled_vid_dir, video_pred_name + "_labeled.mp4")
    else:
        labeled_mp4_file = None
    # predict on video
    export_predictions_and_labeled_video(
        video_file=video_file,
        cfg=cfg,
        ckpt_file=best_ckpt,
        prediction_csv_file=prediction_csv_file,
        labeled_mp4_file=labeled_mp4_file,
        trainer=trainer,
        model=model,
        gpu_id=cfg.training.gpu_id,
        data_module=data_module_pred,
        save_heatmaps=cfg.eval.get("predict_vids_after_training_save_heatmaps", False),
    )
    # compute and save various metrics
    try:
        compute_metrics(
            cfg=cfg, preds_file=prediction_csv_file, data_module=data_module_pred
        )
    except Exception as e:
        print(f"Error predicting on video {video_file}:\n{e}")
        continue


# # ### Display the short labeled video
# # Includes network predictions.
# # Make sure your video is not too large for this; it may cause memory issues.
# # 

# # In[ ]:


# from IPython.display import HTML
# from base64 import b64encode

# vids = os.listdir(labeled_vid_dir)
# mp4 = open(os.path.join(labeled_vid_dir, vids[0]),'rb').read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
# HTML("""
# <video width=400 controls>
#       <source src="%s" type="video/mp4">
# </video>
# """ % data_url)


# # In[ ]:


# # download vids to your local machine if desired
# from google.colab import files
# for vid in vids:
#     if vid.endswith(".mp4"):
#         files.download(os.path.join(labeled_vid_dir, vid))


# ## FiftyOne
# We use `fiftyone` to visualize our models' predictions on labeled images. We will create a dataset with predictions, and then display it in a UI below.

# In[ ]:


# Override the default configs here:
cfg.eval.hydra_paths=[output_directory] # you can add multiple output_directory2, output_directory3 to compare
cfg.eval.fiftyone.dataset_to_create="images"
cfg.eval.fiftyone.dataset_name="lightning-demo-colab"
cfg.eval.fiftyone.build_speed="fast"
cfg.eval.fiftyone.model_display_names=["semi"]


# In[ ]:


import fiftyone as fo
from lightning_pose.utils.fiftyone import (
    FiftyOneImagePlotter,
    FiftyOneKeypointVideoPlotter,
    check_dataset,
    FiftyOneFactory,
)

# initializes everything
FiftyOneClass = FiftyOneFactory(
        dataset_to_create=cfg.eval.fiftyone.dataset_to_create
    )()
fo_plotting_instance = FiftyOneClass(cfg=cfg)

# internally loops over models
dataset = fo_plotting_instance.create_dataset()

# create metadata and print if there are problems
check_dataset(dataset)
fo_plotting_instance.dataset_info_print()


# In[ ]:


# Launch the FiftyOne UI
# - Select the dataset you just built (e.g., `lightning-pose-demo`) on the top left dropdown menu.
# - If you receive a 403 error, be sure to enable all cookies for this site in your browser
fo.launch_app()


# # ## Plot video predictions and unsupervised losses

# # ### Load data

# # In[ ]:


# from collections import defaultdict
# import pandas as pd
# from pathlib import Path

# from lightning_pose.apps.utils import build_precomputed_metrics_df, get_col_names, concat_dfs
# from lightning_pose.apps.utils import update_vid_metric_files_list
# from lightning_pose.apps.utils import get_model_folders, get_model_folders_vis
# from lightning_pose.apps.plots import plot_precomputed_traces

# # select which model(s) to use
# model_folders = get_model_folders("/lightning-pose/outputs")

# # get the last two levels of each path to be presented to user
# model_names = get_model_folders_vis(model_folders)

# # get prediction files for each model
# prediction_files = update_vid_metric_files_list(video="test_vid", model_preds_folder=model_folders)

# # load data
# dframes_metrics = defaultdict(dict)
# dframes_traces = {}
# for p, model_pred_files in enumerate(prediction_files):
#     model_name = model_names[p]
#     model_folder = model_folders[p]
#     for model_pred_file in model_pred_files:
#         model_pred_file_path = os.path.join(model_folder, "video_preds", model_pred_file)
#         if not isinstance(model_pred_file, Path):
#             model_pred_file.seek(0)  # reset buffer after reading
#         if "pca" in str(model_pred_file) or "temporal" in str(model_pred_file) or "pixel" in str(model_pred_file):
#             dframe = pd.read_csv(model_pred_file_path, index_col=None)
#             dframes_metrics[model_name][str(model_pred_file)] = dframe
#         else:
#             dframe = pd.read_csv(model_pred_file_path, header=[1, 2], index_col=0)
#             dframes_traces[model_name] = dframe
#             dframes_metrics[model_name]["confidence"] = dframe
#         data_types = dframe.iloc[:, -1].unique()

# # compute metrics
# # concat dataframes, collapsing hierarchy and making df fatter.
# df_concat, keypoint_names = concat_dfs(dframes_traces)
# df_metrics = build_precomputed_metrics_df(
#     dframes=dframes_metrics, keypoint_names=keypoint_names)
# metric_options = list(df_metrics.keys())

# # print keypoint names; select one of these to plot below
# print(keypoint_names)

# # NOTE: you can ignore all errors and warnings of the type:
# #    No runtime found, using MemoryCacheStorageManager


# # ### Plot video traces

# # In[ ]:


# # rerun this cell each time you want to update the keypoint

# from IPython.display import display, clear_output
# import ipywidgets as widgets

# def on_change(change):
#     if change["type"] == "change" and change["name"] == "value":
#         clear_output()
#         cols = get_col_names(change["new"], "x", dframes_metrics.keys())
#         fig_traces = plot_precomputed_traces(df_metrics, df_concat, cols)
#         fig_traces.show()

# # create a Dropdown widget
# dropdown = widgets.Dropdown(
#     options=keypoint_names,
#     value=None,  # Set the default selected value
#     description="Select keypoint:",
# )

# # update plot upon change
# dropdown.observe(on_change)

# # display widget
# display(dropdown)

