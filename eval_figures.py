import torch
from os.path import join
from os import makedirs
import tqdm
import numpy as np
import argparse
from gan_training import config, data, checkpoint
from gan_training.utils import reshape_to_image, to8b
from gan_training.fid_score import calc_fid_score_from_tensor_and_dict
from torchvision.utils import save_image, make_grid
import imageio
import torch.nn.functional as F
import pandas as pd


parser = argparse.ArgumentParser(description='Evaluate a model.')
parser.add_argument('config', type=str,
                    help='Path to config file')
args = parser.parse_args()


config_dict = config.process_config(args.config)

np.random.seed(0)
torch.manual_seed(0)

data_loader, data_loader_fid = data.get_dataloader(config_dict)

# path
out_folder = config_dict['model']['out_folder']
makedirs(out_folder, exist_ok=True)
out_folder_eval = join(out_folder, 'eval')
makedirs(out_folder_eval, exist_ok=True)

# Model
model = config.get_model(config_dict, mode="test")
model_file = config_dict['test']['model_file']
# Checkpoint
checkpoint_io = checkpoint.CheckpointIO(
    out_folder, model=model)
try:
    load_dict = checkpoint_io.load(model_file)
    print("Loaded model %s from checkpoint." % model_file)
except FileExistsError:
    load_dict = dict()
    print("No model checkpoint found.")

best_eval_metric = load_dict.get('best_eval_metric', np.inf)
eval_metric = config_dict['training']['eval_metric']
test_mode = config_dict['test']['mode']

render_program = [
    'rotation',
    #'rotation_fix10',
    #'rotation_fixfull',
    #'rotation_fixfull7',
    #'rotation_step7',
    'elevation',
    'disentanglement',
    # 'zoom',
    #'focal_zoom',
    #'focal_zoomx',
    #'focal_zoomy',
    #'interpolate_shape',
    #'interpolate_app',
    #'histogram',
    #'stats_file',
]

# Generator
generator = model.get_test_generator()

if 'rotation' in render_program:
    # Rotation
    out_folder_rotation = join(out_folder_eval, 'rotation')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_rotation(tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'rotation_step7' in render_program:
    # Rotation
    out_folder_rotation = join(out_folder_eval, 'rotation7')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_rotation(n_steps=7, tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'rotation_fix10' in render_program:
    # Rotation
    out_folder_rotation = join(out_folder_eval, 'rotation_fix10')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_rotation(fix_rot_range=[-10, 10], tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'rotation_fixfull' in render_program:
    # Rotation
    out_folder_rotation = join(out_folder_eval, 'rotation_fixfull')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_rotation(fix_rot_range=[-180, 120], ele_val=-10, tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'rotation_fixfull7' in render_program:
    # Rotation
    out_folder_rotation = join(out_folder_eval, 'rotation_fixfull7')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_rotation(fix_rot_range=[-180, 120], ele_val=-10, n_steps=7, tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'elevation' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'elevation')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_elevation(tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'disentanglement' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'disentanglement')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_disentanglement(tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'zoom' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'zoom7')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_zoom(tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'focal_zoom' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'focal_zoom')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_focal_zoom(tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'focal_zoomx' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'focal_zoom_x')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_focal_zoom(focal="x", tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'focal_zoomy' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'focal_zoom_y')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_focal_zoom(focal="y", tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))
if 'interpolate_shape' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'interpolate_shape')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_interpolate(tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'interpolate_app' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'interpolate_app')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        img = generator.render_interpolate(shape=False, tmp=0.6)
    for (k, v) in img.items():
        out_folder_k = join(out_folder_rotation, k) 
        makedirs(out_folder_k, exist_ok=True)
        for idx, img_i in enumerate(v):
            save_image(img_i, join(out_folder_k, '%05d.jpg' % idx))

if 'histogram' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'histogram')
    makedirs(out_folder_rotation, exist_ok=True)
    out_file_rot = join(out_folder_rotation, 'hist_rotation.png')
    out_file_ele = join(out_folder_rotation, 'hist_elevation.png')
    out_file_radius = join(out_folder_rotation, 'hist_radius.png')
    with torch.no_grad():
        generator.render_histogram(out_file_rot)
        generator.render_histogram(out_file_ele, mode="elevation")
        generator.render_histogram(out_file_radius, mode="radius")

if 'stats_file' in render_program:
    # Elevation
    out_folder_rotation = join(out_folder_eval, 'camera_stats')
    makedirs(out_folder_rotation, exist_ok=True)
    with torch.no_grad():
        stats_file = generator.create_camera_stats_file()
    np.savez(join(out_folder_rotation, 'stats.npz'), **stats_file)

