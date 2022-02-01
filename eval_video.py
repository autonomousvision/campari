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

# Generator
generator = model.get_test_generator()

out_folder_spiral = join(out_folder_eval, 'video_individual')
makedirs(out_folder_spiral, exist_ok=True)
for idx in tqdm.tqdm(range(64)):
    with torch.no_grad():
        # Rotation
        img, latent_codes = generator.render_spiral(rand_seed=idx, with_zoom=False, fix_ele=True, return_latent_codes=True, increase_size=True, sample_tmp=0.6)
        imageio.mimwrite(join(out_folder_spiral, '%04d_rotation.mp4' % idx), img, fps=30, quality=10) 
        # Elevation
        # img = generator.render_spiral(rand_seed=idx, with_zoom=False, zoom_range=[0.5, 1.2], fix_rot=True, ele_sin=True, latent_codes=latent_codes, increase_size=True, sample_tmp=0.6)
        # imageio.mimwrite(join(out_folder_spiral, '%04d_elevation.mp4' % idx), img, fps=30, quality=10) 
        # Zoom
        # img = generator.render_spiral(rand_seed=idx, with_zoom=True, zoom_range=[0.5, 1.], fix_rot=True, fix_ele=True, n_steps=64, latent_codes=latent_codes, increase_size=True, sample_tmp=0.6)
        # imageio.mimwrite(join(out_folder_spiral, '%04d_zoom.mp4' % idx), img, fps=30, quality=10) 
        # Full
        # img = generator.render_spiral(rand_seed=idx, with_zoom=True, zoom_range=[0.5, 1.], ele_sin=True, latent_codes=latent_codes, increase_size=True, sample_tmp=0.6)
        # imageio.mimwrite(join(out_folder_spiral, '%04d_full.mp4' % idx), img, fps=30, quality=10) 
