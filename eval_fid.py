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


parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('config', type=str,
                    help='Path to config file')
args = parser.parse_args()


config_dict = config.process_config(args.config)

np.random.seed(0)
torch.manual_seed(0)

# path
out_folder = config_dict['model']['out_folder']
makedirs(out_folder, exist_ok=True)
out_folder_eval = join(out_folder, 'eval')
makedirs(out_folder_eval, exist_ok=True)

# Model
model = config.get_model(config_dict, mode="test")
model_file = config_dict['test']['model_file']
print("Using model file: %s" % model_file)
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

evaluator = config.get_evaluator(config_dict, None)

n_img = config_dict['test']['n_test_images']

out_imgs = []
for idx in tqdm.tqdm(range(n_img)):
    with torch.no_grad():
        img_i = generator(batch_size=1, sample_patch=False)['rgb']
    out_imgs.append(img_i.cpu())

out_imgs = torch.cat(out_imgs)
out_folder_eval = join(out_folder_eval, 'fid')
makedirs(out_folder_eval, exist_ok=True)

save_image(make_grid(out_imgs[:100], nrow=10), join(out_folder_eval, 'vis.jpg'))
np.save(join(out_folder_eval, 'eval_images.npy'), (out_imgs * 255).numpy().astype(np.uint8))


eval_dict = evaluator.eval_fid_kid(out_imgs , batch_size=min(out_imgs.shape[0], 200))
print(eval_dict)
np.savez(join(out_folder_eval, 'stats_fid_kid.npz'), **eval_dict)

