import torch
from os.path import join
from os import makedirs
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from gan_training import config, data, checkpoint
from gan_training.utils import to8b, log_and_print
from torchvision.utils import save_image
import time 
import imageio
import logging
from copy import deepcopy
import random 


parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('config', type=str,
                    help='Path to config file')
parser.add_argument('--restart-after', type=int, default=-1,
                    help='Restart after this seconds')
args = parser.parse_args()


config_dict = config.process_config(args.config)

# set random seed
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True
setup_seed(41)


data_loader, data_loader_fid_gt = data.get_dataloader(config_dict)

# path
out_folder = config_dict['model']['out_folder']
makedirs(out_folder, exist_ok=True)
out_folder_vis = join(out_folder, 'vis')
makedirs(out_folder_vis, exist_ok=True)
tb_writer = SummaryWriter(join(out_folder, 'log'))
logging.basicConfig(filename=join(out_folder, 'logging.log'), level=logging.INFO)

# Model
model = config.get_model(config_dict)
if hasattr(model, "generator"):
    print(model.generator)
    nparameters = sum(p.numel() for p in model.generator.parameters())
    print("Generator parameters: %d" % nparameters)
if hasattr(model, "discriminator"):
    nparameters = sum(p.numel() for p in model.discriminator.parameters())
    print(model.discriminator)
    print("Discriminator parameters: %d" % nparameters)

# Optimizer
optimizer_g, optimizer_d = config.get_optimizer(model, config_dict)
# Evaluator
evaluator = config.get_evaluator(config_dict, data_loader_fid_gt)
# Trainer
trainer = config.get_trainer(model, evaluator, optimizer_g, optimizer_d, config_dict)
# Checkpoint
if optimizer_d is None:
    checkpoint_io = checkpoint.CheckpointIO(
        out_folder, model=model, optimizer_g=optimizer_g)
else:
    checkpoint_io = checkpoint.CheckpointIO(
        out_folder, model=model, optimizer_g=optimizer_g, optmizer_d=optimizer_d)
try:
    load_dict = checkpoint_io.load('model.pt')
    log_and_print(logging, "Loaded model checkpoint.")
except FileExistsError:
    load_dict = dict()
    log_and_print(logging, "No model checkpoint found.")

epoch_it = load_dict.get('epoch_it', 0)
it_total = load_dict.get('it', 0)
it_load = load_dict.get('it', 0)
t_total = load_dict.get('time', 0)
best_eval_metric = load_dict.get('best_eval_metric', np.inf)
n_iter = config_dict['training']['n_iter']
print_every = config_dict['training']['print_every']
visualize_every = config_dict['training']['visualize_every']
eval_every = config_dict['training']['eval_every']
checkpoint_every = config_dict['training']['checkpoint_every']
visualize_video_every = config_dict['training']['visualize_video_every']
eval_metric = config_dict['training']['eval_metric']
lr_decay = config_dict['training']['lr_decay']
lr_g = config_dict['training']['lr_g']
lr_d = config_dict['training']['lr_d']
exit_after = config_dict['training']['exit_after']
if (args.restart_after is not None) and (args.restart_after > 0):
    exit_after = args.restart_after
decay_steps = config_dict['training']['decay_steps']
pg_milestones = config_dict['training']['pg_milestones']
backup_every = config_dict['training']['backup_every']


t0_exit_after = time.time()
mean_t, n_t = 0, 0
while True:
    idx_pg = 0
    if len(pg_milestones) > 0:
        for pg_ms in pg_milestones:
            if it_total > pg_ms:
                idx_pg += 1
    data_loader_i = data_loader[idx_pg]
    for it_batch, batch in enumerate(data_loader_i):
        
        t0 = time.time()
        loss_dict = trainer.train_step(batch, it_total)
        ti = time.time() - t0

        t_total += ti
        n_t += 1

        mean_t = mean_t + (1 / n_t) * (ti - mean_t)

        if it_total % print_every == 0:
            txt_print = '(It %d, Epoch %d, time (iter): %.2fs, time (total): %.2fm %s (val): %.4f): '
            for (k, v) in loss_dict.items():
                if type(v) not in [float, int]:
                    v = v.detach().cpu().item()
                tb_writer.add_scalar(k, v, it_total)
                txt_print += '%s: ' % k + '%.4f '
            txt_print = txt_print % (
                it_total, epoch_it, mean_t, t_total / 60., eval_metric, best_eval_metric,
                *loss_dict.values())
            log_and_print(logging, txt_print)

        if (visualize_every > 0) and (((it_total % visualize_every == 0) or (it_total in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]))):
            log_and_print(logging, "Visualizing ...")
            out_dict_vis = trainer.vis_step(it=it_total)
            for (k, v) in out_dict_vis.items():
                tb_writer.add_image(k, v, it_total)
                save_image(v, join(out_folder_vis, '%010d_%s.jpg' % (it_total, k)))
            log_and_print(logging, "done!")

        if (visualize_video_every > 0) and (it_total % visualize_video_every == 0) and (it_total > 0):
            log_and_print(logging, "Visualizing Video ...")
            out_dict_vis = trainer.vis_step_video(it=it_total)
            for (k, v) in out_dict_vis.items():
                video_i = v.permute(0, 2, 3, 1).numpy()
                imageio.mimwrite(join(out_folder_vis, '%010d_%s.mp4' % (it_total, k)), to8b(video_i), fps=30, quality=8) 
            log_and_print(logging, "done!")

        if ((eval_every > 0) and (it_total > 0) and (it_total % eval_every == 0)) or (it_total == n_iter):
            eval_dict = trainer.val_step(it=it_total)
            # Write scalars
            for (k, v) in eval_dict.items():
                if k != 'images':
                    tb_writer.add_scalar(k, v, it_total) 
            # Get eval images grid
            save_image(eval_dict['images'], join(out_folder_vis, '%010d_eval.jpg' % it_total))
            tb_writer.add_image('eval_image', eval_dict['images'], it_total)
            eval_i = eval_dict.get(eval_metric)
            if eval_i < best_eval_metric:
                best_eval_metric = eval_i
                log_and_print(logging, "Found new best best model: %.4f %s" % (
                    best_eval_metric, eval_metric))
                checkpoint_io.backup_model_best('model_best.pt')
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it,
                                   it=it_total, time=t_total,
                                   best_eval_metric=best_eval_metric)
        
        if ((checkpoint_every > 0) and (it_total > it_load) 
                and (it_total % checkpoint_every == 0)) or (it_total == n_iter):
            checkpoint_io.save(
                'model.pt', epoch_it=epoch_it, it=it_total,
                time=t_total, best_eval_metric=best_eval_metric)
            log_and_print(logging, 'Saved checkpoint.')

        if (backup_every > 0) and ((it_total % backup_every == 0) or it_total==5000) and (it_total > 0):
            checkpoint_io.save(
                'model_backup_%07d.pt' % it_total, epoch_it=epoch_it, it=it_total,
                time=t_total, best_eval_metric=best_eval_metric)
            log_and_print(logging, 'Saved checkpoint.')

        if (exit_after > 0) and ((time.time() - t0_exit_after) >= exit_after):
            log_and_print(logging, "Exit time reached: %.2fs" % (time.time() - t0_exit_after))
            checkpoint_io.save(
                'model.pt', epoch_it=epoch_it, it=it_total,
                time=t_total, best_eval_metric=best_eval_metric)
            log_and_print(logging, 'Saved checkpoint.')
            log_and_print(logging, "Exiting with code 3.")
            exit(3)

        if lr_decay:
            decay_rate = 0.1
            decay_steps = decay_steps * 1000
            decay_mult = (decay_rate ** (it_total / decay_steps))
            new_lr_g = lr_g * decay_mult
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = new_lr_g
            if model.discriminator is not None:
                new_lr_d = lr_d * decay_mult
                for param_group in optimizer_d.param_groups:
                    param_group['lr'] = new_lr_d

            if it_total % print_every == 0:
                tb_writer.add_scalar('lr_decay', decay_mult, it_total) 

        if it_total == n_iter:
            log_and_print(logging, "Final number of %d iterations reached!" % n_iter)
            log_and_print(logging, "Best Validation metric %.4f %s" % (best_eval_metric, eval_metric))
            exit(0)

        it_total += 1

        # Check if dataloader needs to be swapped
        idx_pg_i = 0
        if len(pg_milestones) > 0:
            for pg_ms in pg_milestones:
                if it_total > pg_ms:
                    idx_pg_i += 1
            if idx_pg_i > idx_pg:
                print("Breaking now at %d " % it_total)
                break

    epoch_it += 1
