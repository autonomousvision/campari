import torch
from gan_training.utils import (
    toggle_grad, compute_grad2, reshape_to_image, compute_bce,
    get_rays)
import torch.nn.functional as F
from torchvision.utils import make_grid
import tqdm
from gan_training.fid_score import calc_fid_score_from_tensor_and_dict
import numpy as np
from torch.distributions.kl import kl_divergence
from gan_training.training import update_average
import matplotlib.pyplot as plt
from kornia.filters import spatial_gradient


class Trainer():
    def __init__(self, model, evaluator, optimizer_g, optimizer_d,
                 n_eval_images=1000, n_vis_images=8, 
                 multi_gpu=False, pg_milestones=[20000, 80000], 
                 pg_batch_size=[64, 20, 5], pg_n_loop_final=2,  **kwargs):
        self.model = model
        self.device = model.device
        self.evaluator = evaluator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.n_eval_images = n_eval_images
        self.n_vis_images = n_vis_images
        self.pg_milestones = pg_milestones 
        self.bg_batch_size = pg_batch_size
        self.pg_n_loop_final = pg_n_loop_final

        self.discriminator = self.model.discriminator
        self.generator = self.model.generator
        self.generator_test = self.model.get_test_generator()

        if multi_gpu:
            print("Enabling Multi GPU Training.")
            self.generator.set_multi_gpu()
            self.generator_test.set_multi_gpu()
            self.discriminator = torch.nn.DataParallel(self.discriminator)

    def train_step(self, batch, it=0):
        dict_out = {}

        dict_g = self.train_step_generator(batch, it)
        dict_out.update(dict_g)

        dict_d = self.train_step_discriminator(batch, it=it)
        dict_out.update(dict_d)
        return dict_out

    def val_step(self, it=0):
        generator = self.generator_test
        generator.eval()

        print("Generating images for evaluation...")
        pbar = tqdm.tqdm(total=self.n_eval_images)
        img_eval = None
        while(True):
            with torch.no_grad():
                generator_dict = generator(sample_patch=False, batch_size=1)
            img_i = generator_dict['rgb'].cpu()
            if img_eval is None:
                img_eval = img_i
            else:
                img_eval = torch.cat([img_eval, img_i])

            pbar.update(img_i.shape[0])
            if img_eval.shape[0] >= self.n_eval_images:
                break

        print("done!")
        out_dict = self.evaluator.eval_fid_kid(img_eval)
        out_dict['images'] = make_grid(img_eval[np.random.choice(self.n_eval_images, size=(64,), replace=False)], nrow=8)
        return out_dict

    def vis_step(self, it=np.inf):
        if it < 5000:
            generator = self.generator
        else:
            generator = self.generator_test
            
        generator.eval()
        out_dict = {}
        random_samples = []
        for idx in tqdm.tqdm(range(self.n_vis_images)):
            with torch.no_grad():
                generator_dict = generator(sample_patch=False, batch_size=1)
            
            img_i = make_grid(torch.cat([
                generator_dict['rgb'].cpu(),
                generator_dict['rgb_fg'].cpu(),
                generator_dict['acc_fg'].cpu(),
                generator_dict['depth_fg'].cpu(),
                generator_dict['rgb_bg'].cpu(),
                generator_dict['acc_bg'].cpu(),
                generator_dict['depth_bg'].cpu(),
            ]), nrow=7)

            random_samples.append(img_i)
        out_dict['image_fake'] = make_grid(torch.stack(random_samples), nrow=1)

        with torch.no_grad():
            hist = generator.get_camera_histogram()
            out_dict['histogram'] = hist

        torch.cuda.empty_cache()
        return out_dict

    def vis_step_video(self,  it=np.inf):
        if it < 5000:
            generator = self.generator
        else:
            generator = self.generator_test
            
        generator.eval()
        out_dict = {}
        for idx in range(10):
            with torch.no_grad():
                out_i = generator.render_spiral(to_numpy=False, it=it).cpu()
            out_dict['%03d_spiral' % idx] = out_i
            if idx == 3:
                break
        return out_dict   

    def train_step_generator(self, batch, it=np.inf):
        generator = self.generator
        discriminator = self.discriminator
        optimizer_g = self.optimizer_g
        device = self.device

        # Generator Step
        out_dict = {}
        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        optimizer_g.zero_grad()

        n_loop = 1
        batch_size = None 
        if (len(self.pg_milestones) > 0) and it >= self.pg_milestones[-1]:
            n_loop = self.pg_n_loop_final
            assert(self.bg_batch_size[-1] % n_loop == 0)
            batch_size = self.bg_batch_size[-1] // n_loop
        for n_idx in range(n_loop):
            # Adv loss
            generator_dict = generator(it=it, batch_size=batch_size)
            if n_idx == 0:
                out_dict['n_ray'] = generator_dict['n_samples']
                out_dict['n_ray_bg'] = generator_dict['n_samples_bg']
                out_dict['img_size'] = generator_dict['resolution']
                if batch_size is not None:
                    out_dict['batch_size'] = batch_size * n_loop
                else:
                    out_dict['batch_size'] = generator_dict['batch_size']
                out_dict['pscale_min'] = generator_dict['pscale_min']
                out_dict['pscale_max'] = generator_dict['pscale_max']

            # Calc loss
            image_fake = generator_dict['rgb']
            d_fake = discriminator(image_fake, it=it)
            gloss = compute_bce(d_fake, 1)
            if n_idx == 0:
                out_dict['gloss'] = gloss

            gloss.backward()
        optimizer_g.step()
        torch.cuda.empty_cache()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)
        return out_dict

    def train_step_discriminator(self, batch, it=np.inf):
        generator = self.generator
        discriminator = self.discriminator
        optimizer_d = self.optimizer_d
        device = self.device


        out_dict = {}

        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        optimizer_d.zero_grad()

        n_loop = 1
        batch_size = None 
        image_real_list = [batch['image']]
        if (len(self.pg_milestones) > 0) and it >= self.pg_milestones[-1]:
            # n_loop = 2
            # batch_size = 5
            n_loop = self.pg_n_loop_final
            assert(self.bg_batch_size[-1] % n_loop == 0)
            batch_size = self.bg_batch_size[-1] // n_loop
            image_real_list = torch.split(batch['image'], batch_size, 0)
        for n_idx in range(n_loop):
            dloss = 0
            with torch.no_grad():
                generator_dict = generator(it=it, batch_size=batch_size)
            img_fake = generator_dict['rgb']
            #image_real = batch['image']
            image_real = image_real_list[n_idx]
            it = torch.ones(img_fake.shape[0], device=img_fake.device) * it

            sampling_pattern = generator_dict['pgrid']
            image_real = F.grid_sample(image_real.to(device), sampling_pattern, align_corners=True, mode='bilinear')

            image_real.requires_grad_()
            d_real = discriminator(image_real, it=it)
            d_loss_real = compute_bce(d_real, 1)
            dloss += d_loss_real
            out_dict['d_real'] = d_loss_real

            d_fake = discriminator(img_fake, it=it)
            d_loss_fake = compute_bce(d_fake, 0)
            dloss += d_loss_fake
            out_dict['d_fake'] = d_loss_fake
        
            reg = 10 * compute_grad2(d_real, image_real).mean()
            out_dict['d_reg'] = reg
            dloss += reg

            dloss.backward()
        optimizer_d.step()

        return out_dict
