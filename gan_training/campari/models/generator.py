from torch import nn
from gan_training.utils import (interpolate_sphere, pose_spherical_b,
                                get_focal_from_fov, perturb_samples,
                                depth2pts_outside, color_depth_map_tensor,
                                pose_spherical_ele, get_camera_rays,
                                get_rot_theta_from_2x2, pose_spherical_ele_rot,
                                project_to_so)
import torch
import numpy as np
from torch.distributions.normal import Normal
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from gan_training.utils import reshape_to_image
from kornia.filters import spatial_gradient
import os
from pdb import set_trace as st
from kornia import create_meshgrid
from scipy.stats import kstest, ks_2samp

GRID_PAD_VALUE = 1.


class Generator(nn.Module):
    def __init__(self,
                 device="cuda",
                 decoder=None,
                 object_planes=[-0.5, 0.5],
                 img_size=64,
                 white_background=False,
                 c_dim=256,
                 raw_noise_std=1.,
                 latent_cameras=None,
                 radius_range=[0.75, 0.25],
                 rot_range=[0, 90],
                 ele_range=[0, 90],
                 fov_range=[35., 25.],
                 residual_pose=True,
                 sample_milestones=[5000, 50000],
                 camera_normal_std=0.15,
                 init_pose_iter=500,
                 decoder_bg=None,
                 camera_param='normal',
                 add_noise_bg=True,
                 pg_milestones=[20000, 50000],
                 pg_resolution0=32,
                 batch_size_pg=[128, 24, 6],
                 n_samples_ps=[14, 20, 24],
                 is_test_mode=False,
                 n_test_samples=32,
                 test_img_size=64,
                 gt_stats_file=None,
                 priorcamtype='gauss',
                 fix_cam_after_50k=False,
                 fix_uniform_instrinsics=False,
                 not_learn_instrinsics=False,
                 fix_gauss_instrinsics=False):
        super().__init__()
        self.device = device
        self.priorcamtype = priorcamtype
        self.object_planes = object_planes
        self.img_size = img_size
        self.white_background = white_background
        self.c_dim = c_dim
        self.raw_noise_std = raw_noise_std
        self.radius_range = radius_range
        self.rot_range = rot_range
        self.ele_range = ele_range
        self.fov_range = fov_range
        self.residual_pose = residual_pose
        self.sample_milestones = sample_milestones
        self.camera_normal_std = camera_normal_std
        self.init_pose_iter = init_pose_iter
        self.camera_param = camera_param
        self.fix_cam_after_50k = fix_cam_after_50k
        self.add_noise_bg = add_noise_bg
        self.batch_size_pg = batch_size_pg
        self.n_samples_ps = n_samples_ps

        self.pg_milestones = pg_milestones
        self.pg_resolution0 = pg_resolution0

        self.n_test_samples = n_test_samples
        self.is_test_mode = is_test_mode
        self.test_img_size = test_img_size
        self.gt_stats_file = gt_stats_file
        self.not_learn_instrinsics = not_learn_instrinsics
        self.fix_gauss_instrinsics = fix_gauss_instrinsics
        self.fix_uniform_instrinsics = fix_uniform_instrinsics

        if is_test_mode:
            print("#" * 100)
            print(
                "Test Generator created. Using %d ray samples and %d as image resolution."
                % (n_test_samples, test_img_size))
            print("#" * 100)

        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None

        if latent_cameras:
            self.latent_cameras = latent_cameras.to(device)
        else:
            self.latent_cameras = None

        if decoder_bg is not None:
            self.decoder_bg = decoder_bg.to(device)
        else:
            self.decoder_bg = None

    def estimate_moment(self, tensor, key='rot'):
        mean = tensor.mean()
        std = tensor.std()
        return mean, std

    def estimate_moments(self):
        prior_cam = self.sample_prior_cam(100000)
        post_cam_dict = self.get_post_cam(prior_cam)
        out_dict = {}
        keys = ['fx', 'fy', 'rot', 'ele', 'radius']
        for k in keys:
            v = post_cam_dict[k]
            mv, stdv = self.estimate_moment(v, key=k)
            out_dict['%s_mean' % k] = mv
            out_dict['%s_std' % k] = stdv
            out_dict['%s_min' % k] = v.min()
            out_dict['%s_max' % k] = v.max()
            out_dict['%s_minus_std' % k] = mv - stdv
            out_dict['%s_minus_std2' % k] = mv - stdv * 2
            out_dict['%s_plus_std' % k] = mv + stdv
            out_dict['%s_plus_std2' % k] = mv + stdv * 2

        return out_dict

    def create_camera_stats_file(self):
        prior_cam = self.sample_prior_cam(100000)
        post_cam = self.get_post_cam(prior_cam)

        if self.camera_param == 'normal':
            prior_scaled = self.rescale_prior_cam(prior_cam)
            prior_scaled = prior_scaled.detach().cpu().numpy()
            fovx_prior = prior_scaled[..., 0]
            fovy_prior = prior_scaled[..., 1]
            rot_prior = prior_scaled[..., 2]
            radius_prior = prior_scaled[..., 4]
            ele_prior = prior_scaled[..., 3]
        else:
            prior_scaled, rot_prior = self.rescale_prior_cam(prior_cam)
            prior_scaled = prior_scaled.detach().cpu().numpy()
            rot_prior = rot_prior.detach().cpu().numpy()
            radius_prior = prior_scaled[..., 4]
            ele_prior = prior_scaled[..., 2]
            fovx_prior = prior_scaled[..., 0]
            fovy_prior = prior_scaled[..., 1]

        keys = ['fovx', 'fovy', 'radius', 'elevation', 'rotation']
        post_list = [
            post_cam['fx'].detach().cpu().numpy(),
            post_cam['fy'].detach().cpu().numpy(),
            post_cam['radius'].detach().cpu().numpy(),
            post_cam['ele'].detach().cpu().numpy(),
            post_cam['rot'].detach().cpu().numpy(),
        ]
        prior_list = [
            fovx_prior, fovy_prior, radius_prior, ele_prior, rot_prior
        ]
        out_dict = {}
        for (v_prior, v_post, k) in zip(prior_list, post_list, keys):
            out_dict['prior_%s' % k] = v_prior
            out_dict['post_%s' % k] = v_post

        if self.gt_stats_file is not None:
            gt_stats = np.load(self.gt_stats_file)
            gt_rot, gt_ele = gt_stats['rot'], gt_stats['ele']
            out_dict['gt_rotation'] = gt_rot
            out_dict['gt_elevation'] = gt_ele
            # kstest

        return out_dict

    def set_multi_gpu(self):
        self.decoder = torch.nn.DataParallel(self.decoder)
        if self.decoder_bg is not None:
            self.decoder_bg = torch.nn.DataParallel(self.decoder_bg)

    def restructure_grid_list(self, grid_list, keys=["rgb", "depth_fg"]):
        out_dict = {}
        for k in keys:
            out_dict[k] = []

        for grid_img in grid_list:
            grid_tmp = {}
            for k in keys:
                grid_tmp[k] = []
            for img in grid_img:
                for k in keys:
                    if 'depth' in k:
                        depth = color_depth_map_tensor(img[k][:, 0])
                        imgi = depth
                    else:
                        imgi = img[k]
                    grid_tmp[k].append(imgi)
            for (k, v) in grid_tmp.items():
                v = make_grid(torch.cat(v), pad_value=GRID_PAD_VALUE)
                out_dict[k].append(v)
        return out_dict

    def render_rotation(self,
                        n_codes=15,
                        n_steps=5,
                        it=0,
                        fix_rot_range=None,
                        ele_val=None,
                        tmp=1.0):
        device = self.device
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        moment_dict = self.estimate_moments()
        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        radius = torch.ones(batch_size,
                            device=device) * moment_dict['radius_mean']

        ele = torch.ones(batch_size, device=device) * moment_dict['ele_mean']
        if ele_val is not None:
            ele[:] = ele_val
        prior_cam = {
            'fx': fovx,
            'fy': fovy,
            'radius': radius,
            'ele': ele,
        }
        if fix_rot_range is not None:
            rot_values = torch.linspace(fix_rot_range[0],
                                        fix_rot_range[1],
                                        n_steps,
                                        device=device)
        rot_values = torch.linspace(moment_dict['rot_minus_std2'],
                                    moment_dict['rot_plus_std2'],
                                    n_steps,
                                    device=device)
        out_dict = {}
        grid_images = []
        for idx_code in tqdm(range(n_codes)):
            latent_codes = self.sample_latent_codes(batch_size=batch_size,
                                                    it=it, tmp=tmp)
            rot_images = []
            for rot_val in tqdm(rot_values):
                rot_val = torch.ones(batch_size, device=device) * rot_val
                prior_cam['rot'] = rot_val
                with torch.no_grad():
                    out = self.forward(latent_codes,
                                       prior_cam=prior_cam,
                                       fix_cam=True,
                                       sample_patch=False,
                                       it=it,
                                       batch_size=batch_size)
                rot_images.append(out)
            grid_images.append(rot_images)
        grid_images = self.restructure_grid_list(grid_images)
        return grid_images

    def render_single_object(self, n_codes=40, n_imgs=10, it=np.inf):
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        grid_images = []
        grid_depths = []
        for idx_code in tqdm(range(n_codes)):
            latent_codes = self.sample_latent_codes(batch_size=batch_size,
                                                    it=it)
            rot_images = []
            depth_images = []
            for rot_val in tqdm(range(n_imgs)):
                with torch.no_grad():
                    out = self.forward(latent_codes,
                                       sample_patch=False,
                                       it=it,
                                       batch_size=batch_size)
                    depth = out['depth_fg'][:, 0]
                    depth = color_depth_map_tensor(depth)
                    depth_images.append(depth)
                    out = out['rgb']
                rot_images.append(out)
            rot_images = make_grid(torch.cat(rot_images),
                                   nrow=n_imgs,
                                   pad_value=GRID_PAD_VALUE)
            depth_images = make_grid(torch.cat(depth_images),
                                     nrow=n_imgs,
                                     pad_value=GRID_PAD_VALUE)
            grid_images.append(rot_images)
            grid_depths.append(depth_images)
        return grid_images, grid_depths

    def render_teaser(self, n_codes=15, it=0):
        device = self.device
        self.test_img_size = 300
        old_size = self.test_img_size
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        moment_dict = self.estimate_moments()
        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        radius = torch.ones(batch_size,
                            device=device) * moment_dict['radius_max']
        ele_mean = torch.ones(batch_size,
                              device=device) * moment_dict['ele_mean']
        rot_mean = torch.ones(batch_size,
                              device=device) * moment_dict['rot_mean']
        rot_ext = torch.ones(batch_size,
                             device=device) * moment_dict['rot_plus_std2']
        ele_ext = torch.ones(batch_size,
                             device=device) * moment_dict['ele_minus_std2']

        prior_cam = {
            'fx': fovx,
            'fy': fovy,
            'radius': radius,
        }

        keys = ['rgb', 'rgb_fg', 'rgb_bg']
        out_dict = {}
        for k in keys:
            out_dict[k] = []
        for idx_code in tqdm(range(n_codes)):
            latent_codes = self.sample_latent_codes(batch_size=batch_size,
                                                    it=it)
            prior_cam_i = prior_cam.copy()
            prior_cam_i.update({
                'ele': ele_mean,
                'rot': rot_mean,
            })
            with torch.no_grad():
                out0 = self.forward(latent_codes,
                                    prior_cam=prior_cam_i,
                                    fix_cam=True,
                                    sample_patch=False,
                                    it=it,
                                    batch_size=batch_size,
                                    composite_fg_on_white=True)
            for k in keys:
                out_dict[k].append(out0[k].cpu())

            # render second img; top left
            prior_cam_i = prior_cam.copy()
            prior_cam_i.update({
                'ele': ele_ext,
                'rot': rot_ext,
            })
            with torch.no_grad():
                out1 = self.forward(latent_codes,
                                    prior_cam=prior_cam_i,
                                    fix_cam=True,
                                    sample_patch=False,
                                    it=it,
                                    batch_size=batch_size,
                                    composite_fg_on_white=True)
            for k in keys:
                out_dict[k].append(out1[k].cpu())
        self.test_img_size = old_size
        return out_dict

    def render_elevation(self, n_codes=15, n_steps=5, it=0, tmp=1.0):
        device = self.device
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        moment_dict = self.estimate_moments()
        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        radius = torch.ones(batch_size,
                            device=device) * moment_dict['radius_mean']
        rot = torch.ones(batch_size, device=device) * moment_dict['rot_mean']
        prior_cam = {
            'fx': fovx,
            'fy': fovy,
            'radius': radius,
            'rot': rot,
        }

        ele_values = torch.linspace(moment_dict['ele_minus_std2'],
                                    moment_dict['ele_plus_std2'],
                                    n_steps,
                                    device=device)

        out_dict = {}
        grid_images = []
        for idx_code in tqdm(range(n_codes)):
            latent_codes = self.sample_latent_codes(batch_size=batch_size,
                                                    it=it, tmp=tmp)
            rot_images = []
            for rot_val in tqdm(ele_values):
                rot_val = torch.ones(batch_size, device=device) * rot_val
                prior_cam['ele'] = rot_val
                with torch.no_grad():
                    out = self.forward(latent_codes,
                                       prior_cam=prior_cam,
                                       fix_cam=True,
                                       sample_patch=False,
                                       it=it,
                                       batch_size=batch_size)
                rot_images.append(out)
            grid_images.append(rot_images)
        grid_images = self.restructure_grid_list(grid_images)
        return grid_images

    def render_zoom(self, n_codes=10, n_steps=7, it=0, tmp=0.6):
        device = self.device
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        moment_dict = self.estimate_moments()
        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        rot = torch.ones(batch_size, device=device) * moment_dict['rot_mean']
        ele = torch.ones(batch_size, device=device) * moment_dict['ele_mean']

        prior_cam = {
            'fx': fovx,
            'fy': fovy,
            'rot': rot,
            'ele': ele,
        }
        rad_values = torch.linspace(0.3, 1.5, n_steps, device=device)

        out_dict = {}
        grid_images = []
        for idx_code in tqdm(range(n_codes)):
            latent_codes = self.sample_latent_codes(batch_size=batch_size,
                                                    it=it, tmp=tmp)
            rot_images = []
            for rot_val in tqdm(rad_values):
                rot_val = torch.ones(batch_size, device=device) * rot_val
                prior_cam['radius'] = rot_val
                with torch.no_grad():
                    out = self.forward(latent_codes,
                                       prior_cam=prior_cam,
                                       fix_cam=True,
                                       sample_patch=False,
                                       it=it,
                                       batch_size=batch_size)
                rot_images.append(out)
            grid_images.append(rot_images)
        grid_images = self.restructure_grid_list(grid_images)
        return grid_images

    def render_focal_zoom(self, n_codes=10, n_steps=7, it=0, focal="both", tmp=1.0):
        device = self.device
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        moment_dict = self.estimate_moments()
        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        radius = torch.ones(batch_size,
                            device=device) * moment_dict['radius_mean']
        rot = torch.ones(batch_size, device=device) * moment_dict['rot_mean']
        ele = torch.ones(batch_size, device=device) * moment_dict['ele_mean']

        prior_cam = {
            'fx': fovx,
            'fy': fovy,
            'radius': radius,
            'rot': rot,
            'ele': ele,
        }
        fov_values = torch.linspace(moment_dict['fx_min'],
                                    moment_dict['fx_max'],
                                    n_steps,
                                    device=device)

        out_dict = {}
        grid_images = []
        for idx_code in tqdm(range(n_codes)):
            latent_codes = self.sample_latent_codes(batch_size=batch_size,
                                                    it=it, tmp=tmp)
            rot_images = []
            for rot_val in tqdm(fov_values):
                rot_val = torch.ones(batch_size, device=device) * rot_val
                #prior_cam['radius'] = rot_val
                if focal == "both":
                    prior_cam['fx'] = rot_val
                    prior_cam['fy'] = rot_val
                elif focal == "x":
                    prior_cam['fx'] = rot_val
                elif focal == "y":
                    prior_cam['fy'] = rot_val
                with torch.no_grad():
                    out = self.forward(latent_codes,
                                       prior_cam=prior_cam,
                                       fix_cam=True,
                                       sample_patch=False,
                                       it=it,
                                       batch_size=batch_size)
                rot_images.append(out)
            grid_images.append(rot_images)
        grid_images = self.restructure_grid_list(grid_images)
        return grid_images

    def render_interpolate(self, n_codes=18, n_steps=5, it=0, shape=True, tmp=1.0):
        device = self.device
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        moment_dict = self.estimate_moments()
        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        radius = torch.ones(batch_size,
                            device=device) * moment_dict['radius_mean']
        rot = torch.ones(batch_size, device=device) * moment_dict['rot_mean']
        ele = torch.ones(batch_size, device=device) * moment_dict['ele_mean']

        prior_cam = {
            'fx': fovx,
            'fy': fovy,
            'radius': radius,
            'rot': rot,
            'ele': ele,
        }

        out_dict = {}
        grid_images = []
        for idx_code in tqdm(range(n_codes)):
            rot_images = []
            latent_codes0 = self.sample_latent_codes(batch_size=batch_size,
                                                     it=it, tmp=tmp)
            latent_codes1 = self.sample_latent_codes(batch_size=batch_size,
                                                     it=it, tmp=tmp)
            for rot_val in tqdm(range(n_steps)):
                weight = rot_val * 1.0 / (n_steps - 1)
                z_fg_i = interpolate_sphere(latent_codes0['z_fg'],
                                            latent_codes1['z_fg'], weight)
                if not shape:
                    z_fg_i[..., :self.c_dim //
                           2] = latent_codes0['z_fg'][..., :self.c_dim // 2]
                else:
                    z_fg_i[..., self.c_dim //
                           2:] = latent_codes0['z_fg'][..., self.c_dim // 2:]
                z_bg = latent_codes0['z_bg']
                latent_codes = {
                    'z_fg': z_fg_i,
                    'z_bg': z_bg,
                }
                with torch.no_grad():
                    out = self.forward(latent_codes,
                                       prior_cam=prior_cam,
                                       fix_cam=True,
                                       sample_patch=False,
                                       it=it,
                                       batch_size=batch_size)
                rot_images.append(out)
            grid_images.append(rot_images)
        grid_images = self.restructure_grid_list(grid_images)
        return grid_images

    def render_disentanglement(self, n_codes=15, it=0, tmp=1.0):
        device = self.device
        batch_size = 1

        torch.manual_seed(0)
        np.random.seed(0)

        moment_dict = self.estimate_moments()
        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        radius = torch.ones(batch_size,
                            device=device) * moment_dict['radius_mean']
        rot = torch.ones(batch_size, device=device) * moment_dict['rot_mean']
        ele = torch.ones(batch_size, device=device) * moment_dict['ele_mean']
        prior_cam = {
            'fx': fovx,
            'fy': fovy,
            'radius': radius,
            'rot': rot,
            'ele': ele,
        }

        out_dict = {}
        grid_images = []
        for idx_code in tqdm(range(n_codes)):
            latent_codes = self.sample_latent_codes(batch_size=batch_size,
                                                    it=it, tmp=tmp)
            with torch.no_grad():
                out = self.forward(latent_codes,
                                   prior_cam=prior_cam,
                                   fix_cam=True,
                                   sample_patch=False,
                                   it=it,
                                   batch_size=batch_size)
            rot_images = torch.cat(
                [
                    out['rgb'],
                    out['acc_fg'],
                    #out['rgb_fg'],
                    #out['acc_bg'],
                    out['rgb_bg']
                ],
                0)
            rot_images = make_grid(rot_images, pad_value=GRID_PAD_VALUE)
            grid_images.append(rot_images)
        out_dict = {'fused': grid_images}
        return out_dict

    def render_spiral(self,
                      latent_codes=None,
                      batch_size=1,
                      n_steps=64,
                      to_numpy=True,
                      it=0,
                      rand_seed=0,
                      zoom_range=[0.3, 1.5],
                      with_zoom=False,
                      fix_rot=False,
                      fix_ele=False,
                      return_latent_codes=False,
                      ele_sin=False,
                      increase_size=False,
                      return_depth=False, 
                      sample_tmp=1.0):
        device = self.device
        if latent_codes is None:
            latent_codes = self.sample_latent_codes(batch_size, it=it, tmp=sample_tmp)

        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
        if increase_size:
            img_size = self.test_img_size
            self.test_img_size = 512

        # use mean for now
        moment_dict = self.estimate_moments()

        fovx = torch.ones(batch_size, device=device) * moment_dict['fx_mean']
        fovy = torch.ones(batch_size, device=device) * moment_dict['fy_mean']
        radius = torch.ones(batch_size,
                            device=device) * moment_dict['radius_mean']

        rot_values = moment_dict['rot_mean'] + torch.sin(
            torch.linspace(0., np.pi * 2, n_steps, device=device)) * (
                moment_dict['rot_plus_std2'] - moment_dict['rot_mean'])
        if ele_sin:
            ele_values = moment_dict['ele_mean'] + (torch.sin(
                torch.linspace(0., np.pi * 2, n_steps, device=device))) * (
                    moment_dict['ele_plus_std2'] - moment_dict['ele_mean'])
        else:
            ele_values = moment_dict['ele_mean'] + (torch.cos(
                torch.linspace(0., np.pi * 2, n_steps, device=device))) * (
                    moment_dict['ele_plus_std2'] - moment_dict['ele_mean'])
        rad_values = moment_dict['radius_mean'] + torch.sin(
            torch.linspace(0., np.pi * 2, n_steps, device=device)) * (
                moment_dict['radius_plus_std2'] - moment_dict['radius_mean'])
        prior_cam = {'fx': fovx, 'fy': fovy, 'radius': radius}

        if fix_rot:
            rot_values[:] = moment_dict['rot_mean']
        if fix_ele:
            ele_values[:] = moment_dict['ele_mean']

        out_dicts = []
        for (rot_val, ele_val,
             rad_val) in tqdm(zip(rot_values, ele_values, rad_values)):
            rot_i = torch.ones(batch_size, device=device) * rot_val
            ele_i = torch.ones(batch_size, device=device) * ele_val
            prior_cam['rot'] = rot_i
            prior_cam['ele'] = ele_i
            if with_zoom:
                prior_cam['radius'] = torch.ones(batch_size,
                                                 device=device) * rad_val
            out_dict_i = self.forward(latent_codes,
                                      prior_cam=prior_cam,
                                      fix_cam=True,
                                      sample_patch=False,
                                      it=it,
                                      batch_size=batch_size)
            torch.cuda.empty_cache()
            out_dicts.append(out_dict_i)

        rgb = torch.cat([d['rgb'] for d in out_dicts]).cpu()
        if to_numpy:
            rgb = rgb.permute(0, 2, 3, 1)
            rgb = (rgb.numpy() * 255).astype(np.uint8)

        if return_depth:
            depth = torch.cat([d['depth_fg'] for d in out_dicts]).cpu()
            depth = color_depth_map_tensor(depth[:, 0])
            depth = depth.permute(0, 2, 3, 1).numpy()
            depth = (depth * 255).astype(np.uint8)

        if increase_size:
            self.test_img_size = img_size

        if return_depth:
            return rgb, depth
        if return_latent_codes:
            return rgb, latent_codes
        return rgb

    def forward(self,
                latent_codes=None,
                prior_cam=None,
                sampling_pattern=None,
                batch_size=None,
                fix_cam=False,
                sample_patch=True,
                it=np.inf,
                composite_fg_on_white=False):
        it = self.convert_it(it)
        if latent_codes is None:
            latent_codes = self.sample_latent_codes(batch_size, it=it)

        if prior_cam is None:
            prior_cam = self.sample_prior_cam(batch_size, it=it)

        if fix_cam:
            post_cam = prior_cam
        else:
            post_cam = self.get_post_cam(prior_cam, it=it)

        camera_dict = self.restructure_cam(post_cam)

        if sampling_pattern is None:
            v = self.get_patch_sampling_pattern(batch_size, it)

        out = self.volume_render(latent_codes,
                                 camera_dict,
                                 v,
                                 it,
                                 composite_fg_on_white=composite_fg_on_white)
        return out

    def get_rays(self, camera_dict, grid):
        device = self.device
        fx = camera_dict['fx']
        fy = camera_dict['fy']
        c2w = camera_dict['c2w']
        img_size = camera_dict['img_size']
        rays_o, rays_d = get_camera_rays(img_size,
                                         img_size,
                                         fx,
                                         c2w,
                                         device=device,
                                         focal_y=fy,
                                         sampling_pattern=grid)
        return rays_o, rays_d

    def run_network_evaluation(self,
                               network,
                               pts,
                               latent_code,
                               rays_d,
                               depth,
                               far,
                               add_noise=True,
                               return_last_transmittance=False,
                               is_background=False,
                               raw_noise_std=1.):
        batch_size = pts.shape[0]

        if add_noise:
            noise_std = torch.ones(batch_size, 1, 1, 1, 1,
                                   device=self.device) * raw_noise_std
        else:
            noise_std = torch.zeros(batch_size, 1, 1, 1, 1, device=self.device)

        raw = network(pts, latent_code, rays_d, raw_noise_std=noise_std)
        sigma, rgb = raw[..., 0], raw[..., 1:]

        # Nerf-like vol rendering
        if is_background:
            dists = depth[..., :-1] - depth[..., 1:]
            dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])],
                              -1)
            alpha = 1. - torch.exp(-sigma * dists)
            T = torch.cumprod(1. - alpha + 1e-6, dim=-1)[..., :-1]
            T = torch.cat([torch.ones_like(T[..., :1]), T], -1)
        else:
            dists = depth[..., 1:] - depth[..., :-1]
            dists = torch.cat([dists, far - depth[..., -1:]], -1)
            dists = dists * torch.norm(
                rays_d.squeeze(-2), dim=-1, keepdim=True)
            alpha = 1. - torch.exp(-sigma * dists)
            T = torch.cumprod(1. - alpha + 1e-6, dim=-1)
            bg_lambda = T[..., -1]
            T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], -1)

        weights = alpha * T
        rgb = torch.sum(weights.unsqueeze(-1) * rgb, -2)
        acc = torch.sum(weights, -1)
        depth = torch.sum(weights * depth, -1)
        disp = 1. / torch.max(1e-10 * torch.ones_like(depth),
                              depth / torch.sum(weights, -1))
        disp[torch.isnan(disp)] = 1e-10

        if return_last_transmittance:
            return rgb, acc, depth, disp, bg_lambda
        return rgb, acc, depth, disp

    def volume_render(self,
                      latent_codes,
                      camera_dict,
                      sampling_pattern,
                      it=np.inf,
                      composite_fg_on_white=False):
        # Get camera rays of size [B, H, W, 3]
        rays_o, rays_d = self.get_rays(camera_dict, sampling_pattern['grid'])
        # Helper
        batch_size = rays_o.shape[0]
        patch_size = rays_d.shape[1]

        rays_d = rays_d.unsqueeze(-2)
        rays_o = rays_o.unsqueeze(-2)

        near = camera_dict['near'].reshape(batch_size, 1, 1, 1)
        far = camera_dict['far'].reshape(batch_size, 1, 1, 1)
        n_samples = self.get_sample_number(it)

        # get depths for FG
        device = self.device
        fg_depth = near + torch.linspace(0., 1., n_samples, device=device
                                         ).reshape(1, 1, 1, -1) * (far - near)
        # Expand it to B x P x P x n_samples
        fg_depth = fg_depth.repeat(1, patch_size, patch_size, 1)
        if not self.is_test_mode:
            fg_depth = perturb_samples(fg_depth)

        # Eval network
        pts_fg = rays_o + fg_depth.unsqueeze(-1) * rays_d

        z_fg = latent_codes['z_fg'].reshape(batch_size, 1, 1, 1, -1)

        fg_raw_noise = self.raw_noise_std
        rgb_fg, acc_fg, depth_fg, disp_fg, lambda_bg = self.run_network_evaluation(
            self.decoder,
            pts_fg,
            z_fg,
            rays_d,
            fg_depth,
            far,
            add_noise=((it < 10000) and (not self.is_test_mode)),
            return_last_transmittance=True,
            raw_noise_std=fg_raw_noise)

        if self.decoder_bg is not None:
            # Get BG depth
            n_ray_samples_bg = n_samples // 4
            z_bg = latent_codes['z_bg'].reshape(batch_size, 1, 1, 1, -1)

            bg_depth = torch.linspace(0., 1., n_ray_samples_bg,
                                      device=device).reshape(1, 1, 1, -1)
            # Expand to B x P x P x n_samples_bg
            bg_depth = bg_depth.repeat(batch_size, patch_size, patch_size, 1)
            bg_depth = perturb_samples(bg_depth)
            # now BG
            pts_bg, depth_real_bg = depth2pts_outside(rays_o.expand_as(rays_d),
                                                      rays_d, bg_depth)
            # flip because we need to resort the inverse depth from nearest to farest
            pts_bg = pts_bg.flip(-2)
            bg_depth = bg_depth.flip(-1)
            depth_real_bg = depth_real_bg.flip(-1)
            z_bg = latent_codes['z_bg'].reshape(batch_size, 1, 1, 1, -1)
            rgb_bg, acc_bg, depth_bg, disp_bg = self.run_network_evaluation(
                self.decoder_bg,
                pts_bg,
                z_bg,
                rays_d,
                bg_depth,
                far,
                add_noise=((it < 10000) and self.add_noise_bg),
                return_last_transmittance=False,
                is_background=True,
                raw_noise_std=fg_raw_noise)
        else:
            n_ray_samples_bg = 0
            rgb_bg = torch.zeros_like(rgb_fg)
            acc_bg = torch.zeros_like(acc_fg)
            depth_bg = torch.zeros_like(depth_fg)

        # rewrite to channel format
        rgb_fg = rgb_fg.permute(0, 3, 1, 2)
        rgb_bg = rgb_bg.permute(0, 3, 1, 2)
        acc_fg = acc_fg.unsqueeze(1).repeat(1, 3, 1, 1)
        acc_bg = acc_bg.unsqueeze(1).repeat(1, 3, 1, 1)
        depth_bg = depth_bg.unsqueeze(1).repeat(1, 3, 1, 1)
        depth_fg = depth_fg.unsqueeze(1).repeat(1, 3, 1, 1)

        if disp_fg.shape[0] == 1:
            disp_fg = disp_fg / 2. + 0.5
            depth_fg = 1 / torch.max(1e-10 * torch.ones_like(disp_fg), disp_fg)
            depth_fg[disp_fg == 1e10] = far
            depth_fg = (depth_fg - near) / (far - near)
            depth_fg = depth_fg.repeat(1, 3, 1, 1)

        if self.decoder_bg is not None:
            rgb_final = rgb_fg + rgb_bg * lambda_bg.unsqueeze(1)
        else:
            rgb_final = rgb_fg

        depth_fg[acc_fg < 0.2] = 0.

        if self.white_background:
            rgb_final = rgb_final + (1 - acc_fg) * 1.

        if composite_fg_on_white:
            rgb_fg = rgb_fg + (1 - acc_fg) * 1.

        return {
            # Output renderings
            'rgb': rgb_final,
            'rgb_fg': rgb_fg,
            'rgb_bg': rgb_bg,
            'acc_fg': acc_fg,
            'acc_bg': acc_bg,
            'depth_fg': depth_fg,
            'depth_bg': depth_bg,
            # Additional info
            'n_samples': n_samples,
            'n_samples_bg': n_ray_samples_bg,
            'resolution': patch_size,
            'batch_size': batch_size,
            'pscale_min': sampling_pattern['min_scale'],
            'pscale_max': sampling_pattern['max_scale'],
            'pgrid': sampling_pattern['grid'],
            # add more if needed
        }

    def get_sample_number(self, it=np.inf):
        if self.is_test_mode:
            return self.n_test_samples

        idx = 0
        for ms in self.sample_milestones:
            if it > ms:
                idx += 1
        return self.n_samples_ps[idx]

    def get_camera_histogram(self, to_pytorch=True):
        device = self.device
        prior_cam = self.sample_prior_cam(10000)
        post_cam = self.get_post_cam(prior_cam)

        fx = post_cam['fx'].cpu().numpy()
        fy = post_cam['fy'].cpu().numpy()
        rot = post_cam['rot'].cpu().numpy()
        ele = post_cam['ele'].cpu().numpy()
        radius = post_cam['radius'].cpu().numpy()

        fig, axes = plt.subplots(2, 3, tight_layout=True)
        axes[0, 0].hist(rot, bins=256, density=True)
        axes[0, 0].get_yaxis().set_visible(False)
        axes[0, 0].set_title("Rotation Angle")
        axes[0, 1].hist(ele, bins=256, density=True)
        axes[0, 1].get_yaxis().set_visible(False)
        axes[0, 1].set_title("Elevation Angle")
        axes[1, 0].hist(radius, bins=256, density=True)
        axes[1, 0].get_yaxis().set_visible(False)
        axes[1, 0].set_title("Radius")
        axes[1, 1].hist(fx, bins=256, density=True)
        axes[1, 1].get_yaxis().set_visible(False)
        axes[1, 1].set_title("Field of View X (FoV)")
        axes[1, 2].hist(fy, bins=256, density=True)
        axes[1, 2].get_yaxis().set_visible(False)
        axes[1, 2].set_title("Field of View Y (FOV)")
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close()
        if to_pytorch:
            data = torch.from_numpy(data.astype(np.float32) / 255.).permute(
                2, 0, 1)
        return data

    def rescale_prior_cam(self, prior_cam):
        if self.camera_param == 'normal':
            prior_cam.clamp_(-1., 1.)
            prior_cam[
                ...,
                0] = self.fov_range[0] + prior_cam[..., 0] * self.fov_range[1]
            prior_cam[
                ...,
                1] = self.fov_range[0] + prior_cam[..., 1] * self.fov_range[1]
            prior_cam[
                ...,
                2] = self.rot_range[0] + prior_cam[..., 2] * self.rot_range[1]
            prior_cam[
                ...,
                3] = self.ele_range[0] + prior_cam[..., 3] * self.ele_range[1]
            prior_cam[..., 4] = self.radius_range[0] + prior_cam[
                ..., 4] * self.radius_range[1]
        elif self.camera_param == 'full':
            prior_cam[..., :4] = prior_cam[..., :4].clamp(-1., 1.)
            prior_cam[
                ...,
                0] = self.fov_range[0] + prior_cam[..., 0] * self.fov_range[1]
            prior_cam[
                ...,
                1] = self.fov_range[0] + prior_cam[..., 1] * self.fov_range[1]
            prior_cam[
                ...,
                2] = self.ele_range[0] + prior_cam[..., 2] * self.ele_range[1]
            prior_cam[..., 3] = self.radius_range[0] + prior_cam[
                ..., 3] * self.radius_range[1]
            rot_mat = project_to_so(prior_cam[..., 4:].reshape(-1, 2, 2))
            a1 = rot_mat[:, -2, 0]
            a2 = torch.norm(rot_mat[:, -2, 1:3], dim=-1)
            rot = torch.rad2deg(torch.atan2(a1, a2) * 2)  # * 180 / np.pi
            return prior_cam, rot

        return prior_cam

    def render_histogram(self, out_file="./out/hist.png", mode="rotation"):
        import matplotlib
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        device = self.device
        prior_cam = self.sample_prior_cam(100000)
        post_cam = self.get_post_cam(prior_cam)

        if self.camera_param == 'normal':
            prior_scaled = self.rescale_prior_cam(prior_cam)
            prior_scaled = prior_scaled.detach().cpu().numpy()
            rot_prior = prior_scaled[..., 2]
            radius_prior = prior_scaled[..., 4]
            ele_prior = prior_scaled[..., 3]
        else:
            prior_scaled, rot_prior = self.rescale_prior_cam(prior_cam)
            prior_scaled = prior_scaled.detach().cpu().numpy()
            rot_prior = rot_prior.detach().cpu().numpy()
            radius_prior = prior_scaled[..., 4]
            ele_prior = prior_scaled[..., 2]

        fx = post_cam['fx'].cpu().numpy()
        fy = post_cam['fy'].cpu().numpy()
        rot = post_cam['rot'].cpu().numpy()
        ele = post_cam['ele'].cpu().numpy()
        radius = post_cam['radius'].cpu().numpy()

        plt.figure(figsize=(10, 3))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=16)
        plt.yticks([])
        if self.gt_stats_file is not None:
            stats_gt = np.load(self.gt_stats_file)
            gt_rot = stats_gt['rot']
            gt_ele = stats_gt['ele']
        else:
            gt_rot = None
            gt_ele = None

        if mode == "rotation":
            plt.xlabel("Rotation Angle", fontsize=18)
            plt.hist(rot_prior,
                     color="#3db8ff",
                     alpha=0.5,
                     bins=360,
                     density=True,
                     label="Prior")
            plt.hist(rot,
                     color="#3F5D7D",
                     alpha=0.8,
                     bins=360,
                     density=True,
                     label="Posterior")
            if gt_rot is not None:
                plt.hist(gt_rot, alpha=0.8, bins=360, density=True, label="GT")
        if mode == "elevation":
            plt.hist(-ele_prior,
                     color="#3db8ff",
                     alpha=0.5,
                     bins=360,
                     density=True,
                     label="Prior")
            plt.xlabel("Elevation Angle", fontsize=18)
            plt.hist(-ele,
                     color="#3F5D7D",
                     alpha=0.8,
                     bins=360,
                     density=True,
                     label="Posterior")
            if gt_ele is not None:
                plt.hist(gt_ele, alpha=0.8, bins=360, density=True, label="GT")
        if mode == "radius":
            plt.hist(radius_prior,
                     color="#3db8ff",
                     alpha=0.5,
                     bins=360,
                     density=True,
                     label="Prior")
            plt.xlabel("Distance", fontsize=18)
            plt.hist(radius,
                     color="#3F5D7D",
                     alpha=0.8,
                     bins=360,
                     density=True,
                     label="Posterior")

        plt.legend(loc='upper right', fontsize=14)
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()

    def sample_latent_codes(self, batch_size=None, it=0, tmp=1.0):
        device = self.device
        c_dim = self.c_dim

        if batch_size is None:
            batch_size = self.get_batch_size(it)

        z_fg = torch.randn(batch_size, c_dim, device=device) * tmp
        z_bg = torch.randn(batch_size, c_dim, device=device) * tmp

        latent_codes = {
            'z_fg': z_fg,
            'z_bg': z_bg,
        }
        return latent_codes

    def sample_rotmat_ele_prior(self,
                                angle_range=[-90, 90],
                                batch_size=1,
                                reshape_to_mat=False):

        # angle_range = [i/180*np.pi for i in angle_range]
        prior = angle_range[0] + torch.rand(batch_size) * (angle_range[1] -
                                                           angle_range[0])
        prior = torch.deg2rad(prior)
        prior = torch.stack([
            torch.cos(prior),
            -torch.sin(prior),
            torch.sin(prior),
            torch.cos(prior),
        ], -1)
        if reshape_to_mat:
            prior = prior.reshape(-1, 2, 2)
        return prior

    def sample_prior_cam(self, batch_size=None, it=0):

        if batch_size is None:
            batch_size = self.get_batch_size(it)

        if self.camera_param == 'normal':
            if self.priorcamtype == 'gauss':
                prior_cam = torch.randn(
                    batch_size, 5, device=self.device) * self.camera_normal_std
            elif self.priorcamtype == 'uniform':
                prior_cam = (torch.rand(batch_size, 5, device=self.device) * 2
                             - 1) * self.camera_normal_std
        elif self.camera_param == 'full':
            prior_cam = torch.randn(
                batch_size, 8, device=self.device) * self.camera_normal_std
            prior_cam[..., 2] = (torch.rand_like(prior_cam[..., 2]) * 2. -
                                 1.) * 0.95
        return prior_cam

    def get_post_cam(self, prior_cam, it=np.inf):
        if (it < self.init_pose_iter) or (self.latent_cameras is None):
            post_cam = prior_cam.clone()
        else:
            post_cam = self.latent_cameras(prior_cam)
            if self.residual_pose:
                post_cam = post_cam + prior_cam

            if self.not_learn_instrinsics:
                part0 = post_cam[..., :2].detach()
                part1 = post_cam[..., 2:]
                post_cam = torch.cat([part0, part1], dim=-1)
            if self.fix_uniform_instrinsics:
                post_cam[..., :2] = (
                    np.random.rand(*post_cam[..., :2].shape) * 2. - 1.) * 0.5
            if self.fix_gauss_instrinsics:
                post_cam[..., :2] = (
                    np.random.randn(*post_cam[..., :2].shape) * 2. - 1.) * 0.5

        out_dict = {}
        if self.camera_param == 'full':
            p0 = post_cam[..., :4].clamp(-1, 1)
            post_cam = torch.cat([p0, post_cam[..., 4:]], -1)
            fovx, fovy, ele, radius = post_cam[..., 0], post_cam[
                ..., 1], post_cam[..., 2], post_cam[..., 3]
            rot_mat = post_cam[..., 4:].reshape(-1, 2, 2)
            rot_mat = get_rot_theta_from_2x2(rot_mat)
            out_dict['rot_mat'] = rot_mat
            a1 = rot_mat[:, -2, 0]
            a2 = torch.norm(rot_mat[:, -2, 1:3], dim=-1)
            out_dict['rot'] = torch.rad2deg(torch.atan2(a1, a2) * 2)
        else:
            post_cam = post_cam.clamp(-1, 1)
            fovx, fovy, rot, ele, radius = post_cam[..., 0], post_cam[
                ..., 1], post_cam[..., 2], post_cam[..., 3], post_cam[..., 4]
            rot = self.rot_range[0] + rot * self.rot_range[1]
            out_dict['rot'] = rot

        fovx = self.fov_range[0] + fovx * self.fov_range[1]
        fovy = self.fov_range[0] + fovy * self.fov_range[1]
        if self.fov_range[1] == 0:
            fovx = fovx.detach()
            fovy = fovy.detach()
        out_dict['fy'] = fovy
        out_dict['fx'] = fovx
        radius = self.radius_range[0] + radius * self.radius_range[1]
        if self.radius_range[1] == 0:
            radius = radius.detach()
        out_dict['radius'] = radius
        if self.camera_param not in ['carla', 'carla2']:
            out_dict['ele'] = self.ele_range[0] + ele * self.ele_range[1]

        return out_dict

    def calc_near_far_planes(self, camera_dict):
        radius = camera_dict['radius'].detach()
        obj_near, obj_far = self.object_planes
        near_plane = obj_near + radius
        far_plane = obj_far + radius
        return near_plane, far_plane

    def restructure_cam(self, cam):
        out_dict = {}
        out_dict['fx'] = get_focal_from_fov(self.img_size, cam['fx'])
        out_dict['fy'] = get_focal_from_fov(self.img_size, cam['fy'])
        out_dict['rot'] = cam['rot']
        out_dict['ele'] = cam['ele']
        out_dict['radius'] = cam['radius']
        if 'rot_mat' in cam.keys():
            out_dict['rot_mat'] = cam['rot_mat']
        if 'ele_mat' in cam.keys():
            out_dict['ele_mat'] = cam['ele_mat']
        out_dict['c2w'] = self.get_c2w_matrix(out_dict)
        out_dict['img_size'] = self.img_size
        np, fp = self.calc_near_far_planes(out_dict)
        out_dict['near'] = np
        out_dict['far'] = fp

        return out_dict

    def get_c2w_matrix(self, camera_dict):
        radius = camera_dict['radius']
        if ('rot_mat' in camera_dict.keys()) and ('ele_mat'
                                                  in camera_dict.keys()):
            rot_mat = camera_dict['rot_mat']
            ele_mat = camera_dict['ele_mat']
            c2w = pose_spherical_ele_rot(rot_mat, ele_mat, radius)
        elif 'rot_mat' in camera_dict.keys():
            ele = camera_dict['ele']
            rot_mat = camera_dict['rot_mat']
            c2w = pose_spherical_ele(rot_mat, ele, radius)
        else:
            rot = camera_dict['rot']
            ele = camera_dict['ele']
            c2w = pose_spherical_b(rot, ele, radius)
        return c2w

    def get_pg_size(self, it):
        ms = self.pg_milestones
        res = self.pg_resolution0
        for ms_i in ms:
            if it > ms_i:
                res *= 2
        return res

    def get_patch_sampling_pattern(self, batch_size=None, it=np.inf):
        if self.is_test_mode:
            batch_size = 1
            res = self.test_img_size
            grid = create_meshgrid(res,
                                   res,
                                   normalized_coordinates=True,
                                   device=self.device)
            grid = grid.repeat(batch_size, 1, 1, 1)
            return {
                'grid': grid,
                'min_scale': torch.ones(1, ),
                'max_scale': torch.ones(1, ),
            }

        if batch_size is None:
            batch_size = self.get_batch_size(it)
        output_res = self.get_pg_size(it)
        res = self.img_size
        device = self.device

        grid = create_meshgrid(output_res,
                               output_res,
                               normalized_coordinates=True,
                               device=device)
        grid = grid.repeat(batch_size, 1, 1, 1)
        p_min_scale, p_max_scale = res * 1.0 / output_res, res * 1.0 / output_res
        return {
            'grid': grid,
            'min_scale': p_min_scale,
            'max_scale': p_max_scale
        }

    def convert_it(self, it):
        data_type = type(it)
        assert (data_type in [float, int, torch.Tensor])
        if data_type == torch.Tensor:
            it = it.reshape(-1)[0].item()
        return it

    def get_batch_size(self, it):
        idx = 0
        for ms_i in self.pg_milestones:
            if it > ms_i:
                idx += 1
        batch_size = self.batch_size_pg[idx]
        return batch_size
