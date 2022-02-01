from os.path import exists, join
from os import makedirs
from gan_training.metrics.inception import InceptionV3
from gan_training.metrics.utils import (
    calculate_frechet_distance, polynomial_mmd_averages)
from tqdm import tqdm
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import torch
from torchvision.utils import save_image, make_grid


class Evaluator(object):
    def __init__(self, dataset_name='carla', img_size=128, device="cuda",
                 dataloader=None, n_gt_images=20000, out_folder="data/stats_files",
                 save_vis_file=True):
        print("Initialize evaluator ...")
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.device = device
        self.save_vis_file = save_vis_file
        self.dataloader = dataloader
        if dataloader is not None:
            self.batch_size = next(iter(dataloader))['image'].shape[0]
            if (len(dataloader) * self.batch_size) < n_gt_images:
                print("Warning: Dataset only contains %d images." % (len(dataloader) * self.batch_size))
            self.n_gt_images = min(len(dataloader) * self.batch_size, n_gt_images)
        else:
            print("Warning. Dataloader is None. If you only want to load a stats file, it's fine.")
        self.stats_dict = {}

        self.dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx]).to(device)

        if not exists(out_folder):
            makedirs(out_folder)

        self.cache_file = join(out_folder, 'stats_%s%d.npz' % (dataset_name, img_size))
        self.vis_file = join(out_folder, 'vis_%s%d.jpg' % (dataset_name, img_size))

        if exists(self.cache_file):
            print("Using %s as cache file for evaluation." % self.cache_file)
            stats_dict = np.load(self.cache_file)
            self.stats_dict.update(stats_dict)
        else:
            assert(dataloader is not None)
            self.generate_cache_file()
        
        if self.save_vis_file and (not exists(self.vis_file)) and (dataloader is not None):
            self.save_visualization()

    def save_visualization(self):
        n_imgs = 100
        imgs = None
        for data in self.dataloader:
            if imgs is None:
                imgs = data['image']
            else:
                imgs = torch.cat([imgs, data['image']])
            if imgs.shape[0] >= n_imgs:
                break
        save_image(make_grid(imgs[:n_imgs], nrow=10), self.vis_file)
        print("Saved visualization of GT files to %s." % self.vis_file)


    def generate_cache_file(self):
        cache_file = self.cache_file
        print("Generate new cache evaluation file to %s." % cache_file)
        stats_dict = self.get_activations_gt()
        self.stats_dict.update(stats_dict)
        np.savez(cache_file, **stats_dict)
        print("Saved cache file to %s." % cache_file)

    def get_n_subset_kid(self, n_img, n_img_gt):
        min_img = min(n_img, n_img_gt)
        if min_img > 1000:
            return 1000, 100
        else:
            return int(min_img * 0.9), int(min_img * 0.5)

    def discretize_img(self, img):
        img = (img * 255).clamp(0., 255.).to(torch.uint8).float() / 255.
        return img

    def eval_fid_kid(self, img_fake, multiply_kid_by=100, batch_size=200):
        img_fake = self.discretize_img(img_fake)
        n_img = img_fake.shape[0]

        act, mu, sigma = self.get_activations_tensor(img_fake, batch_size=batch_size)
        act_gt, mu_gt, sigma_gt = self.stats_dict.values()
        subset_size, n_subsets = self.get_n_subset_kid(n_img, act_gt.shape[0])
        print(
            "Calculating FID with %d images and KID with %d images and %d subsets..." %
            (n_img, subset_size, n_subsets))
        fid = calculate_frechet_distance(mu_gt, sigma_gt, mu, sigma)        

        kids = polynomial_mmd_averages(
            act_gt, act, n_subsets=n_subsets, subset_size=subset_size)[0]
        kid_mean = kids.mean() * multiply_kid_by
        kid_std = kids.std() * multiply_kid_by
        return {
            'fid': fid, 'kid': kid_mean, 'kid_std': kid_std
        }

    def get_nearest_compatible_batch_size(self, n_img, batch_size):
        for batch_size_out in range(batch_size, 0, -1):
            if  n_img % batch_size_out == 0:
                return batch_size_out

    def get_activations_tensor(self, img_fake, batch_size=200):
        n_img = img_fake.shape[0]
        dims = self.dims
        model = self.model
        device = self.device
        model.eval()
        act = np.empty((n_img, dims))

        if (n_img % batch_size != 0):
            print("Warning: You chose a non-compatible batch size.")
            batch_size = self.get_nearest_compatible_batch_size(n_img, batch_size)
            print("Using %d instead." % batch_size)

        assert(n_img % batch_size == 0)
        start_idx = 0
        for idx in tqdm(range(0, n_img, batch_size)):
            batch = img_fake[idx:idx+batch_size].to(device)
            with torch.no_grad():
                pred = model(batch)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            act[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return act, mu, sigma


    def get_activations_gt(self):
        model = self.model
        n_gt_images = self.n_gt_images
        dims = self.dims
        dataloader = self.dataloader
        batch_size = self.batch_size
        device = self.device
        model.eval()

        act = np.empty((n_gt_images, dims))
        start_idx = 0
        pbar = tqdm(total=n_gt_images)
        for idx, batch in enumerate(dataloader):
            batch = batch['image'].to(device)

            with torch.no_grad():
                pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            act[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

            if start_idx >= n_gt_images:
                break
            pbar.update(batch_size)
        
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return {
            'act': act,
            'mu': mu,
            'sigma': sigma
        }
