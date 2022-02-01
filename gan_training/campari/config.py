from gan_training.campari import models, training
from gan_training.discriminator import discriminator_dict
from gan_training.decoder import decoder_dict
import torch
import numpy as np
from torch import nn


def get_model(cfg, mode='train'):
    discriminator = get_discriminator(cfg)
    generator = get_generator(cfg, mode=mode)
    if cfg['model']['use_moving_average']:
        generator_test = get_generator(cfg, is_test=True, mode=mode)
    else:
        generator_test = None
    model = models.Nerf(generator=generator, discriminator=discriminator,
                        generator_test=generator_test)
    return model


def get_trainer(model, evaluator, optimizer_g, optimizer_d, cfg):
    n_eval_images = cfg['training']['n_eval_images']
    training_kwargs = cfg['training']['training_kwargs']
    pg_milestones = cfg['training']['pg_milestones']
    pg_n_loop_final = cfg['training']['pg_n_loop_final']
    pg_batch_size = cfg['training']['batch_size_pg']

    trainer = training.Trainer(
        model, evaluator, optimizer_g, optimizer_d, 
        n_eval_images=n_eval_images, 
        pg_milestones=pg_milestones,
        pg_n_loop_final=pg_n_loop_final,
        pg_batch_size=pg_batch_size,  **training_kwargs)
    return trainer


def get_optimizer(model, cfg):
    optimizer_type = cfg['training']['optimizer']
    lr_g = cfg['training']['lr_g']
    lr_d = cfg['training']['lr_d']

    if optimizer_type == 'rmsprop':
        opt = torch.optim.RMSprop
    elif optimizer_type == 'adamw':
        opt = torch.optim.AdamW
    else:
        opt = torch.optim.Adam

    if model.discriminator is not None:
        optimizer_d = opt(model.discriminator.parameters(), lr=lr_d)
    else:
        optimizer_d = None

    param_list_g = [
        {'params': model.generator.decoder.parameters(),
        'lr': lr_g,
        },
    ]
    if hasattr(model.generator, "decoder_bg") and model.generator.decoder_bg is not None:
        param_list_g += [
            {'params': model.generator.decoder_bg.parameters(),
             'lr': lr_g,
            },
        ]
    if hasattr(model.generator, "latent_cameras") and model.generator.latent_cameras is not None:
        param_list_g += [
            {'params': model.generator.latent_cameras.parameters(),
            'lr': lr_g,
            },
        ]

    optimizer_g = opt(param_list_g)
    return optimizer_g, optimizer_d


def get_discriminator(cfg):
    discriminator = cfg['model']['discriminator']
    discriminator_kwargs = cfg['model']['discriminator_kwargs']
    pg_milestones = cfg['training']['pg_milestones']
    img_size = cfg['data']['img_size']
    if discriminator is not None:
        discriminator = discriminator_dict[discriminator](
            img_size=img_size, pg_milestones=pg_milestones,
            **discriminator_kwargs)
    return discriminator


def get_generator(cfg, is_test=False, mode='train'):
    decoder, decoder_bg = get_decoders(cfg)
    latent_cameras = get_latent_cameras(cfg)
 
    img_size = cfg['data']['img_size']
    generator = cfg['model']['generator']
    generator_kwargs = cfg['model']['generator_kwargs']
    c_dim = cfg['model']['c_dim']
    batch_size_pg = cfg['training']['batch_size_pg']
    pg_milestones = cfg['training']['pg_milestones']

    if mode == 'train':
        n_test_samples = cfg['test']['n_test_samples_train']
        test_img_size = img_size
    elif mode == 'test':
        n_test_samples = cfg['test']['n_test_samples_test']
        test_img_size = img_size

    generator = models.generator_dict[generator](
        decoder=decoder, img_size=img_size,
        c_dim=c_dim, latent_cameras=latent_cameras,
        decoder_bg=decoder_bg,
        is_test_mode=is_test, test_img_size=test_img_size,
        n_test_samples=n_test_samples,
        pg_milestones=pg_milestones,
        batch_size_pg=batch_size_pg, **generator_kwargs)
    return generator

def get_latent_cameras(cfg):
    latent_cameras = cfg['model']['latent_cameras']
    if latent_cameras == 'embed4':
        latent_cameras = LatentCam()
    elif latent_cameras == 'embed_full4':
        latent_cameras = LatentCam(8)
    return latent_cameras


class LatentCam(nn.Module):
    def __init__(self, dim=5, hidden_dim=64, n_layers=4, lr_mult=1., init=True):
        super().__init__()
        self.fc_in = nn.Linear(dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, dim)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)
        ])
        self.actvn = nn.ReLU(inplace=True)
        self.lr_mult = lr_mult
        if init:
            self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.fc_out.weight, 0., 0.05)
        torch.nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        net = self.fc_in(x) * self.lr_mult
        for idx, l in enumerate(self.blocks):
            net = l(self.actvn(net)) * self.lr_mult
        out = self.fc_out(self.actvn(net)) * self.lr_mult
        return out


def get_decoders(cfg):
    decoder = cfg['model']['decoder']
    decoder_bg = cfg['model']['decoder_bg']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    decoder_bg_kwargs = cfg['model']['decoder_bg_kwargs']
    c_dim = cfg['model']['c_dim']
    decoder = decoder_dict[decoder](c_dim=c_dim, **decoder_kwargs)
    if decoder_bg is not None:
        decoder_bg = decoder_dict[decoder_bg](c_dim=c_dim, **decoder_bg_kwargs)
    return decoder, decoder_bg

