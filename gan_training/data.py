import torch
from gan_training.datasets import ImageDataset
from torchvision import transforms


def get_dataloader(config, mode='train'):
    # Dataloader
    dataset = config['data']['dataset']
    img_size = config['data']['img_size']
    ds_path = config['data']['ds_path']
    n_workers = config['training']['n_workers']
    batch_size_pg = config['training']['batch_size_pg']

    assert(dataset in [
        'celeba',
        'cats',
        'carla',
        'chairs1',
        'chairs2'
    ])

    # define transform
    if dataset in ['celeba']:
        t = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor()])
    else:
        t = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    ds = ImageDataset(ds_path, t, composite_on_white=dataset in ['chairs1', 'chairs2'])

    data_loader = [torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True,  num_workers=nw,
        collate_fn=collate_fn, drop_last=True) for (nw, bs) in zip(n_workers, batch_size_pg)]
    data_loader_fid_gt = torch.utils.data.DataLoader(
        ds, batch_size=50, shuffle=True,  num_workers=12,
        collate_fn=collate_fn)
    return data_loader, data_loader_fid_gt


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
