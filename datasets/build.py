from megengine.data import transform as T
from megengine.data.dataloader import DataLoader
from megengine.data.sampler import SequentialSampler

from .bases import ImageDataset
from .market1501 import Market1501


def make_dataloader(cfg):
    train_transforms = T.Compose(
        [
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(prob=0.5),
            T.Pad(10),
            T.RandomCrop((256, 128)),
            T.Normalize(mean=[103.530, 123.675, 116.280], std=[57.375, 58.395, 57.120]),
            T.ToMode(),
        ]
    )

    val_transforms = T.Compose(
        [
            T.Resize((256, 128)),
            T.Normalize(mean=[103.530, 123.675, 116.280], std=[57.375, 58.395, 57.120]),
            T.ToMode(),
        ]
    )

    num_workers = cfg.num_workers

    dataset = Market1501(root=cfg.data_root)

    train_set = ImageDataset(dataset.train)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    train_loader = DataLoader(
        dataset=train_set,
        sampler=SequentialSampler(dataset.train, cfg.batch_size, drop_last=True),
        transform=train_transforms,
        num_workers=num_workers,
    )

    val_set = ImageDataset(dataset.query + dataset.gallery)
    query_gallery = dataset.query + dataset.gallery
    val_loader = DataLoader(
        dataset=val_set,
        sampler=SequentialSampler(query_gallery, cfg.test_batch_size, drop_last=False),
        transform=val_transforms,
        num_workers=num_workers,
    )

    return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num
