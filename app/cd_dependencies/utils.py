import numpy as np
import torch
from torchvision import utils


def get_loaders(args):
    """
    Get dataloaders for training and validation
    """
    from datasets.CD_dataset import CDDataset
    from datasets.data_config import DataConfig

    data_config = DataConfig().get_data_config(args.data_name)

    # --- START: MODIFICATION FOR CUSTOM DATA PATH ---
    root_dir = (
        args.custom_data_root
        if hasattr(args, "custom_data_root") and args.custom_data_root
        else data_config.root_dir
    )
    data_name = args.data_name
    # --- END: MODIFICATION FOR CUSTOM DATA PATH ---

    # Create training dataset
    train_dataset = CDDataset(
        root_dir=root_dir,
        data_name=data_name,  # Pass data_name
        img_size=args.img_size,
        split=args.split,
        is_train=True,
        label_transform=data_config.label_transform,
        t1_source=args.t1_source,
        t2_source=args.t2_source,
    )

    # Create validation dataset
    val_dataset = CDDataset(
        root_dir=root_dir,
        data_name=data_name,  # Pass data_name
        img_size=args.img_size,
        split=args.split_val,
        is_train=False,
        label_transform=data_config.label_transform,
        t1_source=args.t1_source,
        t2_source=args.t2_source,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader}


def get_loader(
    data_name,
    img_size,
    batch_size,
    is_train=True,
    split="test",
    t1_source="A_SAR",
    t2_source="B_SAR",
    custom_data_root=None,
):
    """
    Get single dataloader (e.g., for testing)
    """
    from datasets.CD_dataset import CDDataset
    from datasets.data_config import DataConfig

    data_config = DataConfig().get_data_config(data_name)

    root_dir = custom_data_root if custom_data_root else data_config.root_dir

    # --- START OF CORRECTION ---
    # The call to CDDataset is simplified to match the new definition in CD_dataset.py.
    # The extra, unused arguments have been removed.
    dataset = CDDataset(
        root_dir=root_dir,
        data_name=data_name,
        img_size=img_size,
        split=split,
        is_train=is_train,
        label_transform=data_config.label_transform,
        t1_source=t1_source,
        t2_source=t2_source,
    )
    # --- END OF CORRECTION ---

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5
