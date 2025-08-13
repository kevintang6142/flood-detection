"""
变化检测数据集
"""

import os

import numpy as np
from datasets.data_utils import CDDataAugmentation
from PIL import Image
from torch.utils import data

# --- CONSTANTS ---
LIST_FOLDER_NAME = "list"
ANNOT_FOLDER_NAME = "label"


def load_img_name_list(dataset_path):
    """Loads a list of image names from a .txt file."""
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def get_img_path(root_dir, img_name, folder_name):
    """Constructs the full path for an image."""
    return os.path.join(root_dir, folder_name, img_name)


def get_label_path(root_dir, img_name):
    """Constructs the full path for a label."""
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name)


class ImageDataset(data.Dataset):
    """A generic image dataset class."""

    def __init__(
        self,
        root_dir,
        data_name="default",
        split="train",
        img_size=256,
        is_train=True,
        t1_source="A",
        t2_source="B",
    ):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.t1_source = t1_source
        self.t2_source = t2_source
        self.is_train = is_train

        # --- START OF CORRECTION ---
        # If using a 'custom' dataset for prediction, we generate the image list by
        # scanning the folder. Otherwise, we load it from the .txt file as before.
        if data_name == "custom":
            # Scan the 'A' folder (t1_source) to get the list of image names
            img_dir = os.path.join(self.root_dir, self.t1_source)
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(
                    f"Image directory not found for custom dataset: {img_dir}"
                )
            # Ensure we only get image files, not other files that might be present
            self.img_name_list = sorted(
                [
                    f
                    for f in os.listdir(img_dir)
                    if f.endswith((".png", ".jpg", ".jpeg"))
                ]
            )
        else:
            # Original behavior: load from a predefined list file
            self.list_path = os.path.join(
                self.root_dir, LIST_FOLDER_NAME, self.split + ".txt"
            )
            self.img_name_list = load_img_name_list(self.list_path)
        # --- END OF CORRECTION ---

        self.A_size = len(self.img_name_list)

        # Setup data augmentation
        if self.is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(img_size=self.img_size)

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, name, self.t1_source)
        B_path = get_img_path(self.root_dir, name, self.t2_source)

        img_A = np.asarray(Image.open(A_path).convert("RGB"))
        img_B = np.asarray(Image.open(B_path).convert("RGB"))

        [img_A, img_B], _ = self.augm.transform([img_A, img_B], [], to_tensor=True)

        return {"A": img_A, "B": img_B, "name": name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):
    """Change Detection dataset class."""

    def __init__(
        self,
        root_dir,
        data_name="default",
        split="train",
        img_size=256,
        is_train=True,
        label_transform=None,
        t1_source="A",
        t2_source="B",
    ):
        super(CDDataset, self).__init__(
            root_dir,
            data_name=data_name,
            split=split,
            img_size=img_size,
            is_train=is_train,
            t1_source=t1_source,
            t2_source=t2_source,
        )
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, name, self.t1_source)
        B_path = get_img_path(self.root_dir, name, self.t2_source)
        L_path = get_label_path(self.root_dir, name)

        img_A = np.asarray(Image.open(A_path).convert("RGB"))
        img_B = np.asarray(Image.open(B_path).convert("RGB"))

        # For prediction, a dummy label is fine if the file doesn't exist.
        if os.path.exists(L_path):
            label = np.array(Image.open(L_path).convert("L"), dtype=np.uint8)
        else:
            # Create a dummy blank label if one isn't found
            label = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        if self.label_transform == "norm":
            label = label // 255

        [img_A, img_B], [label] = self.augm.transform(
            [img_A, img_B], [label], to_tensor=True
        )

        return {"name": name, "A": img_A, "B": img_B, "L": label}
