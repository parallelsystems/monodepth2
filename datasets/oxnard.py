import os
import json
import random
import numpy as np
import PIL.Image as pil
import torch


from .mono_dataset import MonoDataset
from torchvision import transforms

class OxnardDataset(MonoDataset):
    """UP Oxnard PTC line
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.K = np.array([
            [1.44783388, 0.        , 0.4426172 , 0.],
            [0.        , 2.27537375, 0.49484771, 0.],
            [0.        , 0.        , 1.        , 0.],
            [0.        , 0.        , 0.        , 0.]
        ], dtype=np.float32)

        self.full_res_shape = (608, 416)
        
        split = "train" if self.is_train else "val"
        self.root_dir = os.path.join(self.data_path, split)

        with open(os.path.join(self.root_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(os.path.join(folder, f"{frame_index}{self.img_ext}"))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder = os.path.join(self.root_dir, f"{index}")
        side = None

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, i + 1, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def __len__(self):
        return len(self.meta)

    def check_depth(self):
        return False

    def get_image_path(self, folder, frame_index, side):
        pass