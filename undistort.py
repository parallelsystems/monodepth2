import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from railyard.util import get_file_names, read_file, save_file
from railyard.util.calibration import lookup_intrinsics_by_part

def get_intrinsics():
    cam_pn = "Sunex DSL612A-650-F2.0"
    intrinsics = lookup_intrinsics_by_part(cam_pn)
    w, h = intrinsics["image_width"], intrinsics["image_height"]

    K = intrinsics["K"].copy()
    K[0, :] /= w
    K[1, :] /= h
    K[0, 2] = 0.5
    K[1, 2] = 0.5

    intrinsics["K"] = K
    return intrinsics

def main():
    intrinsics = get_intrinsics()
    K = intrinsics["K"]
    K[0, :] *= 608
    K[1, :] *= 416
    D = intrinsics["D"]
    print(K)
    print(D)

    root_dir = "./data/datasets/triplets_daytime_distorted"
    output_dir = "./data/datasets/triplets_daytime_undistorted2"

    meta = read_file(os.path.join(root_dir, "train/meta.json"))
    save_file(meta, os.path.join(output_dir, "train/meta.json"))
    meta = read_file(os.path.join(root_dir, "val/meta.json"))
    save_file(meta, os.path.join(output_dir, "val/meta.json"))

    file_names = get_file_names(root_dir, ext=".jpg")
    sample_names = [os.path.splitext(fn)[0] for fn in file_names]
    for sample_name in tqdm(sample_names):
        image = read_file(os.path.join(root_dir, f"{sample_name}.jpg"))

        image = np.array(image)
        image = cv2.undistort(image, K, D)
        image = Image.fromarray(image)

        save_file(image, os.path.join(output_dir, f"{sample_name}.jpg"))




if __name__ == "__main__":
    main()

