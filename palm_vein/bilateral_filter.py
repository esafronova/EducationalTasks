import os
from os import listdir as ld
from os.path import join as jp

import cv2
import numpy as np

from fire import Fire
from skimage import io
from typing import Optional


def bilateral(
        input_path: str,
        output_path: str,
        d: int = 7,
        sigma_color: float = 21,
        sigma_space: float = 21,
        num_img_process: Optional[int] = None,
):
    """Compute bilateral filtration of images from input_path and write them to
    output_path.
    :param input_path: path to input dir
    :param output_path: path to output dir
    :param d: Diameter of each pixel neighborhood that is used during filtering
    :param sigma_color: Filter sigma in the color space
    :param sigma_space: Filter sigma in the coordinate space
    :param num_img_process: number of images to process
    """
    os.makedirs(output_path, exist_ok=True)
    file_names = ld(input_path)

    if num_img_process is None or num_img_process > len(file_names):
        num_img_process = len(file_names)

    for i in range(num_img_process):
        file = file_names[i]
        image = io.imread(jp(input_path, file))
        image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        image = np.array(image, dtype=np.uint8)

        new_name = f'{file[:-4]}_d_{d}_sigmaC_{sigma_color}_sigmaS_{sigma_space}.png'
        io.imsave(jp(output_path, new_name), image)


if __name__ == '__main__':
    Fire(bilateral)
