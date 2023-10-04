import os
from os import listdir as ld
from os.path import join as jp

import numpy as np
from fire import Fire
from skimage import filters
from skimage import io
from typing import Optional


def gabor_filter(
        input_path: str,
        output_path: str,
        theta: float = 0,
        sigma: float = 2,
        offset: float = np.pi,
        frequency: float = 0.1,
        num_img_process: Optional[int] = None,
):
    """Compute filtration with Gabor kernels of images from input_path and
    write them to output_path.
    :param input_path: path to input dir
    :param output_path: path to output dir
    :param num_img_process: number of images to process
    """
    os.makedirs(output_path, exist_ok=True)
    file_names = ld(input_path)

    if num_img_process is None or num_img_process > len(file_names):
        num_img_process = len(file_names)

    for i in range(num_img_process):
        file = file_names[i]
        image = io.imread(jp(input_path, file))
        image = filters.gabor(image,
                              frequency,
                              theta=theta,
                              sigma_x=sigma,
                              sigma_y=sigma,
                              offset=offset)[0]
        image = np.array(image, dtype=np.uint8)
        new_name = f'{file[:-4]}_theta_{theta / np.pi}Pi_sigma_{sigma}_fr_{frequency}.png'
        io.imsave(jp(output_path, new_name), image)


def draw_kernel(
        output_path: str,
        theta: float = 0,
        sigma: float = 2,
        offset: float = np.pi,
        frequency: float = 0.1,
):
    os.makedirs(output_path, exist_ok=True)
    kernel = np.real(filters.gabor_kernel(frequency,
                                          theta=theta,
                                          sigma_x=sigma,
                                          sigma_y=sigma,
                                          offset=offset))
    min_v, max_v = kernel.min(), kernel.max()
    io.imsave(jp(output_path, f'kernel_theta_{theta / np.pi}Pi_sigma_{sigma}_fr_{frequency}.png'),
              np.array((kernel - min_v)/(max_v - min_v) * 255, dtype=np.uint8))


def main(
        input_path: str,
        output_path: str,
        save_kernels: bool = False,
        kernels_path: Optional[str] = None,
):
    for it in range(6):
        theta = it / 6. * np.pi
        for sigma in (2, 3, 4):
            gabor_filter(input_path, output_path, theta=theta, sigma=sigma)
            if save_kernels:
                draw_kernel(kernels_path, theta=theta, sigma=sigma)


if __name__ == '__main__':
    Fire(main)
