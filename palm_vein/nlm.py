import numpy as np
import cv2
from os import listdir as ld
from os.path import join as jp
from scipy.misc import imresize
from skimage import restoration
from skimage import io
from skimage import transform


def remove_nonuniform_illuminance(image, r):
    res = np.copy(image)
    res = np.array(res, dtype=float)
    illuminance = np.empty((res.shape[0] // r, res.shape[1] // r))
    for j in range(res.shape[1] // r):
        for i in range(res.shape[0] // r):
            y_from = j * r
            y_to = y_from + r
            x_from = i * r
            x_to = x_from + r
            illuminance[i, j] = image[x_from : x_to, y_from : y_to].mean()
    illuminance = imresize(illuminance, image.shape)
    res = res - illuminance
    min, max = res.min(), image.max()
    res = np.array((res - min) / (max - min) * 255, dtype=np.uint8)
    io.imshow(res)
    return res


def denoise(path_input,
            path_output,
            block_size=7,
            search_window=21,
            h=80,
            num_img_process=None):
    """
    :param block_size: Size of patches used for denoising
    :param search_window: Maximal distance in pixels where to search patches
                          used for denoising
    """
    file_names = ld(path_input)

    if num_img_process == None or num_img_process > len(file_names):
        num_img_process = len(file_names)

    for i in range(num_img_process):
        file = file_names[i]
        image = cv2.imread(jp(path_input, file))

        cv2.fastNlMeansDenoising(image, image, h, search_window, block_size)
        new_name = file[:-4] + parametres.to_string() + '.png'
        io.imsave(jp(path_output, new_name), image)
