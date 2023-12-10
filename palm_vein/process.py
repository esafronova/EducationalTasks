import cv2
import numpy as np

from os import listdir as ld
from os.path import join as jp

from scipy.ndimage.filters import gaussian_filter
from skimage import restoration
from skimage import io
from skimage.transform import resize
from skimage import exposure


def adaptive_hist_eq(img):
    res = exposure.equalize_adapthist(img)
    return res


def denoise(img):
    img = np.array(img * 255, dtype=np.uint8)
    res = np.copy(img)
    cv2.fastNlMeansDenoising(img, res, 80, 15, 5)
    return res


def gabor(img):
    if img.max() <= 1:
        img = np.array(img * 255, dtype=np.uint8)
    res = np.zeros(img.shape)
    for sigma in (0.25, 0.5, 1):
        s = sigma
        for theta in range(6):
            th = theta / 6.0 * np.pi
            kernel = cv2.getGaborKernel((int(s * 3), int(s * 3)), s, th, s * 3, 1, 0)
            tmp = cv2.filter2D(255 - img, cv2.CV_32F, kernel)
            res[res < tmp] = tmp[res < tmp]
    res = (res - res.min()) / (res.max() - res.min()) * 255
    res = np.array(255 - res, dtype=np.uint8)
    return res


def process_path(path_input, path_output):
    file_names = ld(path_input)
    num_img_process = len(file_names)
    for i in range(num_img_process):
        file = file_names[i]
        image = io.imread(jp(path_input, file))
        image = adaptive_hist_eq(image)
        image = denoise(image)
        image = adaptive_hist_eq(image)
        image = (image - image.mean()) / (image.max() - image.min()) + 0.5
        new_name = file[:-4] + '.png'
        io.imsave(jp(path_output, new_name), image)

def process_path_2(path_input, path_output):
    file_names = ld(path_input)
    num_img_process = len(file_names)
    for i in range(num_img_process):
        file = file_names[i]
        image = io.imread(jp(path_input, file))
        image = denoise(image)
        image = 255 - image
        image = adaptive_hist_eq(image)
        image = gabor(image)
        new_name = file[:-4] + 'nlm_clahe_gabor.png'
        io.imsave(jp(path_output, new_name), image)


def equalise_hist(path_input, path_output):
    file_names = ld(path_input)
    num_img_process = len(file_names)
    for i in range(num_img_process):
        file = file_names[i]
        image = io.imread(jp(path_input, file))
        image = image[:, :, 0]
        image = cv2.equalizeHist(image)
        new_name = file[:-4] + '_equalized.png'
        io.imsave(jp(path_output, new_name), image)
