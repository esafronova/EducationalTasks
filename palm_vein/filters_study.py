import cv2
import numpy as np

from os import listdir as ld
from os.path import join as jp
from scipy.ndimage.filters import gaussian_filter
from skimage import restoration
from skimage import io
from skimage import transform
from skimage import exposure


def adaptive_hist_eq(path_input, path_output, num_img_process=None):
    file_names = ld(path_input)
    if num_img_process == None or num_img_process > len(file_names):
        num_img_process = len(file_names)
    for i in range(num_img_process):
        file = file_names[i]
        image = cv2.imread(jp(path_input, file), 0)
        clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(240, 240))
        res = clahe.apply(image)
        new_name = file[:-4] + 'eqad_cl_0_kernel_240.png'
        cv2.imwrite(jp(path_output, new_name), res)


def gabor(input_path, output_path, num=None):
    files = ld(input_path)
    if num == None:
        num = len(files)
    for i in range(num):
        image = cv2.imread(jp(input_path, files[i]))
        res = np.zeros(image.shape)
        for sigma in (0.25, 0.5, 1):
            s = sigma
            for theta in range(6):
                th = theta / 6.0 * np.pi
                kernel = cv2.getGaborKernel((int(s * 3), int(s * 3)), s, th, s * 3, 1, 0)
                tmp = cv2.filter2D(255 - image, cv2.CV_32F, kernel)
                res[res < tmp] = tmp[res < tmp]
        cv2.imwrite(jp(output_path, files[i][:-4] + 'gabor.png'.format(s)), 255 - res)


def draw_kernel(output_path):
    for sigma in range(4, 12, 2):
        s = sigma
        for theta in range(6):
            th = theta / 6.0 * np.pi
            kernel = cv2.getGaborKernel((int(s * 3), int(s * 3)), s, th, s * 3, 1, 0)
            kernel = (kernel + 1) / 2 * 255
            cv2.imwrite(jp(output_path, 'kernel_sigma_{}_theta_{}Pi.png'.format(sigma, th / np.pi)), kernel)


def otsu(input_path, output_path, num = None):
    files = ld(input_path)
    if num == None:
        num = len(files)
    for i in range(num):
        image = cv2.imread(jp(input_path, files[i]))
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        cv2.imwrite(jp(output_path, files[i][:-4] + '_otsu.png'), ret)


def gauss_sharp(input_path, output_path, sigma, alpha, num = None):
    files = ld(input_path)
    if num == None:
        num = len(files)
    for i in range(num):
        image = cv2.imread(jp(input_path, files[i]))
        blur = cv2.GaussianBlur(image, None, sigma, 0)
        tmp = np.array(image - blur, dtype = np.int8)
        tmp = alpha * np.array(tmp, dtype=float)
        tmp = np.array(image + tmp, dtype=np.uint8)
        cv2.imwrite(jp(output_path, files[i][:-4] + '_sigma_{}_alpha_{}.png'.format(sigma, alpha)), tmp)


def laplacian_sharp(input_path, output_path, num = None):
    files = ld(input_path)
    if (num == None):
        num = len(files)
    for i in range(num):
        image = cv2.imread(jp(input_path, files[i]))
        laplacian = cv2.Laplacian(image, cv2.CV_16S)
        cv2.imwrite(jp(output_path, files[i][:-4] + '_laplacian.png'), laplacian)
