import cv2
import numpy as np
from os import listdir as ld
from os.path import join as jp
from OM import blockwise_hamming, compute_all_d, draw_far_frr, draw_histogram_distance


def read_img(path):
    names = ld(path)
    names.sort()
    names = names[:100]
    img = cv2.imread(jp(path, names[0]), cv2.IMREAD_GRAYSCALE)
    res = np.empty((len(names), *img.shape), dtype=img.dtype)
    for i, name in enumerate(names):
        res[i] = cv2.imread(jp(path, names[i]), cv2.IMREAD_GRAYSCALE)
    return res, names


def union(masks, path):
    names = ld(path)
    names.sort()
    for i in range(masks.shape[0]):
        img = cv2.imread(jp(path, names[0]), cv2.IMREAD_GRAYSCALE)
        res = img / 255 * masks[i] / 255
        masks[i] = (res * 255).astype(masks.dtype)
    return masks


def apply_median(imgs, ksize=5):
    for i in range(imgs.shape[0]):
        imgs[i] = cv2.medianBlur(imgs[i], ksize)
    return imgs


def apply_otsu(imgs):
    for i in range(imgs.shape[0]):
        th, _ = cv2.threshold(imgs[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_th = np.zeros_like(imgs[i])
        img_th[imgs[i] < th] = 0
        img_th[imgs[i] >= th] = 255
        imgs[i] = 255 - img_th
    return imgs


def write_imgs(imgs, names, path, suff=''):
    for i, name in enumerate(names):
        cv2.imwrite(jp(path, name[:-4] + suff + name[-4:]), imgs[i])
