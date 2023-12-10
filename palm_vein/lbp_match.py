import cv2
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from os import listdir as ld
from os.path import join as jp
from skimage.feature import local_binary_pattern as lbp

from OM import compute_all_d, get_roi, draw_far_frr, draw_histogram_distance, read_d, lbp_images


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def local_line_binary_pattern(img, r=6, angles=4):
    res = np.empty((angles, *img.shape))

    pow_2 = np.zeros(2 * r + 1)
    for i in range(r):
        pow_2[i] = 2 ** (r - 1 - i)
        pow_2[2 * r - i] = pow_2[i]

    for n_a in range(angles):
        phi = 90. / angles * n_a
        M = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), phi, 1.0)
        M[0, 2] = r
        M[1, 2] = r
        img_r = cv2.warpAffine(img, M, (img.shape[0] + 2 * r, img.shape[1] + 2 * r))
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                g = np.zeros(2 * r + 1)
                g[img_r[x: x + 2 * r + 1, y] > img_r[x + r, y + r]] = 1
                g = (g * pow_2).sum()
                v = np.zeros(2 * r + 1)
                v[img_r[x, y: y + 2 * r + 1] > img_r[x + r, y + r]] = 1
                v = (v * pow_2).sum()
                res[n_a, x, y] = np.sqrt(g ** 2 + v ** 2)

    return res


def llbp_distance(llbp1, llbp2):
    return np.mean(np.abs(llbp1 - llbp2))


def compute_lbp_hist(image, lbp_img=None, n_points=3 * 8, radius=3, method='uniform'):
    if image is not None:
        lbp_img = lbp(image, n_points, radius, method)
    n_bins = 26 # int(lbp_img.max() + 1)
    hist, _ = np.histogram(lbp_img, density=True, bins=n_bins, range=(0, n_bins))
    return hist


def lbp_kld_distance(img1, img2):
    h1 = compute_lbp_hist(None, img1)
    h2 = compute_lbp_hist(None, img2)
    return kullback_leibler_divergence(h1, h2)


def llbp_kld_distance(llbp1, llbp2):
    n_bins = 16
    kld = 0
    for i in range(llbp1.shape[0]):
        h1, _ = np.histogram(llbp1[i], density=True, bins=n_bins, range=(0, n_bins))
        h2, _ = np.histogram(llbp2[i], density=True, bins=n_bins, range=(0, n_bins))
        kld += kullback_leibler_divergence(h1, h2)
    return kld / llbp1.shape[0]


def lbp_hamming_distance(lbp_1, lbp_2):
    return (lbp_1 == lbp_2).sum() / lbp_1.size


def blockwise_distance(img1, img2, mask1, mask2, distance=lbp_kld_distance, num_block=4, max_shift=5):
    block_size = img1.shape[0] // num_block
    dist = np.ones((num_block, num_block)) * block_size * block_size
    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            img_rolled = np.roll(img2, i, axis=0)
            img_rolled = np.roll(img_rolled, j, axis=1)
            mask_rolled = np.roll(mask2, i, axis=-2)
            mask_rolled = np.roll(mask_rolled, j, axis=-1)
            tmp_img1 = img1.copy()
            tmp_mask1 = mask1.copy()
            if i > 0:
                img_rolled[: i] = 0
                tmp_img1[: i] = 0
                tmp_mask1[: i] = 0
            elif i < 0:
                img_rolled[i:] = 0
                tmp_img1[i:] = 0
                tmp_mask1[i:] = 0
            if j > 0:
                img_rolled[:, : j] = 0
                tmp_img1[:, : j] = 0
                tmp_mask1[:, : j] = 0
            elif j < 0:
                img_rolled[:, j:] = 0
                tmp_img1[:, j:] = 0
                tmp_mask1[:, j:] = 0
            for x in range(num_block - 1):
                for y in range(num_block - 1):
                    sl = np.s_[..., x * block_size: (x + 1) * block_size, y * block_size: (y + 1) * block_size]
                    mask =tmp_mask1[sl] * mask_rolled[sl]
                    if mask.sum() > 0:
                        d = distance(tmp_img1[sl] * mask, img_rolled[sl] * mask) / mask.sum()
                        if dist[x, y] > d:
                            dist[x, y] = d
    return np.mean(dist)


if __name__ == '__main__':
    out_path = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\OM\blockwise_hamming_distance\new_roi1'
    roi_path = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\casia\exstract_roi\850\preprocess_roi'
    mask_path = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\casia\principal_curvature_process\850'

    roi, names, masks = get_roi(roi_path, mask_path)
    lbps = lbp_images(roi, type='lbp', radius=3)
    compute_all_d(lbps, names, out_path, '_KLD_block4_lbp3_t40', blockwise_distance, masks=masks)
    intra, inter = read_d(out_path, '_KLD_block4_lbp3_t40')
    draw_far_frr(inter, intra, jp(out_path, 'far_frr_KLD_block4_lbp3_t40.png'))
    draw_histogram_distance(intra, inter, jp(out_path, 'hist_KLD_block4_lbp3_t40.png'))
