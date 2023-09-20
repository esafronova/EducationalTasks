import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from matplotlib import pyplot as plt
import math
import cv2

max_img_size = 300


def align(img, g_cord):
    (h, w) = img.shape
    h_each = h // 3
    img_b = img[: h_each, :]
    img_g = img[h_each: h_each * 2, :]
    img_r = img[h_each * 2: h_each * 3, :]

    h_5 = h_each // 20
    w_5 = w // 20
    img_r = img_r[h_5: -h_5, w_5: -w_5]
    img_g = img_g[h_5: -h_5, w_5: -w_5]
    img_b = img_b[h_5: -h_5, w_5: -w_5]

    size = [img_r.shape]
    (h, w) = img_r.shape
    max_shift = [8]
    while h > max_img_size or w > max_img_size:
        h //= 2
        w //= 2
        size.append((h, w))
        max_shift.append(1)
    size.reverse()

    shift_r = np.zeros(2, dtype=int)
    shift_b = np.zeros(2, dtype=int)
    for i in range(len(size)):
        (h, w) = size[i]

        shift_r *= 2
        shift_b *= 2

        r = resize(img_r, (h, w)).astype(np.float32)
        g = resize(img_g, (h, w)).astype(np.float32)
        b = resize(img_b, (h, w)).astype(np.float32)

        shift_r = find_shift(g, r, shift_r, max_shift[i])
        shift_b = find_shift(g, b, shift_b, max_shift[i])

    (h, w) = img_r.shape
    ind = np.s_[max(0, shift_r[0], shift_b[0]): min(h, h + shift_r[0], h + shift_b[0]),
          max(0, shift_r[1], shift_b[1]): min(w, w + shift_r[1], w + shift_b[1])]
    r = np.roll(img_r, shift_r[0], axis=0)
    r = np.roll(r, shift_r[1], axis=1)[ind]
    b = np.roll(img_b, shift_b[0], axis=0)
    b = np.roll(b, shift_b[1], axis=1)[ind]
    g = img_g.copy()[ind]
    result = np.dstack((r, g, b))

    r_cord = (g_cord[0] + h_each - shift_r[0], g_cord[1] - shift_r[1])
    b_cord = (g_cord[0] - h_each - shift_b[0], g_cord[1] - shift_b[1])

    return result, b_cord, r_cord


def cross_corelation(img1, img2):
    res = (img1 * img2).sum()
    res /= np.sqrt((img1 ** 2).sum() * (img2 ** 2).sum())
    return res


def find_shift(img1, img2, shift, max_shift):
    m_max = -100
    img2 = np.roll(img2, shift[0], axis=0)
    img2 = np.roll(img2, shift[1], axis=1)
    new_shift = shift.copy()

    (h, w) = img1.shape

    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            img_rolled = np.roll(img2, i, axis=0)
            img_rolled = np.roll(img_rolled, j, axis=1)
            m = cross_corelation(img1[max(0, i): min(h, h + i), max(0, j): min(w, w + j)],
                                 img_rolled[max(0, i): min(h, h + i), max(0, j): min(w, w + j)])
            if m > m_max:
                m_max = m
                new_shift = shift + np.array((i, j), dtype=int)

    return new_shift