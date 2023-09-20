import numpy as np


def seam_carve(img, mode, mask=None):
    [orientation, operation] = mode.split(' ')

    if orientation == 'vertical':
        img = np.transpose(img, axes=(1, 0, 2))
        if mask is not None:
            mask = np.transpose(mask)

    scale = 256 * img.shape[0] * img.shape[1]
    grayscale = to_grayscale(img)
    g = gradient(grayscale)
    if mask is not None:
        mask = mask * scale
        g += mask

    img, mask, seam_mask = carve(img, mask, g, operation)

    if orientation == 'vertical':
        img = np.transpose(img, axes=(1, 0, 2))
        seam_mask = np.transpose(seam_mask)
        if mask is not None:
            mask = np.transpose(mask)

    if mask is not None:
        mask = mask / scale

    return img, mask, seam_mask


def to_grayscale(img):
    res = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return res


def gradient(img):
    gx = np.zeros(img.shape)
    gx[1: -1, :] = img[2:, :] - img[: -2, :]
    gx[0, :] = img[1, :] - img[0, :]
    gx[-1, :] = img[-1, :] - img[-2, :]

    gy = np.zeros(img.shape)
    gy[:, 1:-1] = img[:, 2:] - img[:, : -2]
    gy[:, 0] = img[:, 1] - img[:, 0]
    gy[:, -1] = img[:, -1] - img[:, -2]

    gradient = np.sqrt(gx ** 2 + gy ** 2)
    return gradient


def carve(img, mask, gr, operation):
    for i in range(1, gr.shape[0]):
        for j in range(gr.shape[1]):
            if j == 0:
                gr[i,j] += min(gr[i - 1, j], gr[i - 1, j + 1])
            elif j == gr.shape[1] - 1:
                gr[i,j] += min(gr[i - 1, j - 1], gr[i - 1, j])
            else:
                gr[i,j] += min(gr[i - 1, j - 1], gr[i - 1, j + 1], gr[i - 1, j])

    seam = np.empty(gr.shape[0], dtype=int)
    seam_mask = np.zeros_like(gr)
    new_mask = None

    if operation == 'shrink':
        result = np.empty((img.shape[0], img.shape[1] - 1, img.shape[2]), dtype=img.dtype)
        if mask is not None:
            new_mask = np.empty((mask.shape[0], mask.shape[1] - 1), dtype=img.dtype)
    else:
        result = np.empty((img.shape[0], img.shape[1] + 1, img.shape[2]), dtype=img.dtype)
        if mask is not None:
            new_mask = np.empty((mask.shape[0], mask.shape[1] + 1), dtype=img.dtype)

    for i in range(seam.size - 1, -1, -1):
        if i == seam.size - 1:
            seam[i] = (gr[-1, :]).argmin()
        else:
            j = seam[i + 1]
            seam[i] = (gr[i, max(j - 1, 0): min(j + 2, gr.shape[1])]).argmin() + max(j - 1, 0)

        j = seam[i]
        seam_mask[i, j] = 1

        if operation == 'shrink':
            result[i] = np.concatenate([img[i, :j, :], img[i, j + 1:, :]])
            if mask is not None:
                new_mask[i] = np.concatenate([mask[i, :j], mask[i, j + 1:]])
        else:
            result[i] = np.concatenate([img[i, :j + 1, :],
                                        np.expand_dims((img[i, j, :] + img[i, min(j + 1, img.shape[1] - 1), :]) / 2,
                                                       axis=0),
                                        img[i, j + 1:, :]])

            if mask is not None:
                new_mask[i] = np.concatenate([mask[i, :j + 1],
                                              np.expand_dims((mask[i, j] + mask[i, min(j + 1, img.shape[1] - 1)]) / 2,
                                                             axis=0),
                                              mask[i, j + 1:]])

    return result, new_mask, seam_mask
