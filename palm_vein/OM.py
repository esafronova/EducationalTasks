import cv2
import numpy as np
import cmath, math
import matplotlib.pyplot as plt
import seaborn as sns

from os import listdir as ld
from os.path import join as jp

from scipy.interpolate import interp1d
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp
from skimage.feature import register_translation
from skimage.feature import local_binary_pattern as lbp


def get_kernel_30(kernel_radius, path=None):
    names = []
    kernels = np.empty((12, 2 * kernel_radius + 1, 2 * kernel_radius + 1))

    sigmaX = kernel_radius / 3
    sigmaY = sigmaX / 2
    gx = cv2.getGaussianKernel(kernel_radius * 2 + 1, sigmaX).reshape((kernel_radius * 2 + 1, 1))
    gy = cv2.getGaussianKernel(kernel_radius * 2 + 1, sigmaY).reshape((1, kernel_radius * 2 + 1))
    g = np.dot(gx, gy)

    M = cv2.getRotationMatrix2D((kernel_radius, sigmaX), 30, 1.0)
    g1 = cv2.warpAffine(g, M, g.shape)
    M = cv2.getRotationMatrix2D((kernel_radius, sigmaX), -30, 1.0)
    g2 = cv2.warpAffine(g, M, g.shape)
    kernel = g1 + g2

    kernel[:-kernel_radius // 2, kernel_radius // 2:] -= g1[kernel_radius // 2:, :-kernel_radius // 2] / 1.5
    kernel[:-kernel_radius // 2, :-kernel_radius // 2] -= g2[kernel_radius // 2:, kernel_radius // 2:] / 1.5

    sigmaX = kernel_radius / 5
    sigmaY = sigmaX
    h, w = kernel_radius, kernel_radius
    gx = cv2.getGaussianKernel(h, sigmaX).reshape((h, 1))
    gy = cv2.getGaussianKernel(w, sigmaY).reshape((1, w))
    g = np.dot(gx, gy)
    g = g / g.max() * kernel.max()
    x, y = kernel_radius, kernel_radius // 2
    kernel[x: x + h, y: y + w] -= g

    kernel[kernel < 0] = kernel[kernel < 0] / (-kernel[kernel < 0].sum()) * kernel[kernel > 0].sum()

    for i in range(12):
        angle2 = i * 30
        if i == 0:
            kernels[0] = kernel
        else:
            M = cv2.getRotationMatrix2D((kernel_radius, kernel_radius), angle2, 1.0)
            kernels[i] = cv2.warpAffine(kernel, M, kernel.shape)
        names.append('_r{}_a{}_rot{}'.format(kernel_radius, 30, angle2))
        if path is not None:
            cv2.imwrite(jp(path, 'kernel_r{}_a{}_rot{}.png'.format(kernel_radius, 30, angle2)),
                        ((kernels[i] - kernels[i].min()) / (
                                kernels[i].max() - kernels[i].min()) * 255).astype(int))

    return kernels, names


def get_kernel_15(kernel_radius, path=None):
    names = []
    kernels = np.empty((12, 2 * kernel_radius + 1, 2 * kernel_radius + 1))

    sigmaX = kernel_radius / 2
    sigmaY = sigmaX / 4
    gx = cv2.getGaussianKernel(kernel_radius * 2 + 1, sigmaX).reshape((kernel_radius * 2 + 1, 1))
    gy = cv2.getGaussianKernel(kernel_radius * 2 + 1, sigmaY).reshape((1, kernel_radius * 2 + 1))
    g = np.dot(gx, gy)

    M = cv2.getRotationMatrix2D((kernel_radius, sigmaX), 15, 1.0)
    g1 = cv2.warpAffine(g, M, g.shape)
    M = cv2.getRotationMatrix2D((kernel_radius, sigmaX), -15, 1.0)
    g2 = cv2.warpAffine(g, M, g.shape)
    kernel = g1 + g2

    kernel[:-kernel_radius // 2, kernel_radius // 2:] -= g1[kernel_radius // 2:, :-kernel_radius // 2] / 1.5
    kernel[:-kernel_radius // 2, :-kernel_radius // 2] -= g2[kernel_radius // 2:, kernel_radius // 2:] / 1.5

    sigmaX = sigmaX / 3
    sigmaY = sigmaX / 3
    h, w = min(int(sigmaX * 6), kernel_radius * 2 + 1), int(sigmaY * 6)
    gx = cv2.getGaussianKernel(h, sigmaX).reshape((h, 1))
    gy = cv2.getGaussianKernel(w, sigmaY).reshape((1, w))
    g = np.dot(gx, gy)
    g = g / g.max() * kernel.max()
    x, y = kernel_radius, kernel_radius // 2
    kernel[x: min(x + h, 2 * kernel_radius + 1), y: min(y + w, 2 * kernel_radius + 1)] -= g

    kernel[kernel < 0] = kernel[kernel < 0] / (-kernel[kernel < 0].sum()) * kernel[kernel > 0].sum()

    for i in range(12):
        angle2 = i * 30
        if i == 0:
            kernels[0] = kernel
        else:
            M = cv2.getRotationMatrix2D((kernel_radius, kernel_radius), angle2, 1.0)
            kernels[i] = cv2.warpAffine(kernel, M, kernel.shape)
        names.append('_r{}_a{}_rot{}'.format(kernel_radius, 15, angle2))
        if path is not None:
            cv2.imwrite(jp(path, 'kernel_r{}_a{}_rot{}.png'.format(kernel_radius, 15, angle2)),
                        ((kernels[i] - kernels[i].min()) / (
                                kernels[i].max() - kernels[i].min()) * 255).astype(int))

    return kernels, names


def get_multi_lobe_kernel(kernel_radii, path=None):
    kernels = []
    names = []

    for k in range(kernel_radii.size):
        kernel_radius = kernel_radii[k]

        kernels.append(np.empty((12, 2 * kernel_radius + 1, 2 * kernel_radius + 1)))

        ke, n = get_kernel_15(kernel_radius, path)
        kernels[k][:12] = ke
        names = names + n

    return kernels, names


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


def get_roi(roi_path, mask_path, output_path=None):
    """
    Inverts roi images and applies masks.
    :param roi_path: path to preprocessed ROI directory
    :param mask_path: path to extracted by principal curvature algorythm vein
                      structure mask
    :return: tuple(result, roi file names, normilized masks)
    """
    roi_file_names = ld(roi_path)
    mask_file_names = ld(mask_path)

    img = cv2.imread(jp(roi_path, roi_file_names[0]))
    result = np.empty((len(roi_file_names), img.shape[0], img.shape[1]))
    masks = np.empty((len(roi_file_names), img.shape[0], img.shape[1]))

    for i in range(len(roi_file_names)):
        img = 255 - cv2.imread(jp(roi_path, roi_file_names[i]))
        mask = cv2.imread(jp(mask_path, mask_file_names[i]))
        img[mask == 0] = 0
        result[i] = img.mean(axis=2)
        roi_file_names[i] = roi_file_names[i][:-4]
        masks[i] = mask[..., 0] / 255

        if output_path is not None:
            cv2.imwrite(jp(output_path, names1[i] + '_masked.png'), result[i])

    return result, roi_file_names, masks


def match_points(points1, points2, img1, img2, num_steps=50):
    kp1 = np.where(points1 == 1)
    kp1 = np.hstack((kp1[0].reshape((-1, 1)), kp1[1].reshape((-1, 1))))
    kp2 = np.where(points2 == 1)
    kp2 = np.hstack((kp2[0].reshape((-1, 1)), kp2[1].reshape((-1, 1))))

    best_model = None
    max_inl = 0

    if kp1.shape[0] < kp2.shape[0]:
        for i in range(num_steps):
            ind = np.random.choice(np.arange(kp2.shape[0]), kp1.shape[0], replace=False)
            kp2_ = kp2[ind]
            model_robust, inl = ransac((kp1, kp2_), model_class=AffineTransform, min_samples=3, residual_threshold=2,
                                       max_trials=100)
            if inl.sum() > max_inl:
                max_inl = inl.sum()
                best_model = model_robust

    elif kp1.shape[0] > kp2.shape[0]:
        for i in range(num_steps):
            ind = np.random.choice(np.arange(kp1.shape[0]), kp2.shape[0], replace=False)
            kp1_ = kp1[ind]
            model_robust, inl = ransac((kp1_, kp2), model_class=AffineTransform, min_samples=3, residual_threshold=2,
                                       max_trials=100)
            if inl.shape[0] > max_inl:
                max_inl = inl.shape[0]
                best_model = model_robust

    else:
        best_model, _ = ransac((kp1, kp2), model_class=AffineTransform, min_samples=3, residual_threshold=2,
                               max_trials=100)

    img2_warped = warp(img2, best_model.inverse, output_shape=img2.shape)

    print()


def get_masks(path):
    names = ld(path)

    img = cv2.imread(jp(path, names[0]))
    result = np.empty((len(names), img.shape[0], img.shape[1]))

    for i in range(len(names)):
        img = cv2.imread(jp(path, names[i]))
        result[i] = img.mean(axis=2)

    return result, names


def max_of_convolutional(images, kernels, path=None):
    result = np.empty_like(images)
    conv_res = np.empty((kernels.shape[0], *images[0].shape))

    for i in range(result.shape[0]):
        img = images[i]
        for j in range(kernels.shape[0]):
            conv_res[j] = cv2.filter2D(img, -1, kernels[j])
        result[i] = conv_res.max(axis=0)
        if path is not None:
            cv2.imwrite(jp(path, 'img_max_conv{}.png'.format(i)), result[i])

    return result


def convs(images, kernels, image_names=None, kernel_names=None, path=None, take_max=False, masks=None):
    num = len(kernels) * kernels[0].shape[0]
    conv_res = np.empty((images.shape[0], num, images.shape[1], images.shape[2]))

    for i in range(images.shape[0]):
        img = images[i]
        for k in range(len(kernels)):
            for j in range(kernels[k].shape[0]):
                ind = k * kernels[k].shape[0] + j
                conv_res[i, ind] = cv2.filter2D(img, -1, kernels[k][j])
                if masks is not None:
                    conv_res[i, ind][masks[i] == 0] = 0

    if take_max:
        conv_res = conv_res.max(axis=1, keepdims=False)

    if path is not None:
        for i in range(conv_res.shape[0]):
            for j in range(conv_res.shape[1]):
                if not take_max:
                    name = jp(path, image_names[i] + kernel_names[j] + '.png')
                else:
                    name = jp(path, image_names[i] + '_convmax.png')
                result = ((conv_res[i, j] - conv_res[i ,j].min()) / (conv_res[i, j].max()
                                                                     - conv_res[i, j].min()) * 255).astype(int)
                cv2.imwrite(name, result)

    return conv_res


def take_local_max(images, r1=1, r2=3, t_percent=None, threshold=None, image_names=None, path=None, masks=None, num_points=40):
    result = np.zeros_like(images)

    for i in range(result.shape[0]):
        tmp = np.zeros_like(images[i])
        for x in range(r1, result.shape[1] - r1, 2 * r1):
            for y in range(r1, result.shape[2] - r1, 2 * r1):
                block = images[i, x: x + 2 * r1 + 1, y: y + 2 * r1 + 1]
                (xm, ym) = np.unravel_index(np.argmax(block, axis=None), block.shape)
                tmp[x + xm, y + ym] = block.max()

        flag = True
        while flag:
            flag = False
            for x in range(r2, result.shape[1] - r2):
                for y in range(r2, result.shape[2] - r2):
                    block = tmp[x: x + 2 * r2 + 1, y: y + 2 * r2 + 1]
                    if (block > 0).sum() > 1:
                        tmp[x: x + 2 * r2 + 1, y: y + 2 * r2 + 1] = 0
                        block = images[i, x: x + 2 * r2 + 1, y: y + 2 * r2 + 1]
                        (xm, ym) = np.unravel_index(np.argmax(block, axis=None), block.shape)
                        tmp[x + xm, y + ym] = block.max()
                        flag = True

        if masks is not None:
            tmp[masks[i] == 0] = 0

        if threshold is None:
            if t_percent is not None:
                threshold = np.percentile(tmp[tmp>0], t_percent)
            if num_points is not None:
                temp = np.sort(tmp.flatten())[::-1]
                threshold = temp[num_points]

        result[i][tmp > threshold] = 1
        # print(result[i].sum())

        if path is not None:
            cv2.imwrite(jp(path, image_names[i] + '_localmax_r{}'.format(r1) + '.png'), (result[i] * 255).astype(int))

    return result


def threshold_out(images, num_points=100, threshold=None, t_percent=None, image_names=None,
                  kernel_names=None, path=None):
    result = np.zeros_like(images)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if num_points is not None:
                tmp = images[i, j].flatten()
                tmp = np.sort(tmp)[::-1]
                threshold = tmp[num_points]
            if t_percent is not None:
                threshold = np.percentile(images[i, j], t_percent)
            result[i, j][images[i, j] > threshold] = 1
            if path is not None:
                if kernel_names is not None:
                    kernel_name = kernel_names[j]
                else:
                    kernel_name = ''
                cv2.imwrite(jp(path, image_names[i] + kernel_name + '.png'), result[i, j] * 255)

    return result


def write_points_on_img(images, points, image_names, path):
    for i in range(points.shape[0]):
        result = np.dstack((images[i], images[i], images[i]))
        result[points[i] > 0] = np.array((255, 0, 0))
        cv2.imwrite(jp(path, 'points' + image_names[i] + '.png'), result)


def hamming_distance(img1, img2, mask1=None, mask2=None, max_shift=5):
    mask = mask1 * mask2
    if len(img1.shape) == 3:
        res = np.ones(img1.shape[0])
    else:
        res = np.ones(1)
    if mask.sum() != 0:
        if len(img1.shape) == 3:
            mask = np.expand_dims(mask, axis=0)
        res = (img1 * img2 * mask).sum(axis=(-1, -2)) / mask.sum()

    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            img_rolled = np.roll(img2, i, axis=-2)
            img_rolled = np.roll(img_rolled, j, axis=-1)
            mask_rolled = np.roll(mask2, i, axis=-2)
            mask_rolled = np.roll(mask_rolled, j, axis=-1)
            tmp_img1 = img1.copy()
            tmp_mask1 = mask1.copy()
            if i > 0:
                img_rolled[..., : i, :] = 0
                tmp_img1[..., : i, :] = 0
                tmp_mask1[..., : i, :] = 0
            elif i < 0:
                img_rolled[..., i:, :] = 0
                tmp_img1[..., i:, :] = 0
                tmp_mask1[..., i:, :] = 0
            if j > 0:
                img_rolled[..., : j] = 0
                tmp_img1[..., : j] = 0
                tmp_mask1[..., : j] = 0
            elif j < 0:
                img_rolled[..., j:] = 0
                tmp_img1[..., j:] = 0
                tmp_mask1[..., j:] = 0
            mask = tmp_mask1 * mask_rolled
            if mask.sum() != 0:
                if len(img1.shape) == 3:
                    mask = np.expand_dims(mask, axis=0)
                tmp = (tmp_img1 * img_rolled * mask).sum(axis=(-1, -2)) / mask.sum()
                res[tmp < res] = tmp[tmp < res]

    return np.mean(res)


def RMS_error(img1, img2):
    if len(img1.shape) == 2:
        _, error, _ = register_translation(img1, img2)
    else:
        error = 0
        for i in range(img1.shape[0]):
            _, e, _ = register_translation(img1[i], img2[i])
            if not math.isnan(e):
                error += e
        error /= img1.shape[0]
    return error


def lbp_images(images, radius=3, method='uniform', n_angles=4, binary=False, type='lbp'):
    if type == 'lbp':
        n_points = radius * 8
        if binary:
            result = np.zeros((images.shape[0], n_points, images.shape[1], images.shape[2]))
        else:
            result = np.empty_like(images)
        for i in range(images.shape[0]):
            res = lbp(images[i], n_points, radius, method)
            if binary:
                for j in range(n_points):
                    result[i][j][res == j] = 1
            else:
                result[i] = res
    else:
        result = np.zeros((images.shape[0], n_angles, images.shape[1], images.shape[2]))
        for i in range(images.shape[0]):
            result[i] = local_line_binary_pattern(images[i], r=radius, angles=n_angles)
    return result


def blockwise_hamming(img1, img2, num_block=8, max_shift=5):
    block_size = img1.shape[0] // num_block
    dist = np.ones((num_block, num_block)) * block_size * block_size
    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            img_rolled = np.roll(img2, i, axis=0)
            img_rolled = np.roll(img_rolled, j, axis=1)
            tmp_img1 = img1.copy()
            if i > 0:
                img_rolled[: i] = 0
                tmp_img1[: i] = 0
            elif i < 0:
                img_rolled[i:] = 0
                tmp_img1[i:] = 0
            if j > 0:
                img_rolled[:, : j] = 0
                tmp_img1[:, : j] = 0
            elif j < 0:
                img_rolled[:, j:] = 0
                tmp_img1[:, j:] = 0
            for x in range(num_block):
                for y in range(num_block):
                    sl = np.s_[x * block_size: (x + 1) * block_size, y * block_size: (y + 1) * block_size]
                    d = (tmp_img1[sl] != img_rolled[sl]) & (img_rolled[sl] != -1) & (tmp_img1[sl] != -1)
                    d = d.sum() / ((img_rolled[sl] != -1) & (tmp_img1[sl] != -1)).sum()
                    if dist[x, y] > d:
                        dist[x, y] = d
    return np.mean(dist)


def blockwise_om_hamming(img1, img2, num_block=8, max_shift=5):
    block_size = img1.shape[1] // num_block
    dist = np.ones((img1.shape[0], num_block, num_block)) * block_size * block_size
    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            img_rolled = np.roll(img2, i, axis=1)
            img_rolled = np.roll(img_rolled, j, axis=2)
            tmp_img1 = img1.copy()
            if i > 0:
                img_rolled[:,  i] = -1
                tmp_img1[:,  i] = -1
            elif i < 0:
                img_rolled[:, i:] = -1
                tmp_img1[:, i:] = -1
            if j > 0:
                img_rolled[:, :, : j] = -1
                tmp_img1[:, :, : j] = -1
            elif j < 0:
                img_rolled[:, :, j:] = -1
                tmp_img1[:, :, j:] = -1
            for x in range(num_block):
                for y in range(num_block):
                    sl = np.s_[:, x * block_size: (x + 1) * block_size, y * block_size: (y + 1) * block_size]
                    d = (tmp_img1[sl] != img_rolled[sl]) & (img_rolled[sl] != -1) & (tmp_img1[sl] != -1)
                    d = d.sum(axis=(1, 2)) / ((img_rolled[sl] != -1) & (tmp_img1[sl] != -1)).sum() * img1.shape[0]
                    dist[:, x, y][dist[:, x, y] > d] = d[dist[:, x, y] > d]

    return np.mean(dist)


def compute_all_d(images, names, path=None, suf='', distance=hamming_distance, masks=None):
    intra = []
    inter = []

    img_descr = {}
    for i in range(images.shape[0]):
        for j in range(i + 1, images.shape[0]):
            if i not in img_descr:
                det = names[i].split('_')
                img_descr[i] = [det[0], det[1]]
            if j not in img_descr:
                det = names[j].split('_')
                img_descr[j] = [det[0], det[1]]

            if masks is None:
                h = distance(images[i], images[j])
            else:
                h = distance(images[i], images[j], masks[i], masks[j])

            if img_descr[i][0] == img_descr[j][0] and img_descr[i][1] == img_descr[j][1]:
                intra.append(h)
            else:
                inter.append(h)
            print('compute for ', i, '(', j, ') / ', images.shape[0])

    intra = np.array(intra)
    inter = np.array(inter)

    if path is not None:
        name = 'intra' + suf + '.npy'
        np.save(jp(path, name), intra)
        name = 'inter' + suf + '.npy'
        np.save(jp(path, name), inter)

    return intra, inter


def read_d(path, suf=''):
    intra = np.load(jp(path, 'intra' + suf + '.npy'))
    inter = np.load(jp(path, 'inter' + suf + '.npy'))
    return intra, inter


def draw_histogram_distance(intra, inter, file_name):
    plt.clf()
    # intra = intra[intra>=0.9]
    sns.distplot(intra, norm_hist=True, label='intra', kde=False)
    sns.distplot(inter, norm_hist=True, label='inter', kde=False)
    # plt.xlabel('NRMSE', fontsize=24)
    # plt.ylabel('Percentage', fontsize=24)
    # plt.xticks((0.6, 0.7, 0.8, 0.9), fontsize=20)
    plt.xlabel('NRMSE')
    plt.ylabel('Percentage')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(file_name)


def compute_far(inter, intra):
    inter_sum = inter.size
    values, counts = np.unique(inter, return_counts=True)
    thresholds = np.zeros(values.size + 2)
    thresholds[1] = min(intra.min(), inter.min()) - 0.1
    thresholds[2: -1] = (values[:-1] + values[1:]) / 2
    thresholds[-1] = max(intra.max(), inter.max()) + 0.1
    far = np.zeros(values.size + 2)
    far[2:] = np.cumsum(counts) / inter_sum
    return far, thresholds


def compute_frr(inter, intra):
    intra_sum = intra.size
    values, counts = np.unique(intra, return_counts=True)
    thresholds = np.zeros(values.size + 2)
    thresholds[1] = min(intra.min(), inter.min()) - 0.1
    thresholds[2: -1] = (values[:-1] + values[1:]) / 2
    thresholds[-1] = max(intra.max(), inter.max()) + 0.1
    frr = np.zeros(values.size + 1)
    frr[1:] = np.cumsum(counts[::-1]) / intra_sum
    frr = frr[::-1]
    frr_res = np.ones(values.size + 2)
    frr_res[1:] = frr
    frr = frr_res
    return frr, thresholds


def draw_far_frr(inter, intra, file_name):
    plt.clf()
    far, thresholds_far = compute_far(inter, intra)
    frr, thresholds_frr = compute_frr(inter, intra)

    plt.plot(thresholds_far, far, label='FAR')
    plt.plot(thresholds_frr, frr, label='FRR')

    thr = np.unique(np.array(list(thresholds_far) + list(thresholds_frr)))
    far_int = interp1d(thresholds_far, far)
    frr_int = interp1d(thresholds_frr, frr)
    far = far_int(thr)
    frr = frr_int(thr)
    ind = np.where(far > frr)[0][0]
    eer = (far[ind-1: ind+1].sum()+  frr[ind-1: ind+1].sum()) / 4
    plt.title('EER = {}'.format(round(eer, 5)))

    plt.xlabel('Threshold', fontsize=24)
    plt.ylabel('Error', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(file_name)


def plot_kernels(kernels, plt_name):
    max_s = 20 * 2 + 1
    plt.figure(figsize=(6, 8))

    n = 0
    for i in range(len(kernels)):
        for j in range(12, kernels[i].shape[0]):
            n += 1
            plt.subplot(8, 6, n)
            plt.xlim((0, max_s))
            plt.ylim((0, max_s))
            plt.imshow(kernels[i][j], cmap='gray')
            plt.axis('off')

    plt.savefig(plt_name)


if __name__ == '__main__':
    kernels, _ = get_multi_lobe_kernel(np.arange(14, 22, 2))
    path = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\OM\blockwise_hamming_distance\850'
    path1 = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\casia\exstract_roi\850\preprocess_roi'
    path2 = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\casia\principal_curvature_process\850'
    roi, r_names, masks = get_roi(path1, path2)
    conv = convs(roi, kernels, r_names, take_max=False, masks=masks)
    compute_all_d(conv, r_names, path, '_RMS_om_MLDF12_masked', RMS_error)
    intra, inter = read_d(path, '_RMS_om_MLDF12_masked')
    draw_far_frr(inter, intra, jp(path, 'far_frr_RMS_om_MLDF12_masked.png'))
    draw_histogram_distance(intra, inter, jp(path, 'hist_RMS_om_MLDF12_masked.png'))
