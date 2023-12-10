import cv2
import numpy as np
import random
from os.path import join as jp


Angles = np.linspace(-90, 90, 11)

def simple_kernel(path=None):
    kernels = np.ones((12, 5, 5))
    kernels[:, :, 2] = 0

    kernels[0, 0, 0] = 0
    kernels[0, 1, 1] = 0

    kernels[1, 2, :2] = 0

    kernels[2, 4, 0] = 0
    kernels[2, 3, 1] = 0

    for i in range(3, 12):
        kernels[i] = np.rot90(kernels[i - 3])

    if path is not None:
        for i in range(kernels.shape[0]):
            cv2.imwrite(jp(path, 'kernel_{}.png'.format(i)), (kernels[i] * 255).astype(int))

    sum = kernels.sum(axis=(1, 2))
    for i in range(kernels.shape[0]):
        kernels[i, :, :] /= sum[i]

    return kernels


def random_lobe(kernel_radius):
    sigma_max = kernel_radius / 3
    sigma_min = kernel_radius / 6
    sigmaX = random.random() * (sigma_max - sigma_min) + sigma_min
    sigmaY = sigmaX / 3
    g_radius = int(sigmaX * 3)
    g_size = 2 * g_radius + 1
    gx = cv2.getGaussianKernel(g_size, sigmaX).reshape((g_size, 1))
    gy = cv2.getGaussianKernel(g_size, sigmaY).reshape((1, g_size))
    g = np.dot(gx, gy)
    angle = Angles[random.randint(0, Angles.size - 1)]
    M = cv2.getRotationMatrix2D((g_size // 2, g_size // 2), angle, 1.0)
    g = cv2.warpAffine(g, M, g.shape)
    result = np.zeros((2 * kernel_radius + 1, 2 * kernel_radius + 1))
    x = random.randint(0, 2 * kernel_radius - g_size)
    y = random.randint(0, 2 * kernel_radius - g_size)
    result[x: x + g_size, y: y + g_size] = g
    return result


def random_di_lobe(kernel_radius, path=None):
    result = np.zeros((2 * kernel_radius + 1, 2 * kernel_radius + 1))
    result += random_lobe(kernel_radius)
    if path is not None:
        cv2.imwrite(jp(path, 'kernel.png'), (result - result.min()) / (result.max() - result.min() * 255).astype(int))
    return result


def get_di_lobe_kernel(kernel_radius, path=None):
    sigmaX = kernel_radius / 2
    sigmaY = sigmaX / 3
    gx = cv2.getGaussianKernel(kernel_radius * 2 + 1, sigmaX).reshape((kernel_radius * 2 + 1, 1))
    gy = cv2.getGaussianKernel(kernel_radius * 2 + 1, sigmaY).reshape((1, kernel_radius * 2 + 1))
    g1 = np.dot(gx, gy)

    sigmaX /= 2
    sigmaY = sigmaX / 3
    g2_size = (kernel_radius // 2) * 2 + 1
    gx = cv2.getGaussianKernel(g2_size, sigmaX).reshape((g2_size, 1))
    gy = cv2.getGaussianKernel(g2_size, sigmaY).reshape((1, g2_size))
    g2 = np.dot(gx, gy)
    tmp = np.zeros(((kernel_radius // 2) * 2 + 1, (kernel_radius // 2) * 2 + 1))
    tmp[(kernel_radius // 2), :] = 1

    num_angle2 = 11
    num_rot = 4
    result = np.zeros((num_rot * num_angle2, *g1.shape))
    names = []

    for i in range(num_angle2):
        if i < 6:
            angle2 = 15 * (i + 1)
            cords = np.array([[kernel_radius // 2], [0], [1]])
            M = cv2.getRotationMatrix2D((g2_size // 2, g2_size // 2), angle2, 1.0)
            g2_rotated = cv2.warpAffine(g2, M, g2.shape)
            res_cord = (np.dot(M, cords)[:, 0]).astype(int)
            result[i * num_rot] = g1 / g1.max() * g2_rotated.max()
            shift = np.abs(np.array([kernel_radius // 2, kernel_radius // 2]) - res_cord)
            result[i * num_rot][shift[0]: shift[0] + g2_size, shift[1]: shift[1] + g2_size] += g2_rotated
            result[i * num_rot] = result[i * num_rot]
            result[i * num_rot] /= result[i * num_rot].sum()
        else:
            angle2 = 15 * (5 - i)
            result[i * num_rot] = result[(i - 6) * num_rot][::-1, :]

        names.append('kernel_{}_angle{}'.format(0, angle2))

        if path is not None:
            cv2.imwrite(jp(path, 'kernel_{}_angle{}.png'.format(0, angle2)),
                        ((result[i * num_rot] - result[i * num_rot].min()) /
                         (result[i * num_rot].max() - result[i * num_rot].min()) * 255).astype(int))
        for j in range(1, num_rot):
            result[i * num_rot + j] = np.rot90(result[i * num_rot + j - 1])
            names.append('kernel_{}_angle{}'.format(j, angle2))
            if path is not None:
                cv2.imwrite(jp(path, 'kernel_{}_angle{}.png'.format(j, angle2)),
                            ((result[i * num_rot + j] - result[i * num_rot + j].min()) /
                             (result[i * num_rot + j].max() - result[i * num_rot + j].min()) * 255).astype(int))
    return result, names
