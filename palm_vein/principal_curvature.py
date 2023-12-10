import cv2
import numpy as np
import matplotlib.pyplot as plt
import cmath, math

from os import listdir as ld
from os.path import join as jp
from numpy import linalg as LA
from skimage.transform import resize

epsilon = 0.8
t_1 = 25

def derivative(img, lmd=np.sqrt(2) - 1):
    h, w = img.shape[0] + 2, img.shape[1] + 2
    img_plus = np.zeros((h, w))
    img_plus[1:-1, 1:-1] = img

    img_plus[0, 1:-1] = img_plus[1, 1:-1]
    img_plus[h - 1, 1:-1] = img_plus[-1, 1:-1]
    img_plus[1:-1, 0] = img_plus[1:-1, 1]
    img_plus[1:-1, w - 1] = img_plus[1:-1, -1]
    img_plus[0, 0] = img_plus[1, 1]
    img_plus[h - 1, 0] = img_plus[-1, 1]
    img_plus[0, w - 1] = img_plus[1, -1]
    img_plus[h - 1, w - 1] = img_plus[-1, -1]

    img_x = lmd * (img_plus[1:-1, 2:] - img_plus[1:-1, :-2]) / 2 + (1 - lmd) / 2 * ((img_plus[2:, 2:] -
                                                                                     img_plus[:-2, 2:]) / 2 +
                                                                                    (img_plus[2:, :-2] -
                                                                                     img_plus[:-2, :-2]) / 2)

    img_y = lmd * (img_plus[2:, 1:-1] - img_plus[:-2, 1:-1]) / 2 + (1 - lmd) / 2 * ((img_plus[2:, 2:] -
                                                                                     img_plus[2:, :-2]) / 2 +
                                                                                    (img_plus[:-2, 2:] -
                                                                                     img_plus[:-2, :-2]) / 2)

    return img_x, img_y


def gauss_derivative(sigma, size):
    size = int(size)
    indexes = np.arange(size * 2 + 1)
    indexes -= size
    res = np.exp(- (indexes ** 2) / (2 * (sigma ** 2))) * (- indexes / (2 * cmath.pi * (sigma ** 4)))
    norm = (np.abs(res)).sum()
    res /= norm
    return res


def gradient_gauss(img, sigma=4, size=None):
    if size is None:
        size = math.ceil(sigma * 4)
    gauss_der = gauss_derivative(sigma, size)
    gauss_x = gauss_der.reshape((1, gauss_der.size))
    gauss_y = gauss_der.reshape((gauss_der.size, 1))
    gx = cv2.filter2D(img, cv2.CV_32F, gauss_x)
    gy = cv2.filter2D(img, cv2.CV_32F, gauss_y)
    return gx, gy


def gradient(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return sobelx, sobely


def hessian_threshold(H, gx, gy, gamma=4):
    norm = np.sqrt(gx * gx + gy * gy)
    gamma = gamma / 255 * (norm.max() - norm.min()) + norm.min()
    (H[0, 0])[norm < gamma] = 0
    (H[0, 1])[norm < gamma] = 0
    (H[1, 0])[norm < gamma] = 0
    (H[1, 1])[norm < gamma] = 0
    return H


def hessian_matrix(gx, gy, sigma):
    hx = gx
    hy = gy
    H = np.empty((2, 2, hx.shape[0], hx.shape[1]))

    H[0, 0], H[0, 1] = gradient_gauss(hx, sigma)
    H[1, 0], H[1, 1] = gradient_gauss(hy, sigma)

    H = hessian_threshold(H, gx, gy)
    return H


def principal_curvature(gx, gy, sigma, file_name=None):
    H = hessian_matrix(gx, gy, sigma)
    lam_max = np.zeros(gx.shape)
    lam_min = np.zeros(gx.shape)
    vector_max = np.zeros((gx.shape[0], gx.shape[1], 2))
    vector_min = np.zeros((gx.shape[0], gx.shape[1], 2))

    # for debug
    if file_name is not None:
        f = open(file_name, 'w')

    for i in range(lam_max.shape[0]):
        for j in range(lam_max.shape[1]):
            a = H[:, :, i, j]
            w, v = LA.eig(a)

            # for debug
            if file_name is not None:
                f.write('x = {0}, y = {1}'.format(i, j))
                f.write('\n')
                f.write('[{0}, {1}]'.format(w[0], w[1]))
                f.write(
                    '[[{0}, {1}] [{2}, {3}]]'.format(v[0, 0] * w[0], v[0, 1] * w[0], v[1, 0] * w[1], v[1, 1] * w[1]))
                f.write('\n')

            if (np.isreal(w)).all() and (H[:, :, i, j] != 0).all():
                lam_max[i, j] = w.max()
                lam_min[i, j] = w.min()
                if lam_max[i, j] > 0:
                    vector_max[i, j] = v[:, np.argmax(w)]
                else:
                    lam_max[i, j] = 0
                vector_min[i, j] = v[:, np.argmin(w)]

    # for debug
    if file_name is not None:
        f.close()

    return lam_max, lam_min, vector_max, vector_min


def otsu(img):
    t, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return res


def take_max_gabor(res, image, percent_max, r=8):
    img = np.copy(image)
    img[img == 0] = 255
    img = 255 - img
    for i in range(res.shape[0] - 2 * r):
        for j in range(res.shape[1] - 2 * r):
            tmp = img[i: i + 2 * r, j: j + 2 * r]
            t = tmp.max() * percent_max
            if tmp[r, r] < t:
                res[i + r, j + r] = 0
    return res


def draw_vector(vec1, img, file_name, lam, vec2=None):
    img = resize(img, (img.shape[0] * 6, img.shape[1] * 6))
    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
    res[:, :, 0] = img
    res[:, :, 1] = img
    res[:, :, 2] = img
    for i in range(6, img.shape[0], 6):
        for j in range(6, img.shape[1], 6):
            x = i // 6
            y = j // 6
            if (not (vec1[x, y] == 0).all()) and lam[x, y] > 0:
                # x1 = int(i - 3 * vec1[x, y, 1])
                x1 = i
                x2 = int(i + 10 * vec1[x, y, 1] * lam[x, y])
                # y1 = int(j - 3 * vec1[x, y, 0])
                y1 = j
                y2 = int(j + 10 * vec1[x, y, 0] * lam[x, y])
                cv2.line(res, (y1, x1), (y2, x2), (1, 0, 0))

    if vec2 is not None:
        for i in range(9, img.shape[1] - 10, 10):
            for j in range(9, img.shape[0] - 10, 10):
                if not (vec2[j, i] == 0).all():
                    x1 = int(i - 3 * vec2[x, y, 1])
                    x2 = int(i + 3 * vec2[x, y, 1])
                    y1 = int(j - 3 * vec2[x, y, 0])
                    y2 = int(j + 3 * vec2[x, y, 0])
                    cv2.line(res, (y1, x1), (y2, x2), (0, 1, 0))

    res = (res * 255).astype(np.uint8)
    cv2.imwrite(file_name, res)
    return res


def use_vec(lam, vec, path, file_name):
    for p in range(40, 100, 5):
        res = np.zeros(lam.shape, dtype=np.uint8)
        res[lam > p / 100] = 255
        # cv2.imwrite(jp(path, file_name + '_0start_t{}.png'.format(p)), res)
        flag = True
        num_iteration = 0
        while (flag):
            # cv2.imwrite(jp(path, file_name + '_{}.png'.format(num_iteration)), res)
            num_iteration += 1
            flag = False
            cordy, cordx = np.where(res[1:-1, 1:-1] == 255)
            for n in range(cordy.size):
                i = cordy[n] + 1
                j = cordx[n] + 1
                len = (vec[i, j] ** 2).sum()
                if (len != 0):
                    ln = np.sqrt(20 * lam[i, j])
                    ln = int(ln)
                    for r in range(-ln, ln + 1):
                        x = min(127, max(0, int(j + r * vec[i, j, 1])))
                        y = min(127, max(0, int(i + r * vec[i, j, 0])))
                        if (res[y, x] == 0 and lam[y, x] > 0):
                            flag = True
                            res[y, x] = 255

        cv2.imwrite(jp(path, file_name + '_1finish_t{}.png'.format(p)), res)


def main_process(image, num_degree=10, start=2):
    s = 2 ** 0.25
    lam_max = np.empty((num_degree, image.shape[0], image.shape[1]))
    vector_min = np.empty((num_degree, image.shape[0], image.shape[1], 2))
    for degree in range(start, num_degree + start):
        sigma = s ** degree
        gx, gy = gradient_gauss(image, sigma)
        lam_max[degree - start], lam_min, vector_max, vector_min[degree - start] = principal_curvature(gx, gy, sigma)

    res_lam = np.empty(image.shape)
    res_vector = np.empty((image.shape[0], image.shape[1], 2))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            index = lam_max[:, i, j].argmax()
            res_lam[i, j] = lam_max[index, i, j]
            res_vector[i, j] = vector_min[index, i, j]

    res_lam = (res_lam - res_lam.min()) / (res_lam.max() - res_lam.min())
    return res_lam, res_vector


def main_function(path_input, path_output, num_img_process=None):
    file_names = ld(path_input)
    if (num_img_process is None or num_img_process > len(file_names)):
        num_img_process = len(file_names)
    for i in range(840, num_img_process):
        file = file_names[i]
        image = cv2.imread(jp(path_input, file))
        image = image[:, :, 0]
        lam_max, vector_min = main_process(image)
        lam_max[lam_max < lam_max.max() * t_1 / 100] = 0
        # draw_vector(vector_min, image, jp(path_output, file_names[i][:-4] + 'vector_min.png'), lam_max)
        use_vec(lam_max, vector_min, path_output, file_names[i][:-4])
        res = np.array(lam_max * 255, dtype=np.uint8)
        name = jp(path_output, file_names[i][:-4] + '_lam_max.png')
        cv2.imwrite(name, res)


def net_plus_pc(path1, path2, path_output):
    file_names_1 = ld(path1)
    file_names_2 = ld(path2)
    for i in range(len(file_names_1)):
        img1 = cv2.imread(jp(path1, file_names_1[i]))
        img1 = img1[:, :, 0]
        img2 = cv2.imread(jp(path2, file_names_2[i]))
        img2 = img2[:, :, 0]
        img1 = cv2.resize(img1, img2.shape)
        res = np.copy(img1)
        res[img1 > img2] = img2[img1 > img2]
        name = jp(path_output, file_names_1[i][:-8] + 'net_plus_pc_res.png')
        cv2.imwrite(name, res)


def visualisation_3d(input_file_name):
    image = cv2.imread(input_file_name)
    image = image[:, :, 0]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(image.shape[1])
    Y = np.arange(image.shape[0])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, image)
    plt.show()


def temp(path_input, path_output):
    file_names = ld(path_input)
    num_img_process = len(file_names)
    for i in range(num_img_process):
        file = file_names[i]
        image = cv2.imread(jp(path_input, file))
        image = image[:, :, 0]
        image = cv2.GaussianBlur(image, (15, 15), 0)
        image = resize(image, (128, 128))
        kernel = np.ones((17, 17)) / (17 ** 2)
        average = cv2.filter2D(image,-1,kernel)
        res = image - average
        res = (res - res.min()) / (res.max() - res.min())
        res = np.array(res * 255, dtype=np.uint8)
        name = jp(path_output, file_names[i][:-4] + '_improved15.png')
        cv2.imwrite(name, res)


if (__name__ == '__main__'):
    path_input = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\casia\exstract_roi\850\preprocess_roi'
    path_output = r'C:\Users\Ekaterina\Documents\ImageProcess\vein\casia\principal_curvature_process\850'
    main_function(path_input, path_output)
