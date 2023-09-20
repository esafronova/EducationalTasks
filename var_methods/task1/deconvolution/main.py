import sys
import cv2
import numpy as np


def shift(img, x, y, fill_value=0):
    result = np.roll(img, x, axis=0)
    result = np.roll(result, y, axis=1)
    if x > 0:
        result[:x] = fill_value
    if x < 0:
        result[x:] = fill_value
    if y > 0:
        result[:, y] = fill_value
    if y < 0:
        result[:, y:] = fill_value
    return result


def deconvolution(input_image, kernel, noise_level, num_iteration=100):
    z = (input_image.copy()).astype(float)
    alpha = noise_level / 255. + 0.5
    lr = 0.5
    Q = [(1, 0), (0, 1), (1, 1), (-1, -1)]
    for i in range(num_iteration):
        lr_i = lr / np.sqrt(i + 1)

        D_discrepancy = cv2.filter2D(z, -1, kernel)
        D_discrepancy = cv2.filter2D(D_discrepancy, -1, kernel[::-1, ::-1])
        D_discrepancy = D_discrepancy - cv2.filter2D(input_image, -1, kernel[::-1, ::-1])

        D_btv = np.zeros_like(z)
        for (x, y) in Q:
            sgn = np.sign(shift(z, x, y) - z)
            d = shift(sgn, -x, -y) - sgn
            d = d / np.sqrt(x ** 2 + y ** 2)
            D_btv += d

        D = D_discrepancy + alpha * D_btv
        z = z - lr_i * D

    return z.astype(int)


if __name__=='__main__':
    arguments = sys.argv[1:]
    input_image = cv2.imread(arguments[0], 0)
    kernel = cv2.imread(arguments[1], 0)
    if kernel.sum() != 0:
        kernel = kernel / kernel.sum()

    result = deconvolution(input_image, kernel, float(arguments[3]))
    cv2.imwrite(arguments[2], result)
