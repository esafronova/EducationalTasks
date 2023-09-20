import sys
import cv2
import numpy as np
from utils import save_mask
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import inv


def create_A(alpha, beta, n):
    A = np.zeros((n, n))
    np.fill_diagonal(A, 2 * alpha + 6 * beta)
    A = np.roll(A, 1, axis=1)
    np.fill_diagonal(A, -alpha - 4 * beta)
    A = np.roll(A, 1, axis=1)
    np.fill_diagonal(A, beta)
    A = np.roll(A, -3, axis=1)
    np.fill_diagonal(A, -alpha - 4 * beta)
    A = np.roll(A, -1, axis=1)
    np.fill_diagonal(A, beta)
    A = np.roll(A, 2, axis=1)
    return A


def P_line(image, ksize=3, sigma=0.):
    smooth = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return smooth


def P_edge(image, ksize=3, sigma=0.):
    smooth = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    gx = cv2.Sobel(smooth, -1, 1, 0)
    gy = cv2.Sobel(smooth, -1, 0, 1)
    g = gx ** 2 + gy ** 2
    g = np.sqrt(g)
    return g


def norm_F(F, tau):
    norm = np.sqrt((F ** 2).sum(axis=1))
    norm = np.max(norm)
    k = 10. / tau
    F = k * F / norm
    return F


def ac_segmentation(input_image, initial_snake, alpha, beta, tau, w_line, w_edge, max_num_steps=2000, eps=0.1):
    N = initial_snake.shape[0]
    prev = initial_snake.copy()
    result = prev.copy()
    diff = eps + 1
    step_num = 0
    image = input_image.astype(float)

    p_line = P_line(image)
    p_edge = P_edge(image)
    p = w_line * p_line + w_edge * p_edge
    interp = RectBivariateSpline(np.arange(p.shape[1]), np.arange(p.shape[0]), p.T)

    A = create_A(alpha, beta, N)

    right_part = np.eye(prev.shape[0]) + tau * A
    inv_right_part = inv(right_part)

    while (step_num < max_num_steps) and (diff > eps):
        fx = interp(prev[:, 0], prev[:, 1], dx=1, grid=False)
        fy = interp(prev[:, 0], prev[:, 1], dy=1, grid=False)
        F = np.vstack((fx, fy)).T
        F = norm_F(F, tau)
        left_eq_part = prev + tau * F
        result = np.matmul(inv_right_part, left_eq_part)

        diff = result - prev
        diff = (diff ** 2).sum(axis=1)
        diff = (np.sqrt(diff)).max()

        prev = result.copy()
        step_num += 1

    return result


if __name__=='__main__':
    arguments = sys.argv[1:]
    input_image = cv2.imread(arguments[0], 0)
    initial_snake = np.loadtxt(arguments[1])
    alpha = float(arguments[3])
    beta = float(arguments[4])
    tau = float(arguments[5])
    w_line = float(arguments[6])
    w_edge = float(arguments[7])
    contour = ac_segmentation(input_image, initial_snake, alpha, beta, tau, w_line, w_edge)
    save_mask(arguments[2], contour, input_image)

