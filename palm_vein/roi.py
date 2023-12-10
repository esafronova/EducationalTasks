import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from os import listdir as ld
from os import makedirs
from os.path import join as jp

from glob import glob
from scipy.signal import find_peaks
from skimage.transform import rotate


def compute_rd(contour, wX, wY):
    w = np.array([wX, wY]).reshape((1, 2))
    d = (contour - w) ** 2
    d = np.sqrt(d.sum(axis=1))
    return d


def extract_roi(in_path, out_path, suf='', L=70, size=128):
    makedirs(jp(out_path, 'L'), exist_ok=True)

    for input_img in sorted(glob(jp(in_path, '0[0-9][0-9]_[l,r]_850_[0-9][0-9].jpg'))):
        try:
            img = cv2.imread(input_img, 0)
            img_name = input_img.split('\\')[-1]
            img_dtls = img_name.split('_')

            th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # th = 0.85 * th
            img_th = np.zeros_like(img)
            img_th[img < th] = 0
            img_th[img >= th] = 255

            _, contours, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = []
            for ci in contours:
                if len(c) < len(ci):
                    c = ci
            # img_rgb = np.dstack((img, img, img))
            # img_c = cv2.drawContours(img_rgb, [c], 0, (0, 255, 0), 1)
            # cv2.imwrite(jp(out_path, 'contour', img_name[:-4] + '_c085.png'), img_c)

            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # img_rgb = np.dstack((img_th, img_th, img_th))
            # cv2.circle(img_rgb, (cX, cY), 3, (255, 0, 0), -1)
            # cv2.imwrite(jp(out_path, 'centroid', img_name[:-4] + '_c085.png'), img_rgb)

            wX = cX - L
            img_th[:, wX:] = 0
            cv2.imwrite(jp(out_path, 'L', img_name[:-4] + '.png'), img_th)
            _, contours, _ = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = []
            for ci in contours:
                if len(c) < len(ci):
                    c = ci

            # img_rgb = np.dstack((img, img, img))
            # cv2.drawContours(img_rgb, [c], 0, (0, 255, 0), 1)
            # cv2.imwrite(jp(out_path, 'L', img_name[:-4] + '_c.png'), img_rgb)
            c = np.array(c)
            c = c[:, 0]
            p = c[:, 1][c[:, 0] == wX - 1]
            wY = (p.max() + p.min()) // 2

            img_rgb = np.dstack((img_th, img_th, img_th))
            cv2.circle(img_rgb, (cX, cY), 3, (255, 0, 0), -1)
            # img_rgb = np.dstack((img, img, img))
            # cv2.circle(img_rgb, (wX, wY), 3, (255, 0, 0), -1)
            # cv2.imwrite(jp(out_path, 'img', img_name[:-4] + 'PW.png'), img_rgb)

            d = compute_rd(c, wX, wY)
            d_h = np.convolve(d, np.ones(140) / 140)[69: -70]
            peaks, _ = find_peaks(-d, distance=70, height=(-d_h+7, None))
            # plt.clf()
            # plt.plot(d)
            # plt.plot(d_h-7, "--", color="gray", lw=0.5)
            # plt.plot(peaks, d[peaks], "x")
            # plt.savefig(jp(out_path, 'img', img_name[:-4] + '_rdf.png'))

            points = np.dstack((c[peaks, 0][c[peaks, 0] < wX - 10], c[peaks, 1][c[peaks, 0] < wX - 10]))[0]
            # img_rgb = np.dstack((img, img, img))
            # for p in points:
            #     cv2.circle(img_rgb, (p[0], p[1]), 3, (125, 0, 125), -1)
            # cv2.imwrite(jp(out_path, 'points1', img_name[:-4] + '.png'), img_rgb)
            if img_dtls[1] == 'r':
                ind = [1, 3]
            else:
                ind = [0, 2]
            # for p in ind:
            #     cv2.circle(img_rgb, (points[p, 0], points[p, 1]), 3, (125, 0, 125), -1)
            # cv2.imwrite(jp(out_path, 'p12_1', img_name[:-4] + '.png'), img_rgb)

            img = img / 255.
            phi = math.atan2(points[ind[0], 1] - points[ind[1], 1],
                             points[ind[0], 0] - points[ind[1], 0]) / math.pi * 180
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), phi, 1.0)
            img_sq = np.zeros((img.shape[1], img.shape[1]))
            img_sq[: img.shape[0], : img.shape[1]] = img
            img_rotated = cv2.warpAffine(img_sq, M, (img.shape[1], img.shape[1]))
            points_new = np.ones((1, 2, 2), dtype=int)
            points_new[0, 0] = points[ind[0]]
            points_new[0, 1] = points[ind[1]]
            points_new = cv2.transform(points_new, M)[0]
            # img_rgb = np.dstack((img_rotated, img_rotated, img_rotated))
            # cv2.circle(img_rgb, (points_new[0, 0], points_new[0, 1]), 3, (125, 0, 125), -1)
            # cv2.circle(img_rgb, (points_new[1, 0], points_new[1, 1]), 3, (125, 0, 125), -1)
            # cv2.imwrite(jp(out_path, 'rotated', img_name[:-4] + '.png'), img_rgb)

            length = (points_new[0, 0] - points_new[1, 0]) // 6
            # img_rgb = np.dstack((img_rotated, img_rotated, img_rotated))
            # cv2.rectangle(img_rgb, (points_new[0, 0], points_new[0, 1] - length),
            #               (points_new[1, 0], points_new[1, 1] - length * 7), (255, 0, 0))
            # cv2.imwrite(jp(out_path, 'draw_roi1', img_name[:-4] + '.png'), img_rgb)
            roi = img_rotated[points_new[0, 1] + length: points_new[0, 1] + 7 * length,
                  points_new[1, 0]: points_new[0, 0]]
            roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(jp(out_path, 'roi1', img_name[:-4] + '.png'), (roi * 255).astype(int))
            # kernel = np.ones((17, 17)) / 17 / 17
            # illumination = cv2.filter2D(roi, -1, kernel)
            # # cv2.imwrite(jp(out_path, 'img', img_name[:-4] + '_ill.png'), (illumination * 255).astype(int))
            # roi = roi - illumination
            # roi = ((roi - roi.min()) / (roi.max() - roi.min()) * 255).astype(int)
            # cv2.imwrite(jp(out_path, 'prepr_roi', img_name[:-4] + '.png'), roi)
        except:
            print(input_img)
