import numpy as np
from skimage import filters
from skimage.transform import resize
from sklearn.svm import SVC, LinearSVC
import math
from sklearn.model_selection import cross_val_score
from skimage.exposure import adjust_gamma
from skimage.io import imshow


def extract_hog(img, cellRows=6, cellCols=6, binCount=9, blockRowCells=2, blockColCells=2, image_size=32):
    grayscale = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    grayscale = filters.gaussian(grayscale, sigma=0.5)
    grayscale = resize(grayscale, (image_size, image_size))
    grayscale = adjust_gamma(grayscale, gamma=2)

    sobelx = filters.sobel_v(grayscale)
    sobely = filters.sobel_h(grayscale)
    gradient_mod = np.sqrt(sobelx ** 2 + sobely ** 2)
    direction = np.arctan2(sobely, sobelx)
    direction[direction < 0] += math.pi

    tmp = np.zeros((image_size, image_size))
    tmp[image_size // 2 - 1: image_size // 2 + 1, image_size // 2 - 1: image_size // 2 + 1] = 1
    crop_size = 4
    s = image_size // 2 - crop_size
    tmp = filters.gaussian(tmp, sigma=s)
    gradient_mod *= tmp
    gradient_mod = gradient_mod[crop_size: -crop_size, crop_size: -crop_size]
    direction = direction[crop_size: -crop_size, crop_size: -crop_size]

    row_cell = int(math.ceil(direction.shape[0] / cellRows))
    col_cell = int(math.ceil(direction.shape[1] / cellCols))
    num_rows_block = cellRows - blockRowCells + 1
    num_cols_block = cellCols - blockColCells + 1

    angle_array = np.empty(binCount)
    for k in range(binCount):
        angle_array[k] = k * math.pi / binCount + math.pi / (2 * binCount)

    cell_hog = np.zeros((cellRows, cellCols, binCount))
    for i in range(cellRows):
        for j in range(cellCols):
            gradient_crop = gradient_mod[i * row_cell: (i + 1) * row_cell, j * col_cell: (j + 1) * col_cell]
            direction_crop = direction[i * row_cell: (i + 1) * row_cell, j * col_cell: (j + 1) * col_cell]

            for x in range(row_cell):
                for y in range(col_cell):
                    dir = direction_crop[x, y]
                    nearest = (np.absolute(angle_array - dir)).argmin()

                    if (dir >= angle_array[nearest] and nearest == binCount - 1) or \
                            (dir < angle_array[nearest] and nearest == 0):
                        if nearest == 0:
                            cell_hog[i, j, 0] += gradient_crop[x, y]
                        else:
                            cell_hog[i, j, -1] += gradient_crop[x, y]
                    else:
                        left_i = nearest
                        if dir <= angle_array[nearest]:
                            left_i = nearest - 1
                        cell_hog[i, j, left_i] += gradient_crop[x, y] * (
                            angle_array[left_i + 1] - dir) * binCount / math.pi
                        cell_hog[i, j, left_i + 1] += gradient_crop[x, y] * (
                            dir - angle_array[left_i]) * binCount / math.pi

    eps = 0.000001
    block_hog = np.zeros((num_rows_block, num_cols_block, binCount * blockRowCells * blockColCells))
    for i in range(num_rows_block):
        for j in range(num_cols_block):
            crop_hog = cell_hog[i: i + blockRowCells, j: j + blockColCells]
            block_hog[i, j] = crop_hog.reshape(crop_hog.size)
            block_hog[i, j] /= math.sqrt((block_hog[i, j] ** 2).sum() + eps)
    hog = block_hog.reshape(block_hog.size)
    return hog


def fit_and_classify(train_x, train_y, test_x):
    clf = SVC(kernel='linear')
    # print(cross_val_score(clf, train_x, train_y, n_jobs=-1))
    clf.fit(train_x, train_y)
    predicted_y = clf.predict(test_x)
    return predicted_y