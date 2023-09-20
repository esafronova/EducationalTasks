import numpy as np
from os import listdir as ld
from os.path import join as jp
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from cv2 import CascadeClassifier
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from skimage.exposure import equalize_adapthist, equalize_hist

size_nn = 224
size_facepoint = 96


def preprocess_imgs(images, cascade_path, is_video, model):
    num = len(images)
    face_cascade = CascadeClassifier(cascade_path)
    indexes = range(num)
    if is_video:
        indexes = range(0, num, 2)
    faces = [] # np.empty((len(indexes), size_nn, size_nn, 3))
    grays = np.empty((len(indexes), size_facepoint, size_facepoint, 1))

    for i in range(len(indexes)):
        img = images[indexes[i]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray)
        if len(face) > 0:
            (x, y, w, h) = face[0]
            for j in range(1, len(face)):
                (x_, y_, w_, h_) = face[j]
                if w_ * h_ > w * h:
                    (x, y, w, h) = face[j]
            dw = int(0.1 * w)
            dh = int(0.15 * h)
            ind = [max(x - dw, 0), min(x + w + dw, img.shape[1]), max(y - dh, 0), min(y + h + dh, img.shape[0])]
            w, h = ind[1] - ind[0], ind[3] - ind[2]
            x, y = (ind[1] + ind[0]) // 2, (ind[3] + ind[2]) // 2
            s = max(w, h) // 2
            w, h = img.shape[1] - 1, img.shape[0] - 1
            img = img[max(y - s, 0): min(y + s, h), max(x - s, 0): min(x + s, w)]
            faces.append(resize(img, (size_nn, size_nn, 3)))
            gray = gray[max(y - s, 0): min(y + s, h), max(x - s, 0): min(x + s, w)]
            grays[i, :, :, 0] = resize(gray, (size_facepoint, size_facepoint))

    faces = np.array(faces)
    facepoints = model.predict(grays)
    facepoints = (facepoints + 1) / 2
    facepoints[1::2] *= size_nn
    facepoints[0::2] *= size_nn
    for i in range(faces.shape[0]):
        points = facepoints[i]
        alpha = math.atan2(points[17] - points[11], points[16] - points[10])
        alpha = alpha * 180 / math.pi
        M = cv2.getRotationMatrix2D(((points[17] - points[11]) / 2, (points[16] - points[10]) / 2),
                                            alpha, 1.0)
        faces[i] = cv2.warpAffine(faces[i], M, (size_nn, size_nn))
        points = points.reshape((-1, 2))
        ones = np.ones(shape=(points.shape[0], 1))
        points_ones = np.hstack([points, ones])
        points = M.dot(points_ones.T).T.flatten()

        d = math.sqrt((points[17] - points[11]) ** 2 + (points[16] - points[10]) ** 2)
        desired_d = 0.4 * size_nn
        M = cv2.getRotationMatrix2D((size_nn / 2, size_nn / 2), 0, desired_d / d)
        faces[i] = cv2.warpAffine(faces[i], M, (size_nn, size_nn))
    return faces


class Classifier:
    def __init__(self, nn_model, cascade_path):
        self.model = nn_model
        self.cascade_path = cascade_path
        self.svc = LinearSVC(C=16)
        #RandomForestClassifier(n_estimators=1000, oob_score=True, criterion='entropy', bootstrap=True, n_jobs=-1)
        self.facepoints_model = load_model('facepoints_model.hdf5')

    def fit(self, train_imgs_dir, train_labels, name):
        x, y = read_images_gt(train_imgs_dir, train_labels)
        # x = preprocess_imgs(x, self.cascade_path, False, self.facepoints_model)
        # features = self.model.predict(x, batch_size=16)
        # np.save(name, features)
        features = np.load(name)
        self.svc.fit(features, y)

    def classify_images(self, test_imgs_dir):
        x, names = read_images_gt(test_imgs_dir)
        x = preprocess_imgs(x, self.cascade_path, False, self.facepoints_model)
        features = self.model.predict(x, batch_size=16)
        y = self.svc.predict(features)
        y_result = {}
        for i in range(len(names)):
            y_result[names[i]] = y[i]
        return y_result

    def classify_videos(self, test_video_dir):
        y = {}
        path_names = ld(test_video_dir)
        path_names.sort()
        for i in range(len(path_names)):
            x, tmp = read_images_gt(jp(test_video_dir, path_names[i])) # 1.0994891284647288
            x = preprocess_imgs(x, self.cascade_path, True, self.facepoints_model) # 11.167728278968028
            features = self.model.predict(x, batch_size=16)
            pred = list(self.svc.predict(features))
            y[path_names[i]] = (max(set(pred), key=pred.count))
            '''
            pred = self.rfc.predict_proba(features)
            votes = pred.sum(axis=0)
            y[path_names[i]] = self.rfc.classes_[votes.argmax()]
            '''
        return y


def read_images_gt(img_path, gt_file=None):
    file_name = ld(img_path)
    file_name.sort()
    num = len(file_name)
    x = []
    if gt_file is not None:
        y = []
    else:
        y = file_name

    for i in range(num):
        img = imread(jp(img_path, file_name[i]))
        x.append(img)
        if gt_file is not None:
            y.append(gt_file.get(file_name[i]))

    if gt_file is not None:
        y = np.array(y)
    return x, y


def read_video(img_path):
    path_names = ld(img_path)
    path_names.sort()
    x = []
    num = [0]
    for i in range(len(path_names)):
        file_names = ld(jp(img_path, path_names[i]))
        file_names.sort()
        for j in range(1, len(file_names), 2):
            img = imread(jp(img_path, path_names[i], file_names[j]))
            x.append(img)
        num.append(len(x))
    return x, num, path_names