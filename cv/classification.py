import numpy as np
from os import listdir as ld
from os.path import join as jp
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, InputLayer
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from os.path import dirname, abspath

image_size = 224


def train_classifier(train_gt, train_img_path, fast_train=False):
    if fast_train:
        return ResNet50(weights='imagenet', include_top=False)

    else:
        train_x, train_y = read_images_gt(train_img_path, train_gt)
        train_y = to_categorical(train_y)
        N = train_y.shape[1]

        base_model = ResNet50(weights='imagenet', include_top=True)
        x = Dense(N, activation='softmax', name='predictions')(base_model.layers[-2].output)
        new_model = Model(input=base_model.input, output=x)

        for layer in new_model.layers[:-1]:
            layer.trainable = False

        lr = 0.01
        nb_epoch = 100
        sgd = SGD(lr=lr, decay=1e-3, momentum=0.9)
        gen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
        new_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        new_model.summary()
        change_lr = ReduceLROnPlateau(patience=5)

        new_model.fit_generator(gen.flow(train_x, train_y),
                                samples_per_epoch=train_x.shape[0],
                                nb_epoch=nb_epoch,
                                validation_data=(train_x, train_y),
                                callbacks=[change_lr])

        new_model.fit_generator(gen.flow(train_x, train_y),
                            samples_per_epoch=train_x.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(train_x, train_y))
        new_model.save('birds_model.hdf5')
        return new_model


def classify(model, test_img_path):
    keys = ld(test_img_path)
    keys.sort()
    dict_y = {}
    for i in range(len(keys)):
        x = imread(jp(test_img_path, keys[i]))
        x = resize(x, (image_size, image_size, 3))
        x = preprocess_input(np.expand_dims(x * 255, axis=0))
        pred = (model.predict(x)[0]).argmax()
        dict_y[keys[i]] = pred

    return dict_y


def read_images_gt(img_path, gt_file=None):
    file_name = ld(img_path)
    file_name.sort()
    num = len(file_name)

    x = np.empty((num, image_size, image_size, 3))
    y = None
    if gt_file is not None:
        y = np.empty(num, dtype=int)

    for i in range(num):
        img = imread(jp(img_path, file_name[i]))
        x[i] = resize(img, (image_size, image_size, 3))
        if gt_file is not None:
            y[i] = gt_file.get(file_name[i])

    x = preprocess_input(255 * x)

    return x, y


def plot_loss(loss, val_loss, file_name=jp(dirname(abspath(__file__)), 'history_birds.png')):
    plt.plot(np.arange(len(loss)), np.array(loss), linewidth=3, label='train')
    plt.plot(np.arange(len(val_loss)), np.array(val_loss), linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.savefig(file_name)