# Computer Vision

The course 'Computer Vision' is conducted by the [Graphics and Media Lab](https://graphics.cs.msu.ru).

This directory contains [9 Jupyter notebooks](jupyter) with various tasks and code for 5 main cource tasks:

## 1. Combining Image Channels

![](https://github.com/esafronova/EducationalTasks/blob/main/images/cv_task1.png "task1")

The task is to process frames from the first color photographs by Sergey Prokudin-Gorsky in Russia. It is necessary to combine the three channels into a single color image while finding the optimal alignment using Mean Squared Error (MSE) or Cross-Correlation.

It is necessary to implement the [`align`](align.py) function, which takes as input an image obtained by scanning a photographic plate and returns the aligned image.

## 2. Context-aware image scaling

![](https://github.com/esafronova/EducationalTasks/blob/main/images/cv_task2.png "task2")

The task is to implement an algorithm used for context-aware image scaling. In the standard approach, when resizing an image, it uniformly distorts the entire length (objects in the image shrink along with the entire image). However, this algorithm takes context into account, and deformation occurs in such a way that objects maintain their sizes. Additionally, if you use a mask to select an object, you can either remove it from the image or leave it unchanged.

It is necessary to implement the [`seam_carve`](seam_carve.py) function with the following arguments:

- Input image.
- The operation mode of the algorithm, one of four strings:
-- 'horizontal shrink' — horizontal compression.
-- 'vertical shrink' — vertical compression.
-- 'horizontal expand' — horizontal expansion.
-- 'vertical expand' — vertical expansion.
- (Optional argument). Image mask - a single-channel image of the same dimensions as the input image. The mask consists of elements {-1, 0, +1}. -1 indicates pixels to be removed, +1 indicates pixels to be preserved, and 0 means that the energy of pixels should not be changed.

The function returns a tuple: the modified image and mask, along with the seam mask (1 — pixels belonging to the seam; 0 — pixels not belonging to the seam). The corresponding seam should also be removed or added to the mask.

## 3. Recognition of road traffic signs

![](https://github.com/esafronova/EducationalTasks/blob/main/images/cv_task3.png "task3")

The task is to write implementation for calculating HOG (Histogram of Oriented Gradients) features and then find the optimal parameters for an SVM classifier.

It is necessary to implement two functions:

- The HOG feature extraction function, [`extract_hog`](fit_and_classify.py), which takes an input image and extracts HOG features. The feature extraction should be implemented from scratch.
- The classification function, [`fit_and_classify`](fit_and_classify.py), which trains and tests an SVM classifier. This function should not perform a search for optimal classifier parameters; the parameters should already be predefined. 

For training the algorithm, a public dataset of signs will be provided. The program will be tested on two tests: in the first test, the algorithm is trained and tested on the public dataset, and in the second test, it is trained on the public dataset and tested on a hidden dataset.

## 4. Facial keypoints detection

![](https://github.com/esafronova/EducationalTasks/blob/main/images/cv_task4.png "task4")

The task is to implement facial keypoint regression using a convolutional neural network (CNN) with the Keras library on top of TensorFlow. The library is compatible with both CPU and GPU.

It is necessary to implement two functions: [`train_detector`](detection.py), which trains the face keypoint detection model, and [`detect`](detection.py), which performs keypoint detection on images using the trained model. The detection function returns a dictionary with a size of N, where the keys are file names, and the values are arrays of 28 numbers representing the coordinates of the face keypoints [x1, y1, ..., x14, y14]. Here, N is the number of images.

The testing script trains the detector and evaluates the detection quality by calculating the mean squared error (err) on an image resized to 100x100 pixels. 

For training, a public dataset of labeled faces with keypoints is provided. The program is tested on two tests. In each of the tests, the neural network is first trained with the fast_train=True flag in the train_detector function. The training function with this flag should run quickly, no more than 5 minutes. You can set it to train for 1 epoch with a few batches. The trained model is not used for testing; this step is only to check the functionality of the training function. Then, in the first test, the algorithm is tested on the public dataset, and in the second test, it is tested on the hidden dataset. For testing, the trained model 'facepoints_model.hdf5' is loaded. The results of the second test and the final score are hidden until the assignment deadline. The final score is calculated based on the last submission with a non-zero accuracy.

## 5. Classificator finetuning

![](https://github.com/esafronova/EducationalTasks/blob/main/images/cv_task5.png "task5")

The task is to fine-tune a pre-trained neural network for the task of bird species classification using the Keras library.

It is necessary to to implement two functions: [`train_classifier`](classification.py), which trains a classifier based on a pre-trained neural network, and [`classify`](classification.py), which classifies input images using the trained model. The train_classifier function returns the trained neural network model, and classify returns a dictionary with a size of N, where the keys are file names, and the values are numbers representing class labels. Here, N is the number of images.

The testing script classification_test takes input directories with training and testing datasets, trains the detector, and calculates the accuracy (acc) of multi-class classification.

For training, a public dataset of labeled bird images is provided. The program is tested on two tests. In each of the tests, the neural network is first trained with the fast_train=True flag in the train_classifier function. The training function with this flag should run quickly, no more than 15 minutes. You can set it to train for 1 epoch with a few batches. The trained model is not used for testing; this step is only to check the functionality of the training function. Then, in the first test, the algorithm is tested on the public dataset, and in the second test, it is tested on the hidden dataset. For testing, the trained model birds_model.hdf5 is loaded. To fit within the file size limit of the model weights, fine-tune a lightweight pre-trained network (e.g., ResNet50). The results of the second test and the final score are hidden until the assignment deadline. The final score is calculated based on the last submission with non-zero accuracy. To reduce memory consumption, you can use batch generators in Keras for training (the model.fit_generator function).
