# Active contours

## Mandatory Part of the Task:

You are required to write a program implementing the image segmentation method using active contours. The choice of external energy is at the discretion of the student, but it must include both line energy (E_line) and edge energy (E_edge). The presence of a balloon force is also optional. A recommended initial approximation for the active contour is provided in the form of a text file, which can be modified if desired. Additionally, a set of utilities (utils.py) is provided to convert the active contour represented as a numpy.array point array into a binary mask.

## Program Requirements:

Implementation Language: Python 3. The program should either consist of a single file or have a main file named main.py. The method should be implemented independently, and using ready-made implementations from libraries like skimage, OpenCV, etc., is not allowed. It is permissible to use libraries for basic operations (interpolation, matrix inversion, gradients, convolutions, etc.) commonly used in image processing (scipy, numpy). The same code should work for all input images, with results dependent only on parameter adjustments. The program's output should be a binary mask of the object, saved as a PNG file, where white (255) represents the object, and black (0) represents the background.
Evaluation Criteria:
A good result is achieved if the program produces a segmentation mask that is close (IoU > 0.7) to the ground truth for 3 out of 5 images.

## Recommendations:

Simple filters such as median or Gaussian can be applied to input images when necessary. You can use numpy.loadtxt to read the initial approximation of the active contour.

## Command-Line Parameters Format:

The program should support command-line execution with a strictly defined command format:
%programname% (input_image) (initial_snake) (output_image) (alpha) (beta) (tau) (w_line) (w_edge) (kappa)

Arguments:

- input_image: File name - input image
- initial_snake: File name - file with the initial approximation for the active contour
- output_image: File name - output image
- alpha: Parameter alpha for internal energy, responsible for contour stretchability
- beta: Parameter beta for internal energy, responsible for contour stiffness
- tau: Gradient descent step size
- w_line: Weight of the intensity term in external energy
- w_edge: Weight of the edge term in external energy
- kappa: Weight of the balloon force (optional)

As a result, please provide the program code, along with a text file specifying the parameters to be used with the program for each image:
%programname% astronaut.png astronaut_init_snake.txt astronaut_result.png 1.0 2.0 0.1 -0.1 0.5 0.1
%programname% coins.png coins_init_snake.txt coins_result.png 2.0 1.0 0.2 0.1 -0.5 0.01
and so on.
