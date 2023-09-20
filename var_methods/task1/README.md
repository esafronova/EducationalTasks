# Deconvolution

## Mandatory Part of the Task:
You are required to write a program that implements the method of image deconvolution through the minimization of a regularization functional. The choice of the residual norm and the stabilizer is at the discretion of the student. The recommended stabilizer is the total variation functional or the total generalized variation functional. The recommended number of iterations is 100.

## Requirements for the Program:

Programming Language: Python 3. The testing environment has Python 3.10 installed with additional packages numpy, scipy, scikit-image, and opencv-python.
Interface: The program should have a command-line interface with strict adherence to the input data format. Input parameters during program testing will always be correct; therefore, input parameter validation is not required.
Image Format: All input images will be in the 24-bit BMP format. The images themselves will be in grayscale, meaning that the red, green, and blue components are identical.
Use of Platform-Independent Libraries: It is permissible and recommended to use platform-independent libraries for image reading and saving, auxiliary operations (vector operations, convolution), and parsing command-line parameters.
Prohibited Use of Library Functions: It is not allowed to use library functions that solve the entire task.
Prohibited Side Effects: Performing side actions such as creating temporary files, waiting for input, debug console output, opening dialog boxes, etc., is not allowed.
Evaluation Criteria:
The task is accepted if the quality of the result is higher than the quality of the original image relative to the reference image, measured by the PSNR metric.

## Recommendations:

During testing, images with Gaussian noise with a standard deviation in the range [0, 20] and the following blur kernels will be used: Gaussian filter (parameter from 0.5 to 5), circle (radius from 0.5 to 5), as well as real complex kernels that occur during motion blur. It is recommended to independently model different situations and find optimal parameters depending on the noise level.

## Command-Line Parameters Format:

The program should support command-line execution with a strictly defined command format:
%programname% (input_image) (kernel) (output_image) (noise_level)

Arguments:
- input_image: File name - input blurred and noisy image.
- kernel: File name - blur kernel image.
- output_image: File name - output image.
- noise_level: Noise level on the input image, a real number representing the root mean square deviation (standard deviation) for pixel values in the range [0, 255].
