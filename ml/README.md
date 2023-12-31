# Machine Learning

There is Jupyter notebook the final task from [ML course](http://www.machinelearning.ru/wiki/index.php?title=Математические_методы_распознавания_образов_%28курс_лекций%2C_В.В.Китов%29).

## [Methods for Density Estimation in Background Subtraction Task](report_task-08.ipynb)

The task is:

1. Implement the Expectation-Maximization (EM) algorithm for recovering a mixture of multivariate normal distributions. Provide the option for multiple runs with random initializations and choose the best result based on the likelihood. Allow for the recovery of a mixture of normal distributions with diagonal covariance matrices. Efficiency requirements: the average time for one EM iteration should not exceed one second for N = 10000 objects, D = 100 features, and K = 10 mixture components. You are only allowed to use numpy, scipy.linalg, and scipy.misc from mathematical libraries.
2. Test the implemented EM algorithm on two-dimensional synthetic data. Generate data from a mixture of distributions with specified parameters and then recover the mixture's parameters using the EM algorithm. Visualize the recovery results, where objects corresponding to the same mixture components are shown in the same color. Ensure that the likelihood value in EM iterations monotonically increases. Provide an example of a situation where the EM algorithm's result depends on the initial approximation.
3. Implement the background model estimation using a one-dimensional Gaussian (Section 4.1) and test it on the pedestrians sequence. Analyze the quality of background subtraction using three methods described in Section 3.
4. Implement the background model estimation based on an adaptive one-dimensional Gaussian (Section 4.2). Test the method on the pedestrians sequence and compare the results with the previous step.
5. Implement the background model estimation using a multivariate Gaussian in the RGB color space and test the results on the pedestrians sequence. Analyze the method's errors and compare it with the previous methods. Repeat the experiment using the HSV color space, where the channels are less correlated. To convert an image, you can use the built-in function matplotlib.colors.rgb_to_hsv. Draw conclusions about the appropriateness of using diagonal covariance matrices for different tasks.
6. Run the mixture separation algorithm for three-dimensional Gaussians to perform background subtraction on the traffic sequence. Select the model based on the results from the previous step (accuracy and speed are crucial). Analyze the method's errors and compare the results with using a single Gaussian.
7. Write a report in the LaTeX system in PDF format or in an IPython notebook with a description of the research results. Include the best solution for the background subtraction algorithm as an animation in the report. The animation can take various forms, such as a widget in an IPython notebook (JSAnimation), a separate video file, or animated images of frames, as shown in the example below.

![](https://github.com/esafronova/EducationalTasks/blob/main/images/em.gif "task8")
