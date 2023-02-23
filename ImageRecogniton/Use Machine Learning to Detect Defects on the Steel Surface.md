# Use Machine Learning to Detect Defects on the Steel Surface
## Definition
### Project Overview
Surface quality is the essential parameter for steel sheet. In the steel industry, manual defect inspection is a tedious assignment. Consequently, it is difficult to guarantee the surety of a flawless steel surface. To meet user requirements, vision-based automatic steel surface investigation strategies have been proven to be exceptionally powerful and prevalent solutions over the past two decades<sup>1</sup>.

The input is taken from the NEU surface defect database<sup>2</sup>, which is available online. This database contains six types of defects including crazing, inclusion, patches, pitted surface, rolled-in-scale, and scratches.

### Problem Statement
The challenge is to provide an effective and robust approach to detect and classify metal defects using computer vision and machine learning.

Image preprocessing techniques such as filtering and extracting the features from the image is a good training model solution from which we can determine which type of defect the steel plate has. This solution can even be used in real-time applications.

### Metrics
The evaluation is done using accuracy metrics. The following shows the accuracy of the system given:

![image](https://user-images.githubusercontent.com/16224205/220947649-3eebe1e6-f346-4673-9d34-da8c17a712eb.png)

Because the classes are balanced, accuracy is an appropriate metric to evaluate the project. The accuracy tells us about how well the algorithm is classifying the defects.

## Analysis
### Data Exploration
The NEU surface dataset<sup>2</sup> contains 300 pictures of each of six deformities (a total of 1800 images). Each image is 200 × 200 pixels. The images given in the dataset are in the .bmp format. The images in the dataset are gray-level images of 40.1 KB each. A few samples are shown in figure 1.

![image](https://user-images.githubusercontent.com/16224205/220947979-80a90509-612e-48c0-8559-fc16022aeba7.png)

Figure 1. Samples of defects (a) crazing, (b) inclusion, (c) patches, (d) pitted surface, (e) rolled-in-scale, and (f) scratches.

### Exploratory Visualization
The following chart shows the histogram of images per class.

![image](https://user-images.githubusercontent.com/16224205/220948174-2fa9c67d-5869-466d-88f3-a6cedb17e569.png)

Figure 2. Histogram samples of defects: (a) crazing, (b) inclusion, (c) patches, (d) pitted surface, (e) rolled-in-scale, and (f) scratches.

An image histogram acts as a [graphical representation](https://en.wikipedia.org/wiki/Graphical_representation) of the [tonal](https://en.wikipedia.org/wiki/Lightness_(color)) distribution in a [digital image](https://en.wikipedia.org/wiki/Digital_image). The [horizontal axis](https://en.wikipedia.org/wiki/Horizontal_axis) of the [graph](https://en.wikipedia.org/wiki/Graphics) represents the intensity variations; the [vertical axis](https://en.wikipedia.org/wiki/Vertical_axis) represents the number of pixels of that particular intensity. A histogram gives us an idea of the contrast of the image that I used as a feature. It is important to observe the histogram of the image to get an overview of the feature, like contrast. From figure 2 it is observed that the histogram of each class is visually distinguishable, which makes contrast an important feature to be included in the feature vector.

As said earlier, the classes are well balanced, justifying accuracy as an evaluation metric.

### Algorithms and Techniques
Different classifiers such as k-nearest neighbors (KNN), support vector classifier (SVC), gradient boosting, random forest classifier, AdaBoost (adaptive boosting), and decision trees will be compared.

Texture features such as contrast, dissimilarity, homogeneity, energy, and asymmetry will be extracted from the gray-level co-occurrence matrix (GLCM), and used for training the classifiers.

### SVM
SVM is classified into linear and nonlinear. The linear SVM classifier is worthwhile to the nonlinear classifier to map the input pattern into a higher dimensional feature space. The data that can be linearly separable can be examined using a hyperplane, and the data that are linearly non-separable are examined methodically with kernel function, like a higher order polynomial. The SVM classification algorithm is based on different kernel methods; that is, radial basic function (RBF), and linear and quadratic kernel function. The RBF kernel is applied on two samples, x and x', which indicate as feature vectors in some input space and it can be defined as:

![image](https://user-images.githubusercontent.com/16224205/220949873-7ac2e387-4891-4efb-a1d7-e29b76469eff.png)

The value of the kernel function is decreased according to distance, and ranges between zero (in the limit) and one (when x = x').

![image](https://user-images.githubusercontent.com/16224205/220949955-c9415285-5cbf-44e5-b3d2-4a123f4aa3b8.png)

