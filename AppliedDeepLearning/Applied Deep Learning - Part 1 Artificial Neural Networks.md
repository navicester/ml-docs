# Applied Deep Learning - Part 1: Artificial Neural Networks

- https://towardsdatascience.com/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6


## Overview
Welcome to the Applied Deep Learning tutorial series. We will do a detailed analysis of several deep learning techniques starting with Artificial Neural Networks (ANN), in particular Feedforward Neural Networks. What separates this tutorial from the rest you can find online is that we’ll take a hands-on approach with plenty of code examples and visualization. I won’t go into too much math and theory behind these models to keep the focus on application.

We will use the Keras deep learning framework, which is a high level API on top of Tensorflow. Keras is becoming super popular recently because of its simplicity. It’s very easy to build complex models and iterate rapidly. I also used barebone Tensorflow, and actually struggled quite a bit. After trying out Keras I’m not going back.

Here’s the table of contents. First an overview of ANN and the intuition behind these deep models. Then we will start simple with Logistic Regression, mainly to get familiar with Keras. Then we will train deep neural nets and demonstrate how they outperform linear models. We will compare the models on both binary and multiclass classification datasets.

```
1. ANN Overview
1.1) Introduction
1.2) Intuition
1.3) Reasoning
2. Logistic Regression
2.1) Linearly Separable Data
2.2) Complex Data - Moons
2.3) Complex Data - Circles
3. Artificial Neural Networks (ANN)
3.1) Complex Data - Moons
3.2) Complex Data - Circles
3.3) Complex Data - Sine Wave
4. Multiclass Classification
4.1) Softmax Regression
4.2) Deep ANN
5. Conclusion
```

The code for this article is available [here](https://github.com/ardendertat/Applied-Deep-Learning-with-Keras/blob/master/notebooks/Part%201%20-%20Artificial%20Neural%20Networks.ipynb) as a Jupyter notebook, feel free to download and try it out yourself.

I think you’ll learn a lot from this article. You don’t need to have prior knowledge of deep learning, only some basic familiarity with general machine learning. So let’s begin…

## 1. ANN Overview
### 1.1) Introduction
Artificial Neural Networks (ANN) are multi-layer fully-connected neural nets that look like the figure below. They consist of an input layer, multiple hidden layers, and an output layer. Every node in one layer is connected to every other node in the next layer. We make the network deeper by increasing the number of hidden layers.

![image](https://user-images.githubusercontent.com/16224205/221116500-12b50dc5-dd0e-4343-8040-9901056f16cd.png)

Figure 1

If we zoom in to one of the hidden or output nodes, what we will encounter is the figure below.

![image](https://user-images.githubusercontent.com/16224205/221116599-c17034f2-1c74-4d43-ba97-d4419ee01bca.png)

Figure 2

A given node takes the weighted sum of its inputs, and passes it through a non-linear activation function. This is the output of the node, which then becomes the input of another node in the next layer. The signal flows from left to right, and the final output is calculated by performing this procedure for all the nodes. Training this deep neural network means learning the weights associated with all the edges.

The equation for a given node looks as follows. The weighted sum of its inputs passed through a non-linear activation function. It can be represented as a vector dot product, where n is the number of inputs for the node.

![image](https://user-images.githubusercontent.com/16224205/221116744-d12c9ba9-1dbf-4247-90e2-2cea30fb2c71.png)

I omitted the `bias` term for simplicity. Bias is an input to all the nodes and always has the value 1. It allows to shift the result of the activation function to the left or right. It also helps the model to train when all the input features are 0. If this sounds complicated right now you can safely ignore the bias terms. For completeness, the above equation looks as follows with the bias included.

![image](https://user-images.githubusercontent.com/16224205/221116973-d8a8059b-78de-41eb-aa93-7b307d3d7af0.png)

So far we have described the _forward pass_, meaning given an input and weights how the output is computed. After the training is complete, we only run the forward pass to make the predictions. But we first need to train our model to actually learn the weights, and the training procedure works as follows:

- Randomly initialize the weights for all the nodes. There are smart initialization methods which we will explore in another article.
- For every training example, perform a _forward pass_ using the current weights, and calculate the output of each node going from left to right. The final output is the value of the last node.
- Compare the final output with the actual target in the training data, and measure the error using a _loss function_.
- Perform a _backwards pass_ from right to left and propagate the error to every individual node using _backpropagation_. Calculate each weight’s contribution to the error, and adjust the weights accordingly using gradient descent. Propagate the error gradients back starting from the last layer.

Backpropagation with gradient descent is literally the “magic” behind the deep learning models. It’s a rather long topic and involves some calculus, so we won’t go into the specifics in this applied deep learning series. For a detailed explanation of gradient descent refer [here](https://iamtrask.github.io/2015/07/27/python-network-part2/). A basic overview of backpropagation is available [here](https://ml.berkeley.edu/blog/2017/02/04/tutorial-3/). For a detailed mathematical treatment refer [here](http://cs231n.github.io/optimization-2/) and [here](http://neuralnetworksanddeeplearning.com/chap2.html). And for more advanced optimization algorithms refer [here](http://ruder.io/optimizing-gradient-descent/index.html).

In the standard ML world this feed forward architecture is known as the _multilayer perceptron_. The difference between the ANN and perceptron is that ANN uses a non-linear activation function such as _sigmoid_ but the perceptron uses the step function. And that non-linearity gives the ANN its great power.

### 1.2) Intuition
There’s a lot going on already, even with the basic forward pass. Now let’s simplify this, and understand the intuition behind it.

> Essentially what each layer of the ANN does is a non-linear transformation of the input from one vector space to another.

Let’s use the ANN in Figure 1 above as an example. We have a 3-dimensional input corresponding to a vector in 3D space. We then pass it through two hidden layers with 4 nodes each. And the final output is a 1D vector or a scalar.

So if we visualize this as a sequence of vector transformations, we first map the 3D input to a 4D vector space, then we perform another transformation to a new 4D space, and the final transformation reduces it to 1D. This is just a chain of matrix multiplications. The forward pass performs these matrix dot products and applies the activation function element-wise to the result. The figure below only shows the weight matrices being used (not the activations).

![image](https://user-images.githubusercontent.com/16224205/221119025-922cd6c2-7e61-480d-b179-90419ab8c968.png)

Figure 3

The input vector x has 1 row and 3 columns. To transform it into a 4D space, we need to multiply it with a 3x4 matrix. Then to another 4D space, we multiply with a 4x4 matrix. And finally to reduce it to a 1D space, we use a 4x1 matrix.

Notice how the dimensions of the matrices represent the input and output dimensions of a layer. The connection between a layer with 3 nodes and 4 nodes is a matrix multiplication using a 3x4 matrix.

These matrices represent the weights that define the ANN. To make a prediction using the ANN on a given input, we only need to know these weights and the activation function (and the biases), nothing more. We train the ANN via backpropagation to “learn” these weights.

If we put everything together it looks like the figure below.

![image](https://user-images.githubusercontent.com/16224205/221119129-1d23f0cb-ed71-470a-b4a1-2eae25060894.png)

Figure 4

A fully connected layer between 3 nodes and 4 nodes is just a matrix multiplication of the 1x3 input vector (yellow nodes) with the 3x4 weight matrix W1. The result of this dot product is a 1x4 vector represented as the blue nodes. We then multiply this 1x4 vector with a 4x4 matrix W2, resulting in a 1x4 vector, the green nodes. And finally a using a 4x1 matrix W3 we get the output.

We have omitted the activation function in the above figures for simplicity. In reality after every matrix multiplication, we apply the activation function to each element of the resulting matrix. More formally

![image](https://user-images.githubusercontent.com/16224205/221119464-303b0f64-e3ed-42c4-9373-ff7ceab8c493.png)

Equation 2

The output of the matrix multiplications go through the activation function f. In case of the sigmoid function, this means taking the sigmoid of each element in the matrix. We can see the chain of matrix multiplications more clearly in the equations.




