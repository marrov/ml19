# FLOW Winter School on Machine Learning and Data-Driven Methods

General repository for the FLOW Winter School on Machine Learning and Data-Driven Methods.

December 2-5, 2019

Linné FLOW Centre, KTH Mechanics

Royal Institute of Technology

Stockholm, Sweden

![Winter @ KTH](https://farm9.static.flickr.com/8011/7149987055_b9300791f6_b.jpg)

## Background

With the advancement of computer architectures and power, together with the related increase in the rate of data generation, new computational methods are required to exploit the vast wealth of available information. In particular, new approaches to classification, as well as new algorithms for modeling and prediction, can be developed through data-driven methods and machine learning. 

Fuelled by advances in computer science and through the contribution of large companies such as Google and Amazon, these new approaches are making their way into all disciplines of science, including fluid mechanics and turbulence. 

Despite the potential of these methods, it is essential to be aware of their limitations to identify the areas in which they can be applied successfully. Therefore, this Winter School is aimed at providing the participants with an introductory overview of machine-learning methods, including neural networks, reinforcement learning, and uncertainty quantification, applied to problems relevant to engineering and fluid dynamics.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for usage, development and testing purposes. **Please note** that only Linux environments are supported in the current implementation.

First clone this repository by:

```
git clone https://github.com/mrovirasacie/ml19
```

To use `ml19` several [Python 3](https://www.python.org/) packages are required. Creating a brand new [Conda](https://docs.conda.io/en/latest/) environment for this is recommended. This can be done easily with the provided `yml` file as follows:

```
conda create --name ml19 --file ml19_full.yml
conda activate ml19
```

After executing these commands a new Conda environment names `ml19` will be created with all necessary packages. The environment is self-contained so as to not influence other local python installations and avoid conflicts with previously installed packages. To deactivate this environment simply type:

```
conda deactivate
```

## Monday -  02/12/19

### Introduction to Machine Learning & Regression

#### Canonical learning problems:
  - Supervised regression
  - Supervised classification
  - Unsupervised learning:
    - Classification
    - Dimensionality reduction
  - Reinforment learning: does not have a penalty for failing - only rewards the correct set of actions. Works best when you know most things about your world and you have a closer environmnet. Used for hyperparameter optimisation.

#### Supervised regression

Optimization over two main variables:
  1. Loss function (error of data wrt prediction)
  2. Regularization (measure of complexity)

Overfitting will lead to low training error - the performance of a model should no be measured on training error. Instead we define new data - a test dataset - and calculate the error with respect to new test data instead of the old "learning" dataset.

So overfitting is when *training error* is **low** but *test error* is **high**. 

Overfitting and underfitting can be measured with bias and variance. A graphical overview of these can be seen below:

![Bias and variance](https://miro.medium.com/max/978/1*CgIdnlB6JK8orFKPXpc7Rg.png)


For example linear functions have low bias.

##### Nested cross-validation

Usually data is small - subsets are not satisticallly significant. Then cross validation is required.

Pipeline for model selection:
  1. Gather data
  2. Divide set up in 3 sets: `train`, `validation` and `test` sets
  3. Model selection with `validation`
  4. Train with `train` U `validation`
  5. Measure performance in `test`

[Read more on nested cross-validation](https://weina.me/nested-cross-validation/)

##### L1 regularizer

**Lasso Regression** = *squared-error loss* + *L1 regularization*

L1 regularization is taking L1 norm (absvalue) of weights.

LASSO ≡ Least absolute shrinkage and selection operator

L1 regularizator promotes sparsity - i.e. weight actually go to zero. Is great for feature selection. This encourages noise reduction and we see what inputs really afect the model.

This image shows why LASSO is able to "turn off" some weights. *Left* is **LASSO** and *right* is **Ridge**

![LASSO vs Ridge](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/06/05215637/regular1.png)

Note: *data matrix* has input data in columns - rows track all inputs.

##### Epsilon-sensitive loss and support vector regression

Squared-error loss is not robust to outliers. To solve this another loss function is the epsilon-sensitive loss.

Passed a certain value of `eps = |y_test-y_predict|`

**Support Vector Regression** = *epison-sensitive loss* + *L2 regularization*

We employ Lagrange multipiers - review the math! 

Two hyperparameters:
  - *C* : determines the trade-off between the model complexity andthe degree to which deviations larger than `eps` are tolerated.
  - *eps* : affects the number of support vectors

##### Iterative optimization

- Gradient descent (GD): 
  1. Arbitrary intial point
  2. Compute gradient
  3. Take setp in direction of minus gradient
  4. Repeat step 2-3 until convergence

- Stochastic gradient descent (SDG): gradient descent is slow because gradient computation can be expensive for multiple datasets. Then only one dataset is used.
  - Best metaphor is that many not well though-out steps in directions close to the correct one are better than fewer optimal steps in the absolute optimum direction.

[Review this difference very visually here](https://www.youtube.com/watch?v=IHZwWFHWa-w)

How to choose the learning rate `eta` (i.e. the length of the step in the minus gradient direction)

##### The Kernel Trick

Basically boils down to the fact that a simple function in an initial low dimensional space might be equivalent to the dot product a high dimensional space.

There exists this dimension if a kerner is symmetric and positive semi-definite.

### Hand-on session I: Regression 

[Luca Gastoni's github repository for the session](https://github.com/lguas/FLOW19_MLSchool)

**Homework**: make gradient descent 



## Tuesday - 03/12/19

## Wednesday - 04/12/19

## Thursday - 05/12/19

## Author

* [Marc Rovira](https://github.com/mrovirasacie)
