In my previous article 'Introduction to ML', we explored what machine learning really is and got a good sense of what it does. From this article, we will become slightly more technical. I will explain certain notations and terminologies, and some basic ideas that we will need later. 

## The Setup:

### Notation:
The following guide outlines the mathematical notation used throughout these articles. Please refer back to it if you are unsure of any symbols.

**1. General Mathematical Symbols**

| **Symbol**                              | **Meaning**                                                            |
| --------------------------------------- | ---------------------------------------------------------------------- |
| $a,b,c \cdots x,y,z$                    | Scalar variables (single numbers)                                      |
| $f(x)$                                  | Function $f$ applied to $x$                                            |
| $:=$                                    | Assignment operator                                                    |
| $\displaystyle \sum_{i=1}^m$            | Summation from i=1 to m                                                |
| $\frac {\partial J}{\partial \theta_j}$ | Partial derivative of function $J$ with respect to variable $\theta_j$ |
**2. Dataset-Specific Notation**

| **Symbol**         | **Meaning**                                                                          | **Type/Dimensions (Typical)** |
| ------------------ | ------------------------------------------------------------------------------------ | ----------------------------- |
| $m$                | Total no. of training examples in the dataset                                        | Scalar                        |
| $d$                | No. of features in each training example                                             | Scalar                        |
| $\mathbf{x}$       | A feature vector representing a single training example                              | Vector / $(d+1) \times 1$     |
| $\mathbf{x}^{(i)}$ | Feature vector of $i^{th}$ training example (1$\leq$$i$$\leq$$m$)                    | Vector / $(d+1) \times 1$     |
| $x_j$              | Value of the $j^{th}$ feature in a feature vector (0$\leq$$j$$\leq$$d$)              | Scalar                        |
| $x_j^{(i)}$        | Value of the $j^{th}$ feature in the $i^{th}$ training example                       | Scalar                        |
| $y$                | The target variable (or label) for a single training example                         | Scalar                        |
| $y^{(i)}$          | Target variable for the $i^{th}$ training example                                    | Scalar                        |
| $X$                | The training data matrix, where each column is a training example $\mathbf{x}^{(i)}$ | Matrix / $m \times (d+1)$     |
| $Y$                | A vector containing all target variables $y^{(i)}$                                   | Vector / $m \times 1$         |
**3. Model Parameters and Hypothesis**

| **Symbol**                    | **Meaning**                                                                                                         | **Type/Dimensions (Typical)** |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| $\boldsymbol{\theta}$ (theta) | Vector of model parameters (a.k.a. weights or coefficients)                                                         | Vector / $(d + 1) \times 1$   |
| $\theta_j$                    | The $j^{th}$ parameter (or weight) in vector $\boldsymbol{\theta}$                                                  | Scalar                        |
| $h(x)$ or $h_\theta(x)$       | The hypothesis function. Given and input $\mathbf{x}$ and parameters $\boldsymbol{\theta}$, it outputs a prediction | Scalar                        |
| $\hat{y}$ (y-hat)             | The predicted value of $y$, often used interchangeably with $h_\theta(x)$                                           | Scalar                        |
**4. Cost Function and Optimization**

| **Symbol**               | **Meaning**                                                                                        | **Type** |
| ------------------------ | -------------------------------------------------------------------------------------------------- | -------- |
| $J(\boldsymbol{\theta})$ | The cost function. Measures how well the model performs for given parameters $\boldsymbol{\theta}$ | Scalar   |
| $\alpha \text{(alpha)}$  | The learning rate in gradient descent. Controls step size                                          | Scalar   |
| $t$                      | Iteration number in iterative algorithm                                                            | Scalar   |
| $b$                      | Mini-batch size                                                                                    | Scalar   |
| $\mathfrak{B}$           | A specific mini-batch i.e. a subset of training examples                                           | Scalar   |


### The Hypothesis:
In machine learning, our main job is to make the machine learn from some data and output a prediction. But how will the machine decide what to predict? For example, when trained on house price data, how will the machine 'learn' from it and then give a prediction on any new data we feed it? To do this, we must have a hypothesis. A hypothesis can be viewed as a simple formula, which takes an input and gives us an output. We can represent it as:

![[hypothesis.excalidraw]]
where:
h: is our hypothesis (or function)
x: is our input
y: is our output

A good hypothesis is one which takes an input, and gives us an accurate output. Now let us see what our input looks like.

### Given:

In case of machine learning, input will refer to our input data. Our input data consists of :
1. Training data - we train our model on this
2. Testing data - we test the accuracy of our hypothesis on this

Now, let us look at what our training data looks like:

Training dataset: $\{(x^{(1)}, y^{(1)}),...,(x^{(m)}, y^{(m)})\}$
where:
${x^{(i)} \in X }\text{ and X is the set of all inputs}$
${y^{(i)} \in Y }\text{ and Y is the set of all labels (actual outputs)}$
$\text{m is the total number of training examples}$

### To Do:
Now that we know what we have been given, we must figure out what to do with the data. In machine learning, our main objective will be to take the data, train the machine on it, and find a 'good' hypothesis h. Here, 'good' can usually be defined by how accurate our hypothesis is. 

It is important to understand that we are not really focused on successfully interpolating *known* data but rather on how well our hypothesis performs on a new pair $(x^{(i)}, y^{(i)})$.

But where does this 'new pair' of data come from? Well, it helps to think of our training data as a small sample from a large population of data. Our new pair is just an example that is present in the large population, but absent in our training data. 

## A More Detailed View:

We have discussed some important concepts regarding machine learning. Now, let us take a closer look at them.

### Representing h:

Let us break down our hypothesis h and see what we can make of it. As I mentioned earlier, a hypothesis is nothing but a function. The simplest kind of functions are straight lines. Consider the diagram below. 

![[chapter2-graph1.excalidraw]]

Here, we can see a line. A line can be described by its slope   ($\theta_1$) and its y-intercept ($\theta_0$). We can define our hypothesis as:

$h(x)\ =\ \theta_0\ +\ \theta_1*x_1\quad where\ \theta_0\ and\ \theta_1\ are\ called\ weights.$ 

As you can see in our case, *h* is a simple affine function (it is not linear since linear functions do not have y-intercepts). For an actual training set with more data points, we could represent *h* as:

$h(x)\ =\ \theta_0*x_0\ +\ \theta_1*x_1\ ...\ \theta_d*x_d$ ($x_0=1$ by convention)

### Representing Actual Training Data:

Real world training data not only has multiple training examples, but also comes with multiple features. In that case, we can represent the training data as:
$$
x^{(i)}_j\quad where\ i=training\ example,\\ \ j=feature\ number
$$
$$
\underbrace{
\mathbf{\theta} = 
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_d
\end{bmatrix}
}_{\text{parameter vector}}
\quad ; \quad
\underbrace{
\mathbf{X^{(1)}} = 
\begin{bmatrix}
x^{(1)}_0 \\
x^{(1)}_1 \\
\vdots \\
x^{(1)}_d
\end{bmatrix}
}_{\text{feature vector of example 1}}
$$

The entire training data can be represented as:

$$
\begin{array}
\mathbf{X}=\begin{bmatrix}
x_0^{(1)}\quad x_1^{(1)}\quad \cdots \quad x_d^{(1)}\\
x_0^{(2)}\quad x_1^{(2)}\quad \cdots\quad x_d^{(2)}\\
\vdots \\
x_0^{(m)}\quad x_1^{(m)}\quad \cdots\quad x_d^{(m)}
\end{bmatrix}_{(m \times {(d+1)})} \in \mathbb{R}
\\[1em]
\text{where } m = \text{number of examples}, \\
(d+1) = \text{number of features (including } x_0\text{)}
\end{array}
$$
We now have:    $h_\theta(x) = \displaystyle \sum_{j=0}^{d} \theta_jx_j$
We want:   $h_\theta(\mathbf{X}) \approx \mathbf{Y}$ i.e. our hypothesis gives fairly accurate results.

### The Cost Function:

The Cost Function, along with gradient descent which we will see next, is one of the most important concepts in all of machine learning. We saw above that we want our predictions to be fairly equal to the actual answer i.e. we want an accurate hypothesis. But how do we determine whether the hypothesis is accurate i.e. it is valid or not? This is the work of a cost function. The idea that we will see here is fairly simple and yet is applicable even for some of the most advanced machine learning models in existence today.

We start off with something called the 'errors' or the 'residuals'. These are nothing but the differences between the predicted value $h_\theta(X^{(i)})$ and the actual expected answer $Y^{(i)}$ i.e. $h_\theta(X^{(i)})-Y^{(i)}$ (this single error value is called the loss). 
The sum of errors for all examples would be: $\displaystyle \sum_{i=1}^{m} (h_\theta(X^{(i)})-Y^{(i)})$ (this sum of all losses is the total cost)

Also note, that is the above is the sum of *signed* errors. Since they may have +ve and -ve values, they will cancel out. The final cost function is the one given below. You will notice that the term has been squared. One of the reasons is to prevent the errors from cancelling each other out.

The final cost function for a particular feature, denoted by $J(\boldsymbol{\theta})$, can be written as:
$$
J(\boldsymbol{\theta}) = 
\underbrace {1/2}_{
\begin{array}
\text{normalization} \\
\text{(cancels out while}\\
\text{taking a derivative)} 
\end{array}
}
\displaystyle \sum_{i=1}^{m} 
\overbrace{
(h_\theta(X^{(i)})-Y^{(i)})^2
}^{
\begin{array}
\text{} \text{squared due to conventional}\\
\text{and historial reasons}
\end{array}
}
$$
The cost function above is known as the "least squares cost function".

*Note: The cost function $J(\boldsymbol{\theta})$ measures the overall error of our model using the current set of all parameters $\boldsymbol{\theta}$. When we optimize, we'll examine how this total cost changes if we adjust a single parameter $\theta_j$. Each $\theta_j$ is a 'weight' associated with a feature $x_j$ (or the bias term $x_0$).*

Now that we have our cost function, how can we use it to determine what is a good hypothesis? Well, we must find the set of values for our parameter/efature vector $\boldsymbol{\theta}$ (i.e. for all $\theta_0,\theta_1,\cdots,\theta_d$) that minimizes the cost function $J(\boldsymbol{\theta})$. Therefore, we must solve for $min(J(\boldsymbol{\theta}))$.  

This action of trying to minimize the cost i.e. $min(J(\theta_j))$ is called 'optimization'. And one of the most popular optimization algorithms is gradient descent, which we shall see next. 

*Note: For some complex, non-linear functions, a minimum might not exist.*


## Conclusion:

We have gone through some of the most commonly used notation in ML and looked at important concepts like the cost function. In the next part of this chapter, we shall continue this conversation and look at one of the most powerful optimization algorithms in use: Gradient Descent.