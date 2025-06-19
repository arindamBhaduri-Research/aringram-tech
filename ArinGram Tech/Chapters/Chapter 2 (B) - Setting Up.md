In the previous part of this chapter, we looked at the different notations used commonly in ML. We also saw how our dataset is represented in terms of vectors and also learnt about an important part of deciding the accuracy of our model - the cost function.

The cost function tells us how 'correct' or 'wrong' our model is. We then need to optimize our model parameters $\boldsymbol{\theta}$ so that we can minimize our cost. One of the most powerful and widely-used optimization algorithms is called Gradient Descent, which we shall see next.

### Gradient Descent:

Gradient descent is an optimization algorithm used to find the values of parameters (denoted as **$\boldsymbol{\theta}$**) that minimize a cost function $J(\boldsymbol{\theta})$. Let us imagine we have a hypothesis defined by a set of parameters $\boldsymbol{\theta}$. We also have a cost function $J(\boldsymbol{\theta})$ that tells us how 'good' or 'bad' our model is for a given set of parameters. If we plot this cost function, maybe for one or two parameters, it might look like a curve with many ups and downs. Our goal is to find the lowest point in on this curve. Note that the curve below DOES NOT belong to the least squares cost function defined earlier. It is for REPRESENTATION PURPOSES ONLY.

![[chapter2-graph2.excalidraw]]
As you can see above, we choose an arbitrary starting point on the curve $\theta^{(0)}$. From this point, gradient descent helps us determine the direction of the steepest downhill slope. We take one tiny step in that direction and reach a new point $\theta^{(1)}$, which should be closer to the minimum. If we take a step in the right direction, we get closer to the minimum, but if we take one in the wrong direction, we get further away from it.

But how will the computer know what is the right direction? And how do we decide the length of the step? For that, we have the gradient descent algorithm which looks something like this: 

$$ \begin{flalign*}
&\text{repeat until convergence:}\\&&
& \theta_j^{(t+1)} := \theta_j^{(t)} - \alpha \frac{\partial J(\theta^{(t)})}{\partial \theta_j} \quad j=0\cdots d &&\\
\end{flalign*}
$$

Here, $\alpha = \text{learning rate}$, which decides the size of the step. The term $\frac{\partial J(\theta^{(t)})}{\partial \theta_j}$, also called the derivative, gives us the slope of the curve at that point:
- If the slope is +ve, $\theta_j$ needs to decrease to lower the cost (hence the - sign in the update).
- If the slope is -ve, $\theta_j$ needs to increase.

If $\alpha$ is small, we will take a small step and vice versa. The value of $\alpha$ is important. If the value is too large, we will get something like this:
![[c2-g3.excalidraw]]
and we will fail to "converge" or reach the local minimum (in case the curve has multiple minima). If the step size if too small, it will take us a very long time to converge i.e. to find the optimum value for that $\theta_j$. With an optimum value of $\alpha$ we should get something like this:
![[c2-g4.excalidraw]]

*Note: The two convex or 'bowl-shaped' graphs you saw above at ones formed by the least squares cost function that we explored in the previous section, and one that we will use for reference in the upcoming sections of this blog. *

A good starting point is to set $\alpha=0.01$ and adjust it according to the results. We will discuss more on deciding the correct value of $\alpha$ in upcoming articles. 

One doubt you may have is that we decide our starting point arbitrarily. Therefore, if the curve of $J(\boldsymbol{\theta})$ has many local minima, we might end up at a local minimum which it not actually the lowest value possible for $J(\boldsymbol{\theta})$. This observation is true. However, in practice we usually repeat the gradient descent process a few times, changing the starting point each time. This usually ensures that are able to find a value for $\theta$ for which $J(\boldsymbol{\theta})$ is reasonably small.

**Note: For cost functions that are 'convex' (imagine a perfect bowl shape with a single global minimum, like the Sum of Squared Errors for linear regression), an interesting property emerges. As gradient descent approaches the optimal $\theta$ values, the magnitude of the gradient (the 'steepness' of the slope) naturally decreases. This means the updates become smaller, allowing for a fine-tuned convergence.  
For these convex functions, it's even possible to mathematically analyze the convergence rate of gradient descent, essentially proving how quickly it will get close to the minimum.  
While many advanced machine learning models, particularly deep neural networks, have highly complex, non-convex loss landscapes (with many local minima and saddle points), understanding convergence on convex functions provides foundational intuition. For those interested in the mathematical details of this convergence proof for convex functions, you can explore it further [here/in the linked section].**

Now it is time for some math. Let us look at the gradient descent step once again:

$\theta_j^{(t+1)} := \theta_j^{(t)} - \alpha \frac{\partial J(\theta^{(t)})}{\partial \theta_j}$

Let us expand and find the value of $\frac{\partial J(\theta^{(t)})}{\partial \theta_j}$. We have assumed $J(\boldsymbol{\theta})$ to be the least squares cost function described in the previous section. 
$$
\begin{flalign*}
&\frac{\partial J(\theta^{(t)})}{\partial \theta_j}
=  \frac{1}{2} \displaystyle \sum_{i=1}^m  \frac{\partial (h_\theta(x^{(i)}-y^{(i)})^2)}{\partial \theta_j}&&
\\
&= \sum_{i=1}^m
\underbrace{(h_\theta(x^{(i)})-y^{(i)})}_{\text{error}}
\frac{\partial (h_\theta(x^{(i)}))}{\partial \theta_j} && \\
& \text{now we know: } h_\theta(x) = \theta_0x_0+\theta_1x_1+\cdots+\theta_dx_d 
&& \\ \\
& \frac{d(h_\theta(x))}{d\theta}=x_j
&& \\\\
&\therefore \theta_j^{(t+1)} := \theta_j^{(t)} - \alpha \sum_{i=1}^m
(h_\theta(x^{(i)})-y^{(i)})(x_j^{(i)})
&& \\\\
& \text{The vectorized version* is as follows:}
&& \\
& \theta^{(t+1)} := \theta^{(t)} - \alpha \sum_{i=1}^m
(h_\theta(x^{(i)})-y^{(i)})(x^{(i)})


\end{flalign*}
$$

**What is a vectorized equation?**
We vectorize our equations since a vectorized equation is computationally cheaper as compared to a non-vectorized equation. Vectorization means performing operations on entire arrays (vectors or matrices) at once, rather than iterating element by element (e.g., with loops). Modern CPUs and libraries like NumPy are highly optimized for these array operations, making them much faster. For example, to get predictions for ALL training examples, instead of looping $m$ times to calculate $h_\theta(x^{(i)})$ for each $x^{(i)}$, we can compute $\boldsymbol{X\theta}$ in one matrix-vector multiplication.

The multiplication will look somewhat like this:

$$
\begin{flalign*}
& \boldsymbol{\theta} = 
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_d
\end{bmatrix}

\quad ; \quad

\mathbf{\mathbf{x}^{(1)}} = 
\begin{bmatrix}
x^{(1)}_0 \\
x^{(1)}_1 \\
\vdots \\
x^{(1)}_d
\end{bmatrix}
&& \\ \\
& {\mathbf{x}^{(1)T}}.\boldsymbol{\theta} = 
\begin{bmatrix}
x^{(1)}_0 \\
x^{(1)}_1 \\
\vdots \\
x^{(1)}_d
\end{bmatrix}
.
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\vdots \\
\theta_d
\end{bmatrix}
&&
\end{flalign*}
$$
As I mentioned above, this concept will become more clear when we look at the actual code implementation for a linear regression algorithm. 

### Batch vs. Stochastic Mini-Batch:

Let us look at our gradient descent update step once again:
$$\theta^{(t+1)} := \theta^{(t)} - \alpha \sum_{i=1}^m
(h_\theta(x^{(i)})-y^{(i)})(x^{(i)})$$
From the above equation we know that we have a total of $m$ training examples (since we are summing up to $m$). Also, let us assume that the number of features for our dataset is *d*. 

Batch gradient descent means that we use the entire training dataset to calculate the gradient of the cost function in each iteration. 

However for larger datasets, $m$ can be in the millions or even billions. Calculating the gradient term $\displaystyle \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})(y^{(i)})$ requires processing every single one of these $m$ examples *for each step of gradient descent*. This makes each update step computationally expensive and very slow.

To address this, we can use variations of gradient descent:

1. **Stochastic Gradient Descent (SGD):** Instead of all $m$    
     examples, the gradient is estimated using only one randomly selected training example per update step.
    
2. **Mini-Batch Gradient Descent:** This is a compromise. We use a small, random subset of the data, called a 'mini-batch', containing $b$ examples (where 1 $\leq$ $b$ << *m*) to estimate the gradient for each update."

In this method, we choose relatively small samples (mini-batches) from our training data. Each batch contains, say, $b$ data points, where $b$ << $m$. $b$ is also called the 'batch-size' of our batches. Stochastic is just a fancy term for 'random'. Therefore we either choose the training examples randomly, or we shuffle the order of the examples before selection.

The update rule for mini-batch gradient descent, using a mini-batch $\mathfrak{B}$ of size $b$ becomes:

$\boldsymbol{\theta}^{(t+1)}:=\boldsymbol{\theta}^{(t)}-\alpha\displaystyle \sum_{\mathbf{x^{(i)}},\boldsymbol{y^{(i)}} \in \mathfrak{B}} (h_\theta(\mathbf{x^{(i)}})-\mathbf{y^{(i)}})\mathbf{x^{(i)}}$

This method has 2 caveats:
- If $b$ is too small, the gradient estimate is very noisy. The path to the minimum will be much more erratic, though it will still generally head in the right direction over many updates. It can take longer to converge due to this noise.
- if $b$ is too large, the operation will become computationally expensive. 

Choosing a batch size $b$ is a hyper-parameter. Common values range from 32 to 512, often powers of 2, as these can be processed efficiently by hardware. The optimal batch size can depend on the dataset, model architecture, and available hardware resources.

Why does using only a small sample work? The key idea is that, on average, the gradient calculated from a random mini-batch is an unbiased estimator of the true gradient (the one calculated from the full dataset). While any single mini-batch gradient might be noisy and not point exactly towards the steepest descent, over many iterations, these noisy steps average out and guide the parameters towards a good solution. The inherent redundancy in large datasets also means that a small batch can often capture the essential characteristics of the overall data distribution.

## Conclusion:
Thank you for sticking with me! I know this chapter has been too long, but I have tried to ensure that my explanations are as simple and clear as possible. However, I am a beginner myself and there may be some areas of improvement. As I continue on my journey, I will revisit these articles and update any explanations that I feel could be better explained. In the next article, we will explore supervised learning in greater detail and maybe even look at some code. Happy learning!