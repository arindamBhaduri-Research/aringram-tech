In the previous section of this chapter, was were introduced to probabilistic models. We also saw why probabilistic  definitions of models are preferred over deterministic ones. Any questions you may have about probabilistic models will hopefully be answered in this chapter. We shall see, in detail, the probabilistic definitions of linear and logistic regression.

## Notation:

## Linear Regression:

A deterministic definition of linear regression would be that this algorithm 'fits a line' to the data. The line will give us a single fixed value for constant value of input $\mathbf{x}$ and parameters $\boldsymbol{\theta}$. Let us see how we can define this in a probabilistic manner.

If you cast your mind back to chapter 2, you will recall that our dataset looked somewhat like:

$$\{(\mathbf{x}^{(i)},y^{(i)})\quad \forall \quad i=1 \cdots m\}$$
where:
$$\mathbf{x}^{(i)} \in \mathbb{R}^{d+1}, \ y^{(i)} \in \mathbb{R}$$
And we needed to find:
$$\begin{flalign*}
& \boldsymbol{\theta} \in \mathbb{R}^{d+1}&& \\
& \text{such that } \boldsymbol{\theta} = argmin_\theta \displaystyle \sum_{i=1}^m(y^{(i)}-h_\theta(\mathbf{x}^{(i)}))^2 && \\
& \text{where} \quad h_\theta(\mathbf{x}) = \boldsymbol{\theta}^T.\mathbf{x} && \\
\end{flalign*}$$
Here, the superscript T on $\theta$ mean transpose of the matrix.

The question that now arises, is that why do we need to minimize the above sum of squares? Let us look at the answer:

Assume $\boldsymbol{\theta}_*$ to be the **actual** set of parameters for which $y^{(i)} = \boldsymbol{\theta}_*^T.\mathbf{x}^{(i)}$ exactly. But in practice, we do not know what this $\boldsymbol{\theta}_*$ is. Therefore we must try and predict those. Since our algorithm cannot be perfect each time, our predictions must vary from the actual answer by a little bit. That is to say, there might me very small error terms of residuals still left even after we optimize for $\boldsymbol{\theta}$. We represent that error term (also called noise) as: $\in^{(i)}$.

Therefore the final equation can be written as:
$$y^{(i)} = \boldsymbol{\theta}_*^T.\mathbf{x}^{(i)} + \in^{(i)}$$
### Properties of $\in^{(i)}$:

The error term $\in^{(i)}$ has some very interesting and important properties. some of these properties will influence many of our later decisions about choosing the right cost functions. let us take a look at them:

1. The value of $\in^{(i)}$ is different for every tuple i.e. it is different for every input example.
2. The average over all noise terms, written as $\mathbb{E}[\in^{(i)}]=0$ i.e.  the errors are unbiased and do not vary too much.
3. The errors are statistically independent i.e. 
   $\mathbb{E}[\in^{(i)} \in^{(j)}]=\mathbb{E}[\in^{(i)}]\mathbb{E}[\in^{(j)}] \text{ for } i\neq j$.

We can tell how noisy our data (or our prediction) is, by calculating the variance of the noise. The formula for calculating that is:
$$\mathbb{E}[(\in^{(i)})^2]=\sigma^2$$

We saw above that the average over all noise terms i.e. their mean $\mu=0$. We also know the variance from the above equation. Using these two metrics, we can draw a Gaussian or Normal distribution over these metrics:

![[c3b-graph1.excalidraw]]

The above graph can be expressed by an equation $\in^{(i)} \ \sim \ N(\mu,\sigma^2)$ where $\sim$ stands for "drawn distributed to".

**Note:** The distribution is **unique** for every error term.

**Why a Gaussian Distribution?**
At first, you might feel confused about this decision to draw a Gaussian distribution for the error terms. There are four main reasons for this:

1. **Convenience:** 
   - We need to derive a cost-function that is differentiable. 
   - The Gaussian distribution is "analytically tractable" i.e. it allows the likelihood to have a **closed-form expression**.
2. **Central Limit Theorem:**
   - CLT says that if we take a large number of independent random variables, their sum or average will tend to follow a Gaussian distribution as the number of variables increases.
   - It means that a Gaussian distribution over our error terms can be assumed to be fairly realistic.
3. **Maximum Entropy Distribution:**
   - Gaussian distribution has the maximum entropy among all continuous distributions; it assumes the very less beyond the given constraints.
   - Therefore assuming that our errors come from a Gaussian distribution, is the **most unbiased** assumption we can make.

In general, we can mathematically represent a Gaussian distribution using the following formula:

$$\begin{flalign*}
& P(Z\ ;\ \mu,\sigma^2)=\underbrace{\frac{1}{\sigma\sqrt{2\pi}}}_{\text{normalization constant}}
\end{flalign*}$$