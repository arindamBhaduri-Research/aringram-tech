In the previous articles, we have gone over some of the core concepts of machine learning, namely cost function and gradient descent. Next, our objective will be to learn about two of the most foundational algorithms in ML: linear and logistic regression. But before we get to that, it is important for me to talk a little bit about deterministic and probabilistic models. We will discuss what they are, their differences, and why probabilistic interpretations or views of models are preferred. It is crucial in order to understand why certain functions are used. It will also give us an idea about how these models fundamentally operate.

## Deterministic Models:

In simple words, 'deterministic' means 'fixed'. For a fixed set of inputs, the model will always produce the same output. There is no randomness in the process. It does not matter whether you run the model once or a hundred times. If your input remains constant, so will your output.

**Mathematical Representation:**
In general, a deterministic model is simply a mathematical function. If $\mathbf{x}$ is our input and $\boldsymbol{\theta}$ are the model parameters, the prediction $\hat{y}$ is simply:

$$\hat{y}=f(\mathbf{x};\boldsymbol{\theta})$$
**Example:**
A purely deterministic view of linear regression is a good example. In its most basic form, when we try to "fit a line" $h_\theta(\mathbf{x})=\boldsymbol{\theta}^T \mathbf{x}$ to the data, the line itself is purely deterministic. For any given $\mathbf{x}$ and learned $\boldsymbol{\theta}$, the output $\hat{y}$ is a single fixed point on the given line. 

We use the Least Squares cost function to find the optimum values for $\boldsymbol{\theta}$. But its main aim is to minimize some error metric, like the Sum of Squared Errors. Once the model is 'trained', gives fixed prediction, without any idea of uncertainty about that prediction. Let me elaborate.

When we begin training our model, we get errors. A deterministic model views errors simply as discrepancies between the prediction and observed data. It then tries to minimize these errors. However, in more complex datasets, there may be a particular reason or source linked to the errors. The source or nature of these errors is not really modeled or taken into consideration during the prediction. 

Another caveat is the uncertainty. A model may predict the price of a house to be, say, $3,00,000. But it may be wrong. Just because we minimized errors on our input data, does not mean that our model will perform just as well on previously unseen data. Our model does not accommodate the possibility of being wrong. It does not say that "the price is $3,00,000 with a 95% chance of being between $2,80,000 and $3,20,000". In technical terms, we say that the model does not 'generalize' well.

## Probabilistic Models:

Unlike deterministic models, a probabilistic model aims to capture the uncertainty in the data and the prediction process. Instead of predicting a single estimate, it outputs a **probability distribution** over possible outcomes.

**Mathematical Representation:**

Mathematically, the model describes the probability of observing an output $y$ given an input $\mathbf{x}$ and parameters $\boldsymbol{\theta}$ as:

$$\mathbf{P}(y|\mathbf{x};\boldsymbol{\theta})$$ Here, the output $y$ is treated as a random variable whose probability distribution is determined by $\mathbf{x}$ and $\boldsymbol{\theta}$. 

Now like above, I could give you an example by illustrating the probabilistic definition of linear regression. But I believe I could do a much better job by explaining it on its own. in the next article, I will talk in detail about the probabilistic definitions of both linear and logistic regression.

But for now, let us summarize the differences between the two approaches below.

### Difference Between Deterministic and Probabilistic Models:


| **Feature**           | **Deterministic Model**                      | **Probabilistic model**                           |
| --------------------- | -------------------------------------------- | ------------------------------------------------- |
| **1. Primary Output** | Single value/class                           | Probability distribution over outcomes            |
| **2. Input-Output**   | $\hat{y}=f(\mathbf{x};\boldsymbol{\theta})$  | $$\mathbf{P}(y\|\mathbf{x};\boldsymbol{\theta})$$ |
| **3. Uncertainty**    | Not quantified                               | Explicitly quantified                             |
| **4. Noise/Errors**   | Minimized, but not factored into prediction. | Explicitly modeled                                |

## Why do we Prefer Probabilistic Models:

1. **Deciding the Cost Function:** 
   Another fancy term for the above is 'principled derivation of cost functions' and 'Maximum Likelihood Estimation'. But we shall worry about this in the next article. For now what this actually means, is that probabilistic models help us choose which cost function to use.
   
   Probabilistic models allow us to approach this problem of choosing a formula from a more fundamental angle. Instead of just picking a formula at random, we start by making an assumption about the *nature of the errors* in the data we are trying to model. Once we have such an assumption, a probabilistic model aims to find model parameters that make the actual data look as likely or plausible as possible, given those assumptions.
   
   The fascinating part is that the mathematical process of finding these "most plausible" model settings often naturally leads us to specific cost functions. For instance, certain common assumptions about data will directly point towards using "Sum of Squared Errors" (which you've seen) as the right way to measure error, while different assumptions will point towards other types of cost functions (like "Cross-Entropy," which is key for tasks like figuring out if an email is spam or not).
   
   So instead of cost functions being arbitrary, probabilistic models provide a **principled reason** for why certain functions might be more appropriate for certain tasks or data.

2. **Understanding our Model's Conviction:**
   
   We need to understand how sure or unsure our model is about its prediction. Probabilistic models do not give a single output. They give a probability distribution over all the possible outcomes. This helps us to quantify the uncertainty and helps us express how confident we are in a certain prediction.
   
   For regression tasks, the variance $\sigma^2$ gives us the prediction intervals. For classification, the output probabilities $\mathbf{P}(\mathbf{Y}=c_k|\mathbf{x})$ directly reflects the confidence.
   
3. **Foundation for More Advanced Techniques:**
   - **Bayesian Inference:** Probabilistic models are the foundation for Bayesian methods, where we can incorporate prior beliefs about parameters and obtain a full posterior distribution $P(\boldsymbol{\theta}|\mathbf{X,Y})$, not just point estimates. This allows for better uncertainty quantification. There is wonderful video (not sponsored) about the difference between Bayesian and Frequentist approaches. I highly recommend you to watch it: [https://www.youtube.com/watch?v=GEFxFVESQXc]()
   - **Generative Models:** Many generative models are inherently probabilistic.

4. **Interpretability of Assumptions:**
   By explicitly making our probabilistic assumptions (e.g. errors are Gaussian), we are clearer about the conditions under which our models are expected to perform well and how it might fail.
   
## Conclusion:

That was it for this blog. We explored some comparatively advanced ideas today. I understand that a good base of statistics is essential for fully understanding how probabilistic models work and the underlying intuition. If you haven't studied these statistical concepts previously, and have the means to do so, I would definitely suggest it. However, if you aren't as interested in all this, and just wish to learn how to implement these models in code, I would recommend you to skip to Chapter 5.

In the next article, we will going over in detail, the probabilistic definitions of linear and logistic regression. 