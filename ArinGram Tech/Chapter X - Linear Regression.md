This one is going to be kind of intense. We will be setting the stage for all future machine learning articles by exploring basic notations and technical terminologies. Being the simplest model (I’ll define what a model is in a bit), we will use linear regression as an example to explore the basics of supervised learning in general, and regression in particular.

## Terminology:

Let us first define a few terms that we will be repeatedly using. We shall also look at some abstract ideas in this section. The abstraction is important since it helps us generalize these ideas across different areas.

### The Hypotheses:

I want to begin by defining the hypotheses. You may remember that our job here in regression, is to make a prediction about something. The word prediction itself implies that it may or may not be entirely true. These predictions are made with the help of functions. If you read the introductory articles, I referred to functions as formulae/algorithms. These functions are used to make predictions. Since the predictions may or may not be accurate, we refer to functions as ‘hypotheses’. How correct our hypothesis is, will depend on many factors which we shall see later.

A hypothesis can be represented as:

![pic11-sup1.png](attachment:65a09820-d769-4919-ad40-df70ffc1d44b:pic11-sup1.png)

Where, h = our hypotheses/function (whatever you want to call it) X = set of input (e.g. a set of cat and dog images) ŷ = output/prediction (e.g. whether each image in the input set is that of a dog or a cat)

We are given a training set which can be represented as:

![image.png](attachment:4a3e0257-ea02-48ac-b282-75cd19ddd85f:image.png)

Where:

![image.png](attachment:e57805b1-6ea6-4b3f-ac71-8d47d9be99f0:image.png)

![image.png](attachment:983fc1b3-43c2-4fe0-af5d-3678972e3306:image.png)

Given the above data, our question is: how do we find a good h? The answer to this question depends on our definition of ‘good’. For this article and for most purposes, i will define ‘good’ to be the accuracy of ‘h’. More the accuracy, better the hypothesis.

_Side Note:_ We aren’t really interested in successfully interpolating known data points. We are more interested in how well it performs on a new pair