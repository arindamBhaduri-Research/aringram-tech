> "Machine Learning is the field of study that gives the computer the ability to learn without being explicitly programmed to do so."
> - Arthur Samuel (1959)

I'm sure all of you love nice cars. What drives it? Its engine. What drives the computer you watch your favorite Netflix shows on? A combination of the computer's motherboard and operating system. Similarly, all the AI tools that you see around you - ChatGPT for example - is driven by machine learning.

## Machine Learning? What's that?

What is machine learning? This question does not have an easy answer. A machine learning model (we will see what a model is later on), can be defined in different ways depending on the level of abstraction we look at it with. But in a generalized sense, machine learning is a set of mathematical formulae that help an AI application predict a number, classify an image as that of a cat or a dog, generate answers to your questions, etc. These formulae sometimes come together to form an algorithm which performs more complex tasks than a single formula could. These formulae/algorithms can either be as easy as the equation of a straight line that you learnt in school (y = mx + c) or something far more difficult that involves complex ideas and computations. These formulae/algorithms tell a computer how to learn something from data that is provided to it. We will look at some basic algorithms in the upcoming articles.

*Side Note: For people who already know about machine learning, it may seem weird for me to constantly refer to models as formulae/algorithms. I will use the current terminology for this article. In the next article about Linear Regression, I will formally define what a model is and then switch to using that term instead.*

## Types of Machine Learning

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks T, as measured by P, improves with experience E."
> - Tom Mitchell (1998)

In the quote above:
- Experience E -> our data that we train the machine on
- Performance measure -> our algorithm's accuracy/win rate, etc.
- Improvement with E: more training data -> better performance (this is a general rule but may not be true always. We will discuss more on this in later articles.)

I mentioned in the last section that the algorithms help computers to learn from data provided to it. In other words, a computer is 'trained' to do something using some data that we feed into it. In machine learning, this data is called 'training data'. Training data is just as important as the algorithms themselves. It is crucial that we clean and analyze the data ourselves before feeding it into the computer as training data. There is an entire branch of computer science called 'Data Science' that deals with data handling and inference. Data science is out of scope for this blog. However, we do need to learn some data science basics in order to successfully train our machine learning algorithms. I will be creating another series on this. 

Coming back to the topic, there are different ways an algorithm can learn from data. Based on what kind of data is fed to an algorithm, how the algorithm learns from it, and most importantly, what tasks can the algorithm perform after the training is done, machine learning is broadly divided into four categories:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning
4. Recommender Systems (technically not a category on its own but is sometimes studied separately since it has high commercial value.)

It is not necessary for the tasks performed by the different categories to be mutually exclusive. The three main tasks/categories can be better represented by the Venn diagram below. In a lot of real-world applications, we use a combination of these techniques.

### Supervised Learning
In supervised learning, our training data consists of two parts: 
1. Features 
2. Ground truth labels
Let us look at an example to understand the two parts. Say we want to predict the prices of houses in a certain area. The dataset contains many rows and many columns. Each row represents one particular house in that area. The columns represent the features of the house i.e. the square footage, number of bedrooms, number of bathrooms, neighborhood etc. These are the 'features' of the training examples (the houses themselves). The last column will actually tell us what the price of a particular house really is. These values are called the 'ground truth labels' because they are the actual output that is expected.

So we not only give the algorithm the features of the house, but also tell it what output is expected for different combinations of features. Therefore, when we input the features of an actual house whose price we want to predict, the algorithm will output a prediction of what the price of that house could be.

Sometimes we may have many features for an item in a dataset, which is called as 'high-dimensional features'. We usually don't require to use all features, since by using all of them we might run the risk of overfitting. 

If you have noticed, we use supervised learning whenever we expect an output from the computer. Supervised learning algorithms are the most widely used algorithms in the world today - roughly 90-95% of them. So as you can see, these algorithms play a huge role in the AI developments we see around us.

Supervised learning algorithms are further divided into two types, based on the kind of tasks they perform:
### Unsupervised Learning
Unlike supervised learning, the data in unsupervised learning does not contain the 'ground truth labels' i.e. we do not tell the computer what the correct answer for various examples is. 

Let us say you operate a large department store and you have a dataset of customer behavior. The dataset has various features like:
- Age
- Annual Income
- Average Spending
- Frequency of Visits
- Kind of Products Purchased, etc.

Now, you want to analyze the customer behavior to group together certain types of customers that tend to have similar shopping habits so you can cater to everyone better.

Using an unsupervised learning algorithm like K-Means Clustering, you can automatically group customers based on their behavior. The algorithm will analyze the data and try to identify clusters of customers who exhibit similar behavior, even though you didn’t tell it what groups to look for. For example, it may group together:
- Teenagers and classify them as low-income, low-spending, usually buying snacks, magazines, etc. 
- Senior citizens as high-income, high-spending, usually buying medical supplies or groceries...

As you can see, we aren't expecting the computer to 'predict' an actual 'output'. We just want it to analyze the data and tell us what it makes of it. Unsupervised learning algorithms are widely used in cyber-security where they find anomalies in data to predict whether an email is spam, a transaction request is fraudulent, etc. 

### Reinforcement Learning
Reinforcement learning, in my opinion, is one of the most fascinating techniques used to train machines. I can best explain it using an example.

Let us say you sent a rover to the moon. The rover has landed on the moon. To its right, maybe 10 meters away, there is a huge crater and if the rover moves towards it, it might fall in and crash. Therefore, moving to the right is an undesirable action. To its left, at the same distance, is a flag which is the rover's final destination. If it moves to the left and reaches the flag, your mission is completed. Therefore, moving to the left is a desirable action. 

The rover doesn't know which way to go. So it randomly chooses a direction. If it chooses to move one step to the left, it is a desirable action. We will 'reward' the rover for it. However, if it moves one step to the right, it is undesirable, and we will penalize it. After a few such iterations, the rover will see a pattern and realize that it gets rewarded when it moves to the left and gets penalized if it moves to the right. Therefore the randomness of its decision will slowly decreased and will be biased towards moving left. And finally, it will continue moving left until it reaches its destination.

In simple words, reinforcement learning allows the algorithm to interact with its surroundings. Depending on how it interacts, we either reward or penalize it, similar to when when try to train a dog. If the dog sits at our command we give it a treat, but if it misbehaves, it does not get the treat. We will look into this method technically and in detail later on.

### Recommender Systems
Recommender systems are traditionally not a separate discipline in machine learning. But since they have huge commercial impact, it is important to know more about them. In my blog, apart from this very brief introduction to recommender systems, I will not talk about them later. 

Recommender systems are used by e-commerce stores and your favorite online streaming platforms to suggest you products, videos or movies that you would like to buy/watch. Amazon, YouTube and Netflix are the best examples of companies that use these algorithms to personalize your feed and potentially reduce the time spent and hassle in searching for the right product or movie. 

There are two main types of recommender systems:
1. **Collaborative Filtering:** These algorithms recommend videos or movies to you based on what *other* people similar to you are watching. These other people may include your friends/family, but more likely they include people who have similar watching patterns as you or who watch similar movies as you. 
2. **Content-based Filtering:** These algorithms recommend you new movies/videos based on similarity the *contents* of the previous movies you watched or liked. For example, if you have watched a lot of action movies in the past, it may recommend you similar movies in the future. 

If you want to learn more about them, here are a few free resources to get you started (none of these are promotional and have been selected through a simple Google search and personal experience):
1. [Recommender Systems Specialization on Coursera](https://www.coursera.org/specializations/recommender-systems)
2. [Building Recommender Systems with Machine Learning and AI on Coursera](https://www.coursera.org/learn/packt-building-recommender-systems-with-machine-learning-and-ai-7kdhj)
3. [Movie Recommendation System: End-to-End Project](https://www.youtube.com/watch?v=1xtrIEwY_zY&t=6624s)
4. [Book Recommendation System: End-to-End Project](https://www.youtube.com/watch?v=1YoD0fg3_EM)


### Conclusion:
I hope you got a fairly good idea about what machine learning really is and how it actually works. In the upcoming articles, we will be delving deeper into all these concepts. I will select some of the most crucial machine learning algorithms one must know and will talk about both the mathematical foundations of he algorithm as well as implement it in code. The next article will be on one of the first algorithms a student learns and also the simplest yet most profound one: Linear regression.