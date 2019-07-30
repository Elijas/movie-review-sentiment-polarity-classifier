# Description
Movie reviews are classified into positive and negative by training two different classifiers (Naive Bayes and Logistic Regression). Solution is provided in Python programming language and the best classifier has achieved TODO test accuracy.

# Q&A
#### 1. Describe text processing pipeline you have selected.
TODO
#### 2. Why you have selected these two classification methods?
In the standard workflow of implementing ML, it is accepted to abide by the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle), especially at the start. This means training the simpler models (e.g. the Naive Bayes or the Logistic Regression) first. The reason for this is to avoid premature optimization, since the knowledge to make measured decisions is not available yet. For example, the corpus has only ~10k sentence samples and introducing a complex algorithm right away is likely to result in a predictive model with a [high variance](https://en.wikipedia.org/wiki/Bias–variance_tradeoff). Therefore, Naive Bayes and Logistic Regression algorithms are suitable choices here.

#### 3. Compare selected classification methods. Which one is better? Why?
TODO
#### 4. How would you compare selected classification methods if the dataset was imbalanced?
If the frequency of label samples in the dataset were imbalaced, then I would have to use a performance metric that is capable of handling such situation. A basic accepted approach is to take [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) metrics (two ratios of True Positive predictions for each label). If it were to be appropriate to give equal importance to the two, then they would be combined into a one score by using a harmonic mean (i.e. the [F1-score](https://en.wikipedia.org/wiki/F1_score)). This would constitute a proper handling of an imbalanced dataset.

# Dataset
[sentence polarity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (includes sentence polarity dataset README v1.0): 5331 positive and 5331 negative processed sentences / snippets. Introduced in Pang/Lee ACL 2005. Released July 2005.
