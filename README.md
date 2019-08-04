
# Description
Movie reviews are classified into positive and negative by training two different classifiers (Naive Bayes and Logistic Regression). Solution is provided in Python programming language and the best classifier has achieved 79% test accuracy.

# Q&A
#### 1. Describe text processing pipeline you have selected.
The selected text processing pipeline starts by transforming each text line into a corresponding token vector. Stop-words (pronouns, etc.) and other tokens such as symbols are filtered out, and the remaining words are stemmed and lower-cased. Then a certain set of tokens is selected as the Features for the model (i.e. Bag-of-words approach) as unigrams, bigrams, or trigrams (an ordered set of words out of 1, 2, or 3 elements, correspondingly). Certain features are frequent for both Positive and Negative reviews, therefore are not that useful. This is why we will incorporate TD-IDF into our pipeline (Frequency-Inverse Document Frequency). Then each input token vector (text line) is converted into a sparse binary vector which signifies presence or absence of that particular feature in the input line. Finally, we pass our input vector to our training algorithm along with the labels to start the supervised learning. We train the same algorithm multiple times with different text processing and training hyperparameters, picking the best algorithm by using n-fold cross-validation.
#### 2. Why you have selected these two classification methods?
In the standard workflow of implementing ML, it is accepted to abide by the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle), especially at the start. This means training the simpler models (e.g. the Naive Bayes or a Linear model such as the Logistic Regression) first. The reason for this is to avoid premature optimization, since the knowledge to make measured decisions is not available yet. For example, the corpus has only ~10k sentence samples and introducing a complex algorithm right away is likely to result in a predictive model with a [high variance](https://en.wikipedia.org/wiki/Bias–variance_tradeoff). Therefore, Naive Bayes and Logistic Regression algorithms are suitable choices here.
#### 3. Compare selected classification methods. Which one is better? Why?
Both algorithms have achieved very similar cross-validation scores, so we can consider them equivalent by performance, although Naive Bayes was roughly 2x faster to train.
#### 4. How would you compare selected classification methods if the dataset was imbalanced?
If the frequency of label samples in the dataset were imbalaced, then I would have to use a performance metric that is capable of handling such situation. A basic accepted approach is to take [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) metrics (two ratios of True Positive predictions for each label). If it were to be appropriate to give equal importance to the two, then they would be combined into a one score by using a harmonic mean (i.e. the [F1-score](https://en.wikipedia.org/wiki/F1_score)). This would constitute a proper handling of an imbalanced dataset.

# Project Structure and Instructions

### Training instructions
Run the following commands in shell:
1. `pip install -r requirements.txt` to install the dependencies.
2. `python main.py --dry-run` to test the configuration (Optional. Note: it will overwrite the model to test file I/O).
3. `python main.py` to train the model and see the evaluation results

### Pre-trained models
You can find them in [a separate repository](https://github.com/Elijas/movie-review-sentiment-polarity-classifier-model). 

# Dataset
[sentence polarity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (includes sentence polarity dataset README v1.0): 5331 positive and 5331 negative processed sentences / snippets. Introduced in Pang/Lee ACL 2005. Released July 2005.

### Other classifiers trained on this dataset
- [cmasch/cnn-text-classification](https://github.com/cmasch/cnn-text-classification)
