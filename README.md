
# Description
Movie reviews are classified into positive and negative by training two different classifiers ([Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) and [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)). Solution is provided in Python programming language and the best classifier has achieved [81% test accuracy](https://github.com/Elijas/movie-review-sentiment-polarity-classifier-model), which compares well with the other classifiers trained on the same dataset (links are available at the end of this README).

# Q&A
#### 1. Describe text processing pipeline you have selected.
The selected text processing pipeline starts by transforming each text line into a corresponding token vector. [Stop-words](https://en.wikipedia.org/wiki/Stop_words) (pronouns, etc.) and other tokens such as symbols are filtered out, and the remaining words are stemmed and lower-cased. Then a certain set of tokens is selected as the Features for the model (i.e. [Bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) approach) as [unigrams, bigrams, or trigrams](https://en.wikipedia.org/wiki/N-gram) (an ordered set of words out of 1, 2, or 3 elements, correspondingly). Certain features are frequent for both Positive and Negative reviews, therefore are not that useful. This is why we will incorporate TD-IDF into our pipeline ([Term Frequency-Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)). Then each input token vector (text line) is converted into a [sparse binary vector](https://en.wikipedia.org/wiki/Sparse_matrix) which signifies presence or absence of that particular feature in the input line. Finally, we pass our input vector to our training algorithm along with the labels to start the [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning). We train the same algorithm multiple times with different text processing and training [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter), picking the best algorithm by using [n-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).
#### 2. Why you have selected these two classification methods?
In the standard workflow of implementing ML, it is accepted to abide by the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle), especially at the start. This means training the simpler models (e.g. the Naive Bayes or a Linear model such as the Logistic Regression) first. The reason for this is to avoid premature optimization, since the knowledge to make measured decisions is not available yet. For example, the corpus has only ~10k sentence samples and introducing a complex algorithm right away is likely to result in a predictive model with a [high variance](https://en.wikipedia.org/wiki/Bias–variance_tradeoff). Therefore, Naive Bayes and Logistic Regression algorithms are suitable choices here.
#### 3. Compare selected classification methods. Which one is better? Why?
Both algorithms have achieved [very similar cross-validation scores](https://github.com/Elijas/movie-review-sentiment-polarity-classifier-model), so we can consider them equivalent by performance, although Naive Bayes was faster to train.
#### 4. How would you compare selected classification methods if the dataset was imbalanced?
If the frequencies of label samples in the dataset were imbalaced, then I would have to use a performance metric that is capable of handling such situation. A basic accepted approach is to take [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) metrics (two ratios of True Positive predictions for each label). If it were to be appropriate to give equal importance to the two, then they would be combined into a one score by using a harmonic mean (i.e. the [F1-score](https://en.wikipedia.org/wiki/F1_score)). This would constitute a proper handling of an imbalanced dataset.

# Project Structure and Instructions

### Training instructions
Run the following commands in shell:
1. `pip install -r requirements.txt` to install the dependencies.
1. `python main.py --dry-run` to test the configuration (Optional) (Note: it will overwrite the previously trained saved model to test file I/O).
1. `python main.py` to train the model and see the evaluation results

### Pre-trained models
You can find them in [a separate repository](https://github.com/Elijas/movie-review-sentiment-polarity-classifier-model). 

### Caveats
1. If you change constants related to dataset shuffle/split (such as `RANDOMNESS_SEED`, `DATASET_TEST_SPLIT_RATIO`) then `data/raw_structured` folder has to be deleted (it will be recreated automatically) for the new constants to apply.

# Dataset
[sentence polarity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (includes sentence polarity dataset README v1.0): 5331 positive and 5331 negative processed sentences / snippets. Introduced in Pang/Lee ACL 2005. Released July 2005.

### Other classifiers trained on this dataset
- [cmasch/cnn-text-classification](https://github.com/cmasch/cnn-text-classification)
- [elijas/review_thingie](https://github.com/elijas/review_thingie) (fork from [ashirviskas/reviewthingie](https://github.com/ashirviskas/review_thingie), but with bugfix for the dataset test/train splitting)
