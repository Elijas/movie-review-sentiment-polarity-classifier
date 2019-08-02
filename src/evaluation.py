"""
Functions related to evaluating classifier performance
"""
from datetime import datetime

import sklearn.metrics

from src.io import Dataset


def get_classification_report(classifier, dataset: Dataset) -> str:
    """Build a text report showing the main classification metrics"""
    test_accuracy = classifier.score(dataset.tst.x, dataset.tst.y)
    predicted_y = classifier.predict(dataset.tst.x)
    classification_report = sklearn.metrics.classification_report(dataset.tst.y, predicted_y, digits=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(dataset.tst.y, predicted_y, labels=[1, 0])

    return f"""=== Test set accuracy ===
{100 * test_accuracy:.1f}%
    
=== Performance evaluation ===
{classification_report}

=== Confusion matrix ===
True Positives: {confusion_matrix[0][0]}
True Negatives: {confusion_matrix[1][1]}
False Positives: {confusion_matrix[1][0]}
False Negatives: {confusion_matrix[0][1]}

=== Training result ===
Best score: {classifier.best_score_}
Best parameters: {classifier.best_params_} 
Best estimator: {repr(classifier.best_estimator_)} 

=== Repr of Classifier ===
{repr(classifier)}

=== Meta ===
Report generated at (UTC): {str(datetime.utcnow())}"""
