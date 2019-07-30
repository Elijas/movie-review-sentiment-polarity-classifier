from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier_performance(classifier, dataset, title="Results"):
    test_accuracy = classifier.score(dataset.test.x, dataset.test.y)
    y_actual = dataset.test.y
    y_predicted = classifier.predict(dataset.test.x)

    print(f"====== {title} ======")
    print(f"Test set accuracy:\n{100*test_accuracy:.1f}%")

    print(f"\n=== Performance evaluation ===")
    print(classification_report(y_actual, y_predicted, digits=3))

    print(f"\n=== Confusion matrix ===")
    cm = confusion_matrix(y_actual, y_predicted, labels=[1, 0])
    print(f"True Positives: {cm[0][0]}")
    print(f"True Negatives: {cm[1][1]}")
    print(f"False Positives: {cm[1][0]}")
    print(f"False Negatives: {cm[0][1]}")

    print(f"\n=== Training parameters ===")
    print(classifier.best_params_)