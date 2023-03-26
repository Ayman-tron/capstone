import csv
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python fault_detect.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        data = []
        for row in reader:
            Acceleration_x = float(row[1])
            Acceleration_y = float(row[1])
            Acceleration_z = float(row[1])

            data.append({
                "evidence": [Acceleration_x, Acceleration_y, Acceleration_z],
                "label":  Healthy
            })

    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]

    return evidence, labels


def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    # Compute how well we performed
    # return (labels, predictions)
    positive_labels = labels.count(1)
    negative_labels = labels.count(0)
    print(positive_labels, negative_labels)
    total = len(predictions)
    correct = 0
    incorrect = 0
    for i in range(total):
        if predictions[i] == labels[i]:
            if predictions[i] == 1:
                correct += 1
            else:
                incorrect += 1

    sensitivity = float((correct / positive_labels))
    specificity = float((incorrect / negative_labels))
    #print("!!!!!!!", sensitivity)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
