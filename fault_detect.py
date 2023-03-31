import csv
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy import random
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# TEST_SIZE = 0.9


def main():
    cred = credentials.Certificate(
       "live-b1071-firebase-adminsdk-m1rco-f2f91025a2.json")
    firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python fault_detect.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    # X_train, X_test, y_train, y_test = train_test_split(
    #     evidence, labels, test_size=TEST_SIZE
    # )

    # Train model and make predictions
    model = train_model(evidence, labels)

    test_id = int(random.rand() * len(evidence))
    x_test = np.array(evidence[test_id]).reshape(1, -1)
    print(labels[test_id])
    print(model.predict(x_test)[0])
    y_test = bin(model.predict(x_test)[0]).replace("0b", "")
    print(y_test)
    y_test = y_test.rjust(11,'0')
    print(y_test)

    status = str(y_test)
    timestamp = datetime.now()
    doc_ref = db.collection(u'Flange_status').document()
    doc_ref.set({
        u'status': status,u'timestamp': timestamp
    })

    ### uncomment to enable evaluate function
    # sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    # print(f"Correct: {(y_test == predictions).sum()}")
    # print(f"Incorrect: {(y_test != predictions).sum()}")
    # print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    # print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    with open(filename) as f:
        # open the file to read
        reader = csv.reader(f)
        next(reader)
        
        data = []
        for row in reader:
            Peaks = [float(x) for x in row[1:-2]]
            Healthy = int(row[-1],2)
            data.append({
                "evidence": Peaks,
                "label":  Healthy
            })

    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]

    return evidence, labels


def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    # Compute how well we performed
    # return (labels, predictions)
    positive_labels = labels.count(1)
    negative_labels = labels.count(0)
    # print(positive_labels, negative_labels)
    total = len(predictions)
    correct_pos = 0
    correct_neg = 0
    incorrect = 0
    for i in range(total):
        # print(predictions[i], labels[i])
        if predictions[i] == labels[i]:
            if labels[i] == 1:
                correct_pos += 1
            else:
                correct_neg += 1

        else:
            incorrect += 1

    sensitivity = float((correct_pos / positive_labels))
    specificity = float((correct_neg / negative_labels))
    #print("!!!!!!!", sensitivity)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
