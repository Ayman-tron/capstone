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
import convertToTrainingData as convert
from time import sleep


TEST_SIZE = 0.2


class fault_detect():
    def __init__(self):
        self.path_trainData = 'fault_detect_algorithm\\ML_trainingTest.csv'
        # Check command-line arguments
        try:
            # Load data from terminal input 2nd argument spreadsheet and split into input and output data
            self.evidence, self.labels = self.load_training_data(sys.argv[1])
        except:
            # Load data from path spreadsheet and split into input and output data
            self.evidence, self.labels = self.load_training_data(
                self.path_trainData)

        # obtain test data from training data
        _, self.X_test, _, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=TEST_SIZE
        )

        # Train model
        self.model = self.train_model(self.evidence, self.labels)

        # Setup Firebase interface
        cred = credentials.Certificate(
            "C:\\Projects\\digital twin\\capstone\\fault_detect_algorithm\\live-b1071-firebase-adminsdk-m1rco-f2f91025a2.json")
        firebase_admin.initialize_app(cred)

        self.db = firestore.client()

    def main(self):

        # Test single sample from training data
        while 1:
            test_id = int(4)
            x_test = self.load_test_data()
            prediction = self.predict_fault(x_test)
            print(self.labels[test_id])
            print(prediction)
            self.push_firebase(prediction[0])
            # update every 2 seconds. Can change
            sleep(2)

    def push_firebase(self, status):
        status = str(status)
        timestamp = datetime.now()
        doc_ref = self.db.collection(u'Flange_status').document()
        doc_ref.set({
            u'status': status, u'timestamp': timestamp
        })

    def load_training_data(self, filename):
        with open(filename) as f:
            # open the file to read
            reader = csv.reader(f)
            next(reader)

            data = []
            for row in reader:
                Peaks = [float(x) for x in row[1:-2]]
                Healthy = int(row[-1], 2)
                data.append({
                    "evidence": Peaks,
                    "label":  Healthy
                })
        evidence = [row["evidence"] for row in data]

        labels = [row["label"] for row in data]

        return evidence, labels

    def load_test_data(self):
        freqCnt = convert.calcDomFreqCnt()
        raw = convert.makeDataRow(convert.num - 1, freqCnt)
        evidence = [float(x) for x in raw[:-2]]
        return [evidence]

    def predict_fault(self, inputs):
        try:
            prediction = self.model.predict(inputs)
        except ValueError:
            # Do this if input is a single sample
            inputs = inputs.reshape(1, -1)
            prediction = self.model.predict(inputs)

        # Convert predictions into 10-bit binary
        prediction = [bin(x).replace("0b", "").zfill(11) for x in prediction]

        return prediction

    def train_model(self, evidence, labels):
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(evidence, labels)
        return model

    def evaluate(self, labels, predictions):
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
    fault_detect().main()
