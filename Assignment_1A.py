# load libraries
import os 
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from baselines import baseline_classification2
from NN1 import NeuralNet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from lr import create_model as create_lr_model

# read dataset as dataframe
df = pd.read_table("Data/dialog_acts.dat", index_col=False, names=["words"])
label = []
text = []
# choose the column
line_list = df.iloc[:, 0]
# create new dataframe to create new dataset for training
dt = pd.DataFrame(columns=['dialogue', 'uttr'])
# extract the first word from string
for line in line_list:
    first_word = line.split()[0]
    uttr = line.replace("{} ".format(first_word), '')
    label.append(first_word.lower())
    text.append(uttr.lower())

# prepare trainset and testset
x_train, x_test, y_train, y_test = train_test_split(text, label, test_size=0.15, random_state=0)

nn = NeuralNet()

# run a model using utterances and dialogue and return the predicted dialogue class for the utterance
def run_features(utterances, dialogue, model):
    predictions = []
    i = 0
    for utterance in utterances:
        classification = model(utterance)
        predictions.append(classification)
        i += 1

    return predictions


# print the accuracy, precision, recall, and f1 score for the predicted dialogue classes and plot a confusion matrix
def calculate_metrics(predictions, dialogue):
    disp = ConfusionMatrixDisplay.from_predictions(dialogue, predictions, xticks_rotation='vertical')
    acc = accuracy_score(dialogue, predictions)
    recall = recall_score(dialogue, predictions, average='weighted')
    precision = precision_score(dialogue, predictions, average='weighted')
    f1 = f1_score(dialogue, predictions, average='weighted')
    print("Accuracy: " + str(acc) + " ,F1: " + str(f1), " ,Recall: " + str(recall), " ,Precision: " + str(precision))
    plt.show()
    print(disp)


baseline_predictions = run_features(x_test, y_test, baseline_classification2)
calculate_metrics(baseline_predictions, y_test)


nn_predictions = run_features(x_test, y_test, nn.predict)
calculate_metrics(nn_predictions, y_test)