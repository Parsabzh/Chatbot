# load libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
# add dialogue and utterance to new dataframe
dt['dialogue'] = label
dt['uttr'] = text


# return the most frequent class as baseline
def baseline_classification():
    baseline = dt['dialogue'].mode()[0]
    print(baseline)
    return baseline


# classify an utterance based on simple keywords
def baseline_classification2(utterance):
    keywords = {
        'thankyou': ['thank', 'thank', 'thanks', 'thank', 'thanks', 'you'],
        'ack': ['okay', 'ok', 'fine', 'great', 'cool'],
        'affirm': ['yes', 'yeah', 'yep', 'yup', 'sure', 'of course', 'right', 'correct', 'indeed', 'absolutely'],
        'bye': ['goodbye', 'bye', 'see you', 'later', 'soon'],
        'confirm': ['is', 'it'],
        'deny': ['dont', 'do not'],
        'hello': ['hi', 'hello', 'hey', 'greetings', 'howdy'],
        'inform': ['looking', 'for', 'want', 'need', 'searching', 'care'],
        'negate': ['no', 'nope', 'nah'],
        'null': ['cough', 'hm', 'unintelligble'],
        'repeat': ['can', 'repeat'],
        'reqalts': ['how', 'about', 'other', 'options', 'else', 'another', 'alternatives'],
        'reqmore': ['more'],
        'request': ['what', 'where', 'when', 'who', 'which', 'how', 'why', 'whats', 'wheres', 'whens', 'whos', 'whens',
                    'hows', 'whys', 'phone', 'number', 'price', 'range', 'address', 'can', 'type', 'code', 'post'],
        'restart': ['start', 'over', 'restart'],
    }
    counts = {}
    for word in utterance.split():
        for key in keywords:
            if word in keywords[key]:
                counts[key] = counts.get(key, 0) + 1

    if counts != {}:
        return max(counts, key=counts.get)
    else:
        return baseline_classification()


# prepare trainset and testset
x_train, x_test, y_train, y_test = train_test_split(dt['dialogue'], dt['uttr'], test_size=0.15, random_state=0)


# define model
# lr = LogisticRegression()
# # fit the model
# lr.fit(x_train, y_train)
# # make a prediction
# predictions = lr.predict(x_test)
# score = lr.score(x_test, y_test)
# print(score)

def model_testing(x, y, model):
    count = 0
    correct = 0
    length = len(x)
    for i in range(length):
        count = count + 1
        utterance = y[i]
        classification = model(utterance)
        print(classification)
        dialogue = x[i]
        if classification == dialogue:
            correct = correct + 1
        else:
            print(utterance + " should be " + dialogue)
    print(correct / count)


model_testing(dt['dialogue'], dt['uttr'], baseline_classification2)