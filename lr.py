# load libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

#read dataset as dataframe
def create_dataset():
    df = pd.read_table("Data/dialog_acts.dat",index_col=False,names=["words"])
    label=[]
    text=[]
    #choose the column
    line_list=df.iloc[:,0]
    #create new dataframe to create new dataset for training 
    dt=pd.DataFrame(columns=['dialogue','uttr'])
    #extract the first word from string
    for line in line_list:
        first_word= line.split()[0]
        uttr= line.replace("{} ".format(first_word),'') 
        label.append(first_word.lower())
        text.append(uttr.lower())
    
    #add dialogue and utterance to new dataframe
    dt['dialogue']=label
    dt['uttr']=text
    return dt
def vectorize(dt):
    #vectorize feactures and labels to change the text to the number
    dt['dialogue_act_id'] = dt['dialogue'].factorize()[0]
    vectorizer = CountVectorizer(min_df=5, encoding='latin-1', ngram_range=(1, 5), stop_words='english')
    features = pd.DataFrame(vectorizer.fit_transform(dt['uttr'].values).toarray().astype(np.float32)).values
    labels = dt['dialogue_act_id'].values
     #prepare dataset
    x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.15, random_state=0)
    return x_train, x_test, y_train, y_test 
def train(x_train, x_test, y_train, y_test):
   
    # define model and fit it
    lr = LogisticRegression(penalty='l2',solver='saga')
    lr.fit(x_train, y_train)
    filename = 'lr_model.sav'
    pickle.dump(lr, open(filename, 'wb'))
    score = lr.score(x_test, y_test)
    print(score)
def predict_lr(x_test,y_test):
    model = pickle.load(open('lr_model.sav', 'rb'))
    model.predict(x_test)

dt = create_dataset()
x_train, x_test, y_train, y_test = vectorize(dt)
# train(x_train, x_test, y_train, y_test)
# train(vectorize(dt))
predict_lr(x_test,y_test)