# load libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#read dataset as dataframe
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

###### LOGISTIC REGRESSION
#vectorize feactures and labels to change the text to the number
dt['dialogue_act_id'] = dt['dialogue'].factorize()[0]
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(dt['uttr']).toarray()
labels = dt['dialogue_act_id']
#prepare dataset
x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.15, random_state=0)
# define model and fit it
lr = LogisticRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
score = lr.score(x_test, y_test)
print(score)