import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import joblib

#adding the label columns and combining the 2 columns
true_df = pd.read_csv("True.csv")
true_df['label'] = 1
fake_df = pd.read_csv("Fake.csv")
fake_df['label'] = 0
df = pd.concat([true_df,fake_df],axis=0).reset_index()
df = df.drop('index',axis=1)

def preprocessing(text):
    text = str(text).lower()                       # lowercase
    text = re.sub(r'[^a-z\s]', '', text)           # remove punctuation/numbers
    return text

#preprocessing the dataset
df['content'] = df['text'] + " " + df['title']
df['cleaned_content'] = df['content'].apply(preprocessing)

#splitting the columns into train and test datasets
x = df['cleaned_content']
y = df['label']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,stratify=y)

#converting textual columns into numerical columns
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
xtrain_num = vectorizer.fit_transform(xtrain)
xtest_num = vectorizer.transform(xtest)

#training and evaluating the model
model = LogisticRegression(max_iter=5000,random_state=42)
model.fit(xtrain_num,ytrain)
ypred = model.predict(xtest_num)
print("accuracy score:",accuracy_score(ytest,ypred))
print("precision score:",precision_score(ytest,ypred))
print("recall score:",recall_score(ytest,ypred))
print("f1 score:",f1_score(ytest,ypred))

#saving results
joblib.dump(model,"fake_news_model.pkl")
joblib.dump(vectorizer,"tfifd_vectorizer.pkl")

test_df = pd.DataFrame({
    "content":xtest,
    "label":ytest
})
test_df.to_csv("test_df.csv",index=False)

sample_df = pd.DataFrame({
    "content":xtest
})
sample_df.to_csv("sample_df.csv",index=False)
