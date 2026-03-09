import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# load dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

data = data[['v1','v2']]
data.columns = ['label','message']

data['label'] = data['label'].map({'ham':0,'spam':1})

X = data['message']
y = data['label']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X,y)

# web app UI
st.title("Spam Email Classifier")

msg = st.text_input("Enter your message")

if st.button("Check"):

    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)

    if result[0] == 1:
        st.error("Spam Message")
    else:
        st.success("Not Spam")
