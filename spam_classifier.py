import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# keep only useful columns
data = data[['v1','v2']]

# rename columns
data.columns = ['label','message']

# convert labels to numbers
data['label'] = data['label'].map({'ham':0,'spam':1})

X = data['message']
y = data['label']

# text → numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# train model
model = MultinomialNB()
model.fit(X_train,y_train)

# prediction
pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
