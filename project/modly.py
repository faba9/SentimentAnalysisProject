import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('/home/me/jupyter-workspace/Amazon_Unlocked_Mobile.csv')
df.dropna(inplace=True)
#remove any neutral rating equals to 3.
df = df[df['Rating']!=3]
#Encode 4 star and 5 star as positively rated 1.
#Encode 1 star and 2 star as poorely rated 0.
df['Positively Rated'] = np.where(df['Rating']>3,1,0)

X = df['Reviews']
y = df['Positively Rated']
#spliting data into training and test set.
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


def create_predict(review):
    predict = model.predict(vect.transform([review]))
    pr = predict.tolist()
    p = pr[0]
    return p