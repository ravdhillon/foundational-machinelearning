import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
from matplotlib import style
style.use('ggplot')
import os


dataset_dir = '/home/ravisher/Development/AI-Lab/machine-learning-with-python/Datasets'
dataset_name = 'titanic.csv'

df = pd.read_csv(os.path.join(dataset_dir, dataset_name))

df.drop(['Name'], 1, inplace=True)
print(df.head())
df.apply(pd.to_numeric, errors='ignore').dtypes
df.fillna(0, inplace=True)

# This is for handling the nominal data
def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            
            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)
print(df.head())
print('-'*120)
X = np.array(df.drop(['Survived'], 1).astype(float))
print(X)
print('-'*120)
# Some techniques - scaling
# Scaling is done to the feature set so that they have properties of a standard normal distribution with a mean of 0 and std.dev of 1.
X = preprocessing.scale(X)
print(X)
y = np.array(df['Survived'])
print('-'*120)

clf = KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1,  len(predict))

    prediction = clf.predict(predict)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))