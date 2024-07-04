import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection
from matplotlib import style
style.use('ggplot')
import os


dataset_dir = '/home/ravisher/Development/AI-Lab/machine-learning-with-python/Datasets'
dataset_name = 'titanic.csv'

df = pd.read_csv(os.path.join(dataset_dir, dataset_name))
original_df = pd.DataFrame.copy(df)

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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i] #iloc refers to the index and i is the row index.

n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'])==float(i) ]
    survival_cluster = temp_df[ (temp_df['Survived']==1) ]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

