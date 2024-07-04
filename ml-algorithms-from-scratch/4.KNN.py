import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import style
from math import sqrt
from collections import Counter
import warnings
import pandas as pd
import random
import os

# style.use('fivethirtyeight')

dataset = { 'k': [[1,2], [2,3], [3,1]],  'r':[[6,5], [7,7], [8,6]] }
new_feature = [5,7]

# Visualize the dataset
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

# plt.show()

"""
    data: the feature set (training data)
    predict: the example of dataset we want to predict
    k is the number of points to consider.
"""

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than total classes')
    
    distances = []
    for group in data:
        for features in data[group]:
            # Following are some of the ways to calculate the Euclidean distance
            # euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1] - predict[1])**2 )
            # euclidean_distance = np.sqrt(np.sum( (np.array(features) - np.array(predict)) **2 ))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) # faster
            distances.append([euclidean_distance, group]) 

    ## TODO: can we use heap to filter the K items.
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

dataset_dir = '/home/ravisher/Development/AI-Lab/machine-learning-with-python/Datasets'
dataset_name = 'breast-cancer-wisconsin.data'

# 1. Load the dataset into a dataframe. it could be anything like csv file, txt-file, html 
df = pd.read_csv(os.path.join(dataset_dir, dataset_name))

# 2. Prepare the dataset for training
# 2.1 Assign the columns to the dataframe if the dataset doesn't have any columns associated with it.
columns = ['id', 'thickness', 'cell_size', 'cell_shape', 'marginal_adhesion', 'single_epi_cell_size', 'bare_nuclei', 
'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df.columns = columns

# 3. EDA - Exploratory Data Analysis 
# 3.1 Missing data
df.replace('?', -9999, inplace=True)

# 3.2 Remove redundant columns which we may not need.
df.drop(['id'], 1, inplace=True)
print(df.head())
# 3.3. Convert the dataset to float (Optional). This is done in case there are any string values
full_data = df.astype(float).values.tolist()
# 3.4 Shuffle the data
random.shuffle(full_data)

# Train test split without sklearn
test_size = 0.2
# Here 2 and 4 are classes.
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}  
train_data = full_data[:-int(test_size * len(full_data))] # everything up to last 20% of the data
test_data = full_data[int(test_size * len(full_data)):] # last 20% of the data.

#train_set[i[-1]] --> this gives us the class column.
for i in train_data:
    train_set[i[-1]].append(i[:-1]) #append everything up to the last column (which is the class column)

for i in test_data:
    test_set[i[-1]].append(i[:-1]) #append everything up to the last column (which is the class column)

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: ', correct/total)