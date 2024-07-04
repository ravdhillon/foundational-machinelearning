import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
from collections import Counter
import warnings

style.use('fivethirtyeight')

dataset = { 'k': [[1,2], [2,3], [3,1]],  'r':[[6,5], [7,7], [8,6]] }
new_feature = [5,7]

# Visualize the dataset
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

plt.show()

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

result = k_nearest_neighbors(dataset, new_feature, 3)
print(result)
    


