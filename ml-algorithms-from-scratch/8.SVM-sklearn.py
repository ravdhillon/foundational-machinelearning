import numpy as np
import pandas as pd

from sklearn import preprocessing, neighbors, model_selection, svm
import os


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

# 4. Prepare dataset for training.
# 4.1 Define your features (X) and labels (y) => Features are basically everything except the label column.

X = np.array(df.drop(['class'], 1)) # drop class column
print(X)
y = np.array(df['class'])

# 4.2 Split the data into training and testing.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# 4.3 Define your classifier
clf = svm.SVC()

# 5. Fit (Train) your model on the training data (features and labels)
clf.fit(X_train, y_train)

# 6. Model Evaluation
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

# 7. Prediction
example = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)
print('Prediction (2 or 4): ', prediction)