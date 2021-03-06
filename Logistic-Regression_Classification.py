from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from warnings import simplefilter
import matplotlib.pyplot as plt

simplefilter(action='ignore', category=FutureWarning)

# def plot_decision_regions(X, y, classifier, test_idx=None, resolution = 0.02):
#     #setup marker generator and color map
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])

#     #plot the decision surface
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel().T]))
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.3, cmap = cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())

#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[ y == cl,0], y=X[ y==cl, 1],alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
    
#     #highlight test samples
#     if test_idx:   
#         #plot samples
#         X_test, y_test = X[test_idx, :], y[test_idx]

#        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker ='o', s=100, label='test set')


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# print ("test", X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# print ("Bincount",np.bincount(y))
# print ("Bincount",np.bincount(y_train))
# print ("Bincount",np.bincount(y_test))

# sc = StandardScaler()       #Standardizing 
# sc.fit(X_train)     #Estimates the sample mean and standard deviation for each feature set
# X_train_std = sc.transform(X_train) #Standardizes them according to the sc.fit
# X_test_std = sc.transform(X_test)

lr = LogisticRegression(random_state= 1, C = 100.0)
lr.fit(X_train, y_train)
print("Array", lr.predict_proba(X_test[:3,:]))
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))


