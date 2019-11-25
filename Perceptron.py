from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, [2,3]]    
y = iris.target

# print ("test", X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

print ("Bincount",np.bincount(y))
print ("Bincount",np.bincount(y_train))
print ("Bincount",np.bincount(y_test))

sc = StandardScaler()       #Standardizing 
sc.fit(X_train)     #Estimates the sample mean and standard deviation for each feature set
X_train_std = sc.transform(X_train) #Standardizes them according to the sc.fit
X_test_std = sc.transform(X_test)

per = Perceptron(random_state= 1)
per.fit(X_train_std,y_train)
per.predict(X_test_std[:3,:1])
print ("Array:",per.predict)