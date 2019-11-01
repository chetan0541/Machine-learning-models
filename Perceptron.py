from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScalar
from sklearn.linear_model import Prerceptron

iris = datasets.load_iris()
x = iris.data[:, [2,3]]
y = iris.target
# print ('Class labels:', y)  #prints all labels
#print ('Unique class labels:', np.unique(y)) #prints unique lables 

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)    #randomly splits data into training and testing.If you don't mention the random_state in the code, then whenever you execute your code a new random value is generated and the train and test datasets would have different values each time.However, if you use a particular value for random_state(random_state = 1 or any other value) everytime the result will be same,i.e, same values in train and test datasets. stratify basically splits the such that test subsets have the same propotions of class labels as the input dataset.

# print ('Label counts in y', np.bincount(y))
# print ('Label counts in y_train', np.bincount(y_train))
# print ('Label counts in y_test', np.bincount(y_test))

sc = StandardScalar()
sc.fit(X_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

p = Prerceptron()