import sklearn 
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
data = pd.read_csv("Housing.csv")

X = data['price']
y = data['lotsize']

# print (len(X))
# print (len(y))

X = X.values.reshape(len(X),1)
y = y.values.reshape(len(X),1)

# print ("X:", X)
# print ("shape:", X.shape)
# print ("Y:", y)
# print ("shape of y:",y.shape)

X_train , X_test, y_train, y_test = train_test_split(X, y, random_state = 1,test_size= 0.3)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

plt.scatter(X_test, y_test, color='black')
plt.title("Test Data")
plt.xlabel("Size")
plt.ylabel("Price")
plt.xticks(())
plt.yticks(())

lr = LinearRegression()

lr.fit(X_train,y_train)

plt.plot(X_test, lr.predict(X_test), color='red')
plt.show()