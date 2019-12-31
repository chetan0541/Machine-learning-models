import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 

data = pd.read_csv("Datasets/Salary_Data.csv")

X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# print(X_train.shape)
# print(X_train)
# print("Shape",y_train.shape)
# print("ytrain",y_train)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# print(y_pred)
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, model.predict(X_train), color= 'black')
# plt.title('Salary and Experience Relationship(Training Set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color= 'black')
plt.title('Salary and Experience Relationship(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()