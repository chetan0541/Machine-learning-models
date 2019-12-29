import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

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
