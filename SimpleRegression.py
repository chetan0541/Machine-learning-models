import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from yellowbrick.regressor import ResidualsPlot

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


# y_pred = model.predict(X_test)

# print(y_pred)
#Plotting Training values on a scatter plot
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, model.predict(X_train), color= 'black')
# plt.title('Salary and Experience Relationship(Training Set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

#Checking Homoscedasticity
# hom = ResidualsPlot(model)
# hom.fit(X_train, y_train)
# hom.score(X_test, y_test)
# hom.poof() 

#Checking Normal Distribution of error terms
# model = sm.OLS(y_train, X_train).fit()
# res = model.resid
# plot = sm.qqplot(res, fit=True, line='r')
# plt.show()

#Checking autocorrelation using Durbin-Watson
# model = sm.OLS(y_train, X_train).fit()
# print(model.summary())

# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_test, y_pred, color= 'black')
# plt.title('Salary and Experience Relationship(Test Set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()