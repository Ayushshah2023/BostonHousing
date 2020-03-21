import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
   
# Importing Data 
from sklearn.datasets import load_boston 
boston = load_boston()

boston.data.shape

boston.feature_names

#Converting data from nd-array to dataframe and adding feature names to the data
data = pd.DataFrame(boston.data) 
data.columns = boston.feature_names 
  
data.head(10)

# Adding 'Price' (target) column to the data  
boston.target.shape
data['Price'] = boston.target 
data.head()

#Description of the dataset
data.describe()

#information of the dataset
data.info()


# Input Data 
x = boston.data 
   
# Output Data 
y = boston.target 
   
   
# splitting data to training and testing dataset.  
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2, 
                                                    random_state = 0) 
   
print("xtrain shape : ", xtrain.shape) 
print("xtest shape  : ", xtest.shape) 
print("ytrain shape : ", ytrain.shape) 
print("ytest shape  : ", ytest.shape)

# Fitting Multi Linear regression model to training model 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(xtrain, ytrain) 
   
# predicting the test set results 
y_pred = regressor.predict(xtest)

# Plotting Scatter graph to show the prediction  
# results - 'ytrue' value vs 'y_pred' value 
plt.scatter(ytest, y_pred, c = 'green') 
plt.xlabel("Price: in $1000's") 
plt.ylabel("Predicted value") 
plt.title("True value vs predicted value : Linear Regression") 
plt.show()


# Results of Linear Regression. 
from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(ytest, y_pred) 
print("Mean Square Error : ", mse) 
