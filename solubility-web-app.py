# Import necessary libraries
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the dataset
delaney_with_descriptors_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'
dataset = pd.read_csv(delaney_with_descriptors_url)

# Prepare the features (X) and target (Y)
X = dataset.drop(['logS'], axis=1)
Y = dataset.iloc[:,-1]

# Build the linear regression model
model = linear_model.LinearRegression()
model.fit(X, Y)

# Predict the solubility (logS)
Y_pred = model.predict(X)

# Evaluate the model
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f' % mean_squared_error(Y, Y_pred))
print('Coefficient of determination (R^2): %.2f' % r2_score(Y, Y_pred))

# Model equation
print('LogS = %.2f %.2f LogP %.4f MW + %.4f RB %.2f AP' % (model.intercept_, model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3]))

# Data visualization
plt.figure(figsize=(5,5))
plt.scatter(x=Y, y=Y_pred, c="#7CAE00", alpha=0.3)

# Add trendline
z = np.polyfit(Y, Y_pred, 1)
p = np.poly1d(z)
plt.plot(Y, p(Y), "#F8766D")
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')

# Save the model as a pickle object
pickle.dump(model, open('solubility_model.pkl', 'wb'))
