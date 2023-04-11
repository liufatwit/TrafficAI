import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#%%
traffic_data = pd.read_csv('Traffic_data.csv')
traffic_data.loc[len(traffic_data.index)] = [15 , 16, 20, 20]
model = LinearRegression()
#%% md
#Viewing the data
#%%
traffic_data
#%% md
#Plotting our data
#%%
plt.scatter(traffic_data["Numbers of cars on X"], traffic_data["Numbers of cars on Y"])
plt.axvline(x = 20, color = 'red')
plt.axhline(y = 20, color = 'black')
plt.legend(['Traffic Data','Traffic seconds for X', "Traffic seconds for Y"], loc = 'center left', bbox_to_anchor = (1.04, 0.5))
plt.xlabel("Number of Cars on X")
plt.ylabel("Number of Cars on Y")
plt.show()
#%% md
#Training the model and Testing the model
#%%
X = traffic_data[['Numbers of cars on X', 'Numbers of cars on Y']]  # Input features
y = traffic_data[['Traffic seconds for X', 'Traffic seconds for Y']]  # Target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)
#%% md
#Printing the Mean Squared Error, Root Mean Squared Error, and R-squared
#%%
from sklearn.metrics import mean_squared_error

# Calculate MSE on validation data
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

from sklearn.metrics import r2_score

# Calculate R2 score on validation data
r2 = r2_score(y_val, y_pred_val)

print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (RMSE): ", rmse)
print("R-squared (R2) Score: ", r2)
#%% md
#Adjusting the regression
#%%
from sklearn.tree import DecisionTreeRegressor

# Create an instance of DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# Set hyperparameter values
regressor.max_depth = 5  # Maximum depth of the decision tree
regressor.min_samples_split = 2  # Minimum number of samples required to split an internal node
regressor.min_samples_leaf = 1  # Minimum number of samples required to be at a leaf node

# Fit the model to the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (RMSE): ", rmse)

#%% md
#Plotting the evaluation model
#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Create an instance of DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1)

# Fit the model to the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Extract the "Number of cars" from the test data
num_cars_test = X_test['Numbers of cars on X']

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Plot the predicted vs. actual values for the first output (X)
plt.scatter(num_cars_test, y_test['Traffic seconds for X'], color='blue', label='Actual Traffic seconds for X')
plt.scatter(num_cars_test, y_pred[:, 0], color='red', label='Predicted Traffic seconds for X')
plt.xlabel('Number of cars on X')
plt.ylabel('Traffic seconds for X')
plt.title('Actual vs. Predicted Traffic seconds for X')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()

# Plot the predicted vs. actual values for the second output (Y)
plt.scatter(num_cars_test, y_test['Traffic seconds for Y'], color='blue', label='Actual Traffic seconds for Y')
plt.scatter(num_cars_test, y_pred[:, 1], color='red', label='Predicted Traffic seconds for Y')
plt.xlabel('Number of cars on Y')
plt.ylabel('Traffic seconds for Y')
plt.title('Actual vs. Predicted Traffic seconds for Y')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
#%% md
#Trying to create a plot from the 2 plots merged together
#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Create an instance of DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1)

# Fit the model to the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Extract the "Number of cars" from the test data
num_cars_test = X_test['Numbers of cars on X']

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Plot the predicted vs. actual values for both outputs (X and Y)
plt.scatter(num_cars_test, y_test['Traffic seconds for X'], color='blue', label='Actual Traffic seconds for X')
plt.scatter(num_cars_test, y_test['Traffic seconds for Y'], color='green', label='Actual Traffic seconds for Y')
plt.scatter(num_cars_test, y_pred[:, 0], color='red', label='Predicted Traffic seconds for X')
plt.scatter(num_cars_test, y_pred[:, 1], color='orange', label='Predicted Traffic seconds for Y')
plt.xlabel('Number of cars')
plt.ylabel('Traffic seconds')
plt.title('Actual vs. Predicted Traffic seconds for X and Y')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
#%% md
#Trying to fix up the plot
#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Create an instance of DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1)

# Fit the model to the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Extract the "Number of cars" from the test data
num_cars_test = X_test['Numbers of cars on X']

# Extract the predicted traffic seconds for X and Y
y_pred_x = y_pred[:, 0]
y_pred_y = y_pred[:, 1]

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Plot the predicted vs. actual values for both X and Y
plt.scatter(num_cars_test, y_test['Traffic seconds for X'], color='blue', label='Actual Traffic seconds for X')
plt.scatter(num_cars_test, y_test['Traffic seconds for Y'], color='green', label='Actual Traffic seconds for Y')
plt.scatter(num_cars_test, y_pred_x, color='red', label='Predicted Traffic seconds for X')
plt.scatter(num_cars_test, y_pred_y, color='orange', label='Predicted Traffic seconds for Y')
plt.xlabel('Number of cars')
plt.ylabel('Traffic seconds')
plt.title('Actual vs. Predicted Traffic seconds')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
#%% md
#Changed plot to 3D plot to show that the Number of cars on X and Y affect traffic seconds
#%%
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Create an instance of DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=2, min_samples_leaf=1)

# Fit the model to the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Extract the "Number of cars on X" and "Number of cars on Y" from the test data
num_cars_x_test = X_test['Numbers of cars on X']
num_cars_y_test = X_test['Numbers of cars on Y']

# Extract the predicted traffic seconds for X and Y
y_pred_x = y_pred[:, 0]
y_pred_y = y_pred[:, 1]

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the actual traffic seconds
ax.scatter(num_cars_x_test, num_cars_y_test, y_test['Traffic seconds for X'], color='blue', label='Actual Traffic seconds for X')
ax.scatter(num_cars_x_test, num_cars_y_test, y_test['Traffic seconds for Y'], color='green', label='Actual Traffic seconds for Y')

# Plot the predicted traffic seconds
ax.scatter(num_cars_x_test, num_cars_y_test, y_pred_x, color='red', label='Predicted Traffic seconds for X')
ax.scatter(num_cars_x_test, num_cars_y_test, y_pred_y, color='orange', label='Predicted Traffic seconds for Y')

ax.set_xlabel('Number of cars on X')
ax.set_ylabel('Number of cars on Y')
ax.set_zlabel('Traffic seconds')
ax.set_title('Actual vs. Predicted Traffic seconds')

# Move the legend outside of the plot, and adjust the position to avoid overlapping with labels
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
plt.show()

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
#%% md
#Doing some clean
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('traffic_data.csv')

# Extract the features (Number of cars on X and Number of cars on Y)
X = df[['Numbers of cars on X', 'Numbers of cars on Y']]

# Extract the target variable (Traffic seconds for X and Traffic seconds for Y)
y = df[['Traffic seconds for X', 'Traffic seconds for Y']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=100, min_samples_split=15, min_samples_leaf=1)

# Fit the model to the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Extract the "Number of cars on X" and "Number of cars on Y" from the test data
num_cars_x_test = X_test['Numbers of cars on X']
num_cars_y_test = X_test['Numbers of cars on Y']

# Extract the predicted traffic seconds for X and Y
y_pred_x = y_pred[:, 0]
y_pred_y = y_pred[:, 1]

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

#%%
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the actual traffic seconds
ax.scatter(num_cars_x_test, num_cars_y_test, y_test['Traffic seconds for X'], color='purple', marker='o', label='Actual Traffic seconds for X')
ax.scatter(num_cars_x_test, num_cars_y_test, y_test['Traffic seconds for Y'], color='green', marker='o', label='Actual Traffic seconds for Y')

# Plot the predicted traffic seconds
ax.scatter(num_cars_x_test, num_cars_y_test, y_pred_x, color='red', marker='o', label='Predicted Traffic seconds for X')
ax.scatter(num_cars_x_test, num_cars_y_test, y_pred_y, color='orange', marker='s', label='Predicted Traffic seconds for Y')

ax.set_xlabel('Number of cars on X')
ax.set_ylabel('Number of cars on Y')
ax.set_zlabel('Traffic seconds')
ax.set_title('Actual vs. Predicted Traffic seconds')

# Add grid lines
ax.grid(True)

# Adjust the aspect ratio of the plot
ax.dist = 12

# Move the legend outside of the plot, and adjust the position to avoid overlapping with labels
ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
plt.show()

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
