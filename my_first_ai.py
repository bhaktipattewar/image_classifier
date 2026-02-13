import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("All AI libraries loaded successfully!")

# Let's create some tiny 'dummy' data for your assignment
# Imagine X is hours studied and y is exam score
X = np.array([[1], [2], [3], [4], [5]]) 
y = np.array([10, 20, 30, 40, 50])

# Create the model (Project 5 step)
model = LinearRegression()
model.fit(X, y)

print("The AI has learned the pattern!")
print(f"Prediction for 6 hours of study: {model.predict([[6]])[0]}")