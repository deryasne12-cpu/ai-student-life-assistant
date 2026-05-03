from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Dummy training data
X = np.array([
    [4, -0.5],
    [8, 0.6],
    [6, 0.1],
    [3, -0.8],
    [7, 0.7]
])

y = ["Low", "High", "Medium", "Low", "High"]

model = DecisionTreeClassifier()
model.fit(X, y)

def predict_productivity(sleep, sentiment):
    return model.predict([[sleep, sentiment]])[0]
