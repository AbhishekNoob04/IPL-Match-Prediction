import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
matches = pd.read_csv("data/matches.csv")

# Train simple model
model = LogisticRegression()
X = matches[['feature1', 'feature2']]  # Replace with your actual features
y = matches['target']                  # Replace with your target column
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")
