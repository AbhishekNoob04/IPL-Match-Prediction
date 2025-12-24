import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load your data (replace 'train_data.csv' if file name is different)
df = pd.read_csv('train_data.csv')

# 2. Feature columns and target
X = df.drop('win', axis=1)
y = df['win']

# 3. Preprocessing: Categorical encoding
categorical_cols = ['batting_team', 'bowling_team', 'city']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# 4. Build the pipeline
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train the model
clf.fit(X, y)

# 6. Save it as pipe.pkl
with open('pipe.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved as pipe.pkl")
