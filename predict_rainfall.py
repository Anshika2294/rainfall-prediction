
# ✅ Predicting Rainfall using Machine Learning

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 📥 Load the dataset
df = pd.read_csv("weatherAUS.csv")

# 🧹 Drop unnecessary columns (only if they exist)
columns_to_drop = ['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Location', 'Date', 'RISK_MM']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# ❌ Drop rows with missing values
df.dropna(inplace=True)

# 🔁 Encode binary yes/no columns
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# 🔢 Encode categorical wind directions
for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# 🎯 Define features and label
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# ✂️ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔮 Predict and evaluate
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# 💾 Save the trained model
with open("rain_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as 'rain_model.pkl'")
