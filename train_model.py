import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv("Instagram_fake_profile_dataset.csv")

# ------------------------------
# Feature engineering
# ------------------------------
eps = 1e-6
df['followers_to_follows'] = df['followers'] / (df['follows'] + eps)
df['posts_per_1k_followers'] = df['posts'] / ((df['followers'] + eps) / 1000)
for col in ['followers', 'follows', 'posts']:
    df[f'log1p_{col}'] = np.log1p(df[col])

X = df.drop(columns=['fake'])
y = df['fake']

# ------------------------------
# Train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# Scale numeric features
# ------------------------------
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ------------------------------
# Train logistic regression
# ------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Save model and scaler
# ------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
