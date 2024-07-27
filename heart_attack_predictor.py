import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
heart_data = pd.read_csv('dataset/heart.csv')
o2_data = pd.read_csv('dataset/o2Saturation.csv', header=None, names=['o2Saturation'])

# Preprocess data
# Assuming o2Saturation is an additional feature to heart_data
heart_data['o2Saturation'] = o2_data['o2Saturation']

# Split data into features and target
X = heart_data.drop('output', axis=1)
y = heart_data['output']

# Save the columns of X
with open('columns.txt', 'w') as f:
    for column in X.columns:
        f.write(f"{column}\n")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model
joblib.dump(model, 'heart_attack_model.pkl')