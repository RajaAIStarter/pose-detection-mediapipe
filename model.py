import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
# -----------------------------
# Load and Preprocess Training Data
# -----------------------------
train_csv = 'train_pose_landmarks.csv'
train_data = pd.read_csv(train_csv)

# Drop the filename column (assumed to be the first column)
train_data = train_data.drop(columns=train_data.columns[0])

# Separate features and label.
X_train = train_data.drop(columns='label')
y_train = train_data['label']
print(X_train.shape)
# Encode labels.
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# Scale features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a Random Forest classifier.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train_encoded)

# (Optional) Evaluate on the training set via cross-validation.
cv_scores = cross_val_score(clf, X_train_scaled, y_train_encoded, cv=5)
print("Mean CV Accuracy on Training Data:", cv_scores.mean())

# -----------------------------
# Load and Preprocess Test Data
# -----------------------------
test_csv = 'test_pose_landmarks.csv'
test_data = pd.read_csv(test_csv)

# Drop the filename column.
test_data = test_data.drop(columns=test_data.columns[0])

# Separate features and label.
X_test = test_data.drop(columns='label')
y_test = test_data['label']

# Use the same label encoder as for training.
y_test_encoded = le.transform(y_test)

# Scale test features using the same scaler.
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Evaluate the Classifier on the Test Data
# -----------------------------
y_pred = clf.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
#
#
# # Make sure these files exist from your training step.
# joblib.dump(clf, 'rf_model.pkl')
# joblib.dump(scaler,'scaler.pkl')
# joblib.dump(le, 'label_encoder.pkl')

