import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Load dataset
train_data = pd.read_csv("digit-recognizer/train.csv")

# Features and labels
X = train_data.drop(columns=['label']).values / 255.0  # Normalize
y = train_data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
svm_clf = SVC(kernel='linear', random_state=42)
svm_clf.fit(X_train, y_train)

# Evaluate
y_pred = svm_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(svm_clf, "svm_digit_model.pkl")
joblib.dump(acc, "accuracy.pkl")
print("✅ Model saved as svm_digit_model.pkl")
