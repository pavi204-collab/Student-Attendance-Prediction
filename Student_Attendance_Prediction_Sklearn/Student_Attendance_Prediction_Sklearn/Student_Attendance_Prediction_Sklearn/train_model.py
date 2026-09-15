import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_attendance.csv"
MODEL_PATH = BASE_DIR / "student_attendance_model.pkl"
FEATURE_IMAGE = BASE_DIR / "feature_importance.png"

df = pd.read_csv(DATA_PATH)

features = [
    "study_hours_per_day",
    "assignment_completion_percent",
    "previous_attendance_percent",
    "internal_marks",
    "extracurricular_hours",
    "late_count"
]
target = "will_attend_regularly"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    max_depth=8
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

importance = pd.Series(
    model.feature_importances_, index=features
).sort_values(ascending=True)

importance.plot(kind="barh", figsize=(9, 5), title="Feature Importance")
plt.tight_layout()
plt.savefig(FEATURE_IMAGE)
print(f"Feature importance chart saved to: {FEATURE_IMAGE}")
