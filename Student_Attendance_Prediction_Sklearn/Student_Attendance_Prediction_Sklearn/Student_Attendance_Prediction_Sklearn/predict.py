import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_attendance_model.pkl"

model = joblib.load(MODEL_PATH)

print("Student Attendance Prediction")
print("-" * 35)

study_hours = float(input("Study hours per day: "))
assignment_completion = float(input("Assignment completion percentage: "))
previous_attendance = float(input("Previous attendance percentage: "))
internal_marks = float(input("Internal marks: "))
extracurricular_hours = float(input("Extracurricular hours per week: "))
late_count = int(input("Number of late arrivals: "))

sample = pd.DataFrame([{
    "study_hours_per_day": study_hours,
    "assignment_completion_percent": assignment_completion,
    "previous_attendance_percent": previous_attendance,
    "internal_marks": internal_marks,
    "extracurricular_hours": extracurricular_hours,
    "late_count": late_count
}])

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0].max()

if prediction == 1:
    result = "Likely to attend classes regularly"
else:
    result = "May have irregular attendance"

print("\nPrediction:", result)
print(f"Confidence: {probability:.2%}")
