# Student Attendance Prediction using Python + Scikit-learn

## Project Overview
This machine learning project predicts whether a student is likely to attend classes regularly based on academic and attendance-related features.

## Technology Used
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Matplotlib
- Joblib

## Input Features
- Study hours per day
- Assignment completion percentage
- Previous attendance percentage
- Internal marks
- Extracurricular hours
- Late arrival count

## Target
- `1` = Likely to attend regularly
- `0` = May have irregular attendance

## How to Run

### 1. Install libraries
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

This creates:
- `student_attendance_model.pkl`
- `feature_importance.png`

### 3. Make a prediction
```bash
python predict.py
```

Enter the student details when asked.

## Important Note
This project uses a synthetic dataset for educational purposes. It should not be used for official academic decisions without real, validated data and proper review.
