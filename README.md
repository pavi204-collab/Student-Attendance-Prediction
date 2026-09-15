# Student-Attendance-Prediction
Student Attendance Prediction is a machine learning project built with Python and Scikit-learn that predicts whether a student is likely to attend classes regularly. It uses academic performance, previous attendance, study habits, assignment completion, extracurricular activities, and late-arrival data with a Random Forest Classifier.
# Student Attendance Prediction using Python and Scikit-learn

## Project Overview

Student Attendance Prediction is a machine learning project that predicts whether a student is likely to attend classes regularly based on academic and attendance-related factors.

The project uses a Random Forest Classifier to analyze student information and provide an attendance prediction with confidence.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest Classifier
* Matplotlib
* Joblib
* Streamlit

## Input Features

The model uses the following features:

* Study hours per day
* Assignment completion percentage
* Previous attendance percentage
* Internal marks
* Extracurricular hours per week
* Number of late arrivals

## Target Variable

* `1` – Likely to attend classes regularly
* `0` – May have irregular attendance

## Project Workflow

1. Load the student attendance dataset.
2. Select relevant academic and attendance features.
3. Split the dataset into training and testing sets.
4. Train a Random Forest Classification model.
5. Evaluate the model using accuracy and classification metrics.
6. Save the trained model using Joblib.
7. Generate a feature-importance chart.
8. Use the trained model to predict attendance for new student data.

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

This generates the trained model file and feature-importance chart.

### Make Predictions

```bash
python predict.py
```

Enter the requested student details to receive an attendance prediction and confidence score.

### Run the Streamlit Application

```bash
streamlit run app.py
```

## Project Structure

```text
Student_Attendance_Prediction_Sklearn/
├── data/
│   └── student_attendance.csv
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
├── HOW_TO_RUN.md
├── student_attendance_model.pkl
└── feature_importance.png
```

## Applications

* Student attendance monitoring
* Early identification of students with irregular attendance
* Academic performance analysis
* Educational data analysis
* Machine learning classification practice

## Disclaimer

This project uses a synthetic dataset for educational purposes. Predictions should not be used as the sole basis for official academic decisions without validated real-world data and appropriate human review.
