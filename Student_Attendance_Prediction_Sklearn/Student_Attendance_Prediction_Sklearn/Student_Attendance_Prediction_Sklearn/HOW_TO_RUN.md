# How to Run: Student Attendance Prediction using Python + Scikit-learn

This machine learning project predicts whether a student is likely to attend classes regularly based on academic and attendance-related features.

---

## 1. Prerequisites

- **Python**: Python 3.9+ (Python 3.14 tested)
- **Terminal**: Windows PowerShell, Command Prompt, or VS Code integrated terminal
- **Dependencies**: `pandas`, `scikit-learn`, `numpy`, `joblib`, `matplotlib`

---

## 2. Open Terminal & Navigate to Project Directory

Open PowerShell or Command Prompt, then change directory to the project folder:

```powershell
# Note: Project files are inside the nested "Student_Attendance_Prediction_Sklearn" folder:
cd "c:\Users\user\Desktop\jj college\Student_Attendance_Prediction_Sklearn\Student_Attendance_Prediction_Sklearn"
```

---

## 3. Install Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

> **Windows Note**: If `python` or `pip` opens the Microsoft Store, specify the direct Python path:
> ```powershell
> & "C:\Users\user\AppData\Local\Python\bin\python.exe" -m pip install -r requirements.txt
> ```

---

## 4. Step 1: Train the Machine Learning Model

Execute the training script to train and save the model:

```powershell
python train_model.py
```
*(or `py train_model.py` / `& "C:\Users\user\AppData\Local\Python\bin\python.exe" train_model.py`)*

### What this does:
1. Loads the dataset from `data/student_attendance.csv`.
2. Trains a **RandomForestClassifier** model predicting `will_attend_regularly`.
3. Evaluates model performance (Accuracy Score, Classification Report).
4. Saves the trained model (`.pkl` file).
5. Generates and saves visualization plots/charts (e.g., feature importance or segment visualizations).

---

## 5. Step 2: Run Predictions

Run the prediction script to test predictions:

```powershell
python predict.py
```

### Interactive Inputs:
This script prompts for inputs in the terminal. Here is an example of what to enter when prompted:

| Prompt / Field | Example Value |
| :--- | :--- |
| `Study hours per day:` | `4` |
| `Assignment completion percentage:` | `85` |
| `Previous attendance percentage:` | `85` |
| `Internal marks:` | `40` |
| `Extracurricular hours per week:` | `3` |
| `Number of late arrivals:` | `1` |

---

## 6. Directory Structure

```text
Student_Attendance_Prediction_Sklearn/
├── data/
│   └── student_attendance.csv
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md
```
