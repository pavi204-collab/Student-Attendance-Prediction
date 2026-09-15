import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Student Attendance Prediction",
    page_icon="🎓",
    layout="wide"
)

# Robust Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "student_attendance_model.pkl").exists() and (BASE_DIR / "Student_Attendance_Prediction_Sklearn" / "student_attendance_model.pkl").exists():
    BASE_DIR = BASE_DIR / "Student_Attendance_Prediction_Sklearn"

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-regular {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .badge-irregular {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.3);
    }
    .metric-banner {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Student Academic Attendance & Retention Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Early-warning intervention system using <b>RandomForestClassifier</b> to identify at-risk students before attendance shortage affects exam eligibility.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "student_attendance_model.pkl"
csv_path = BASE_DIR / "data" / "student_attendance.csv"
chart_path = BASE_DIR / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Student Early-Warning Predictor", "📈 Academic Predictor Drivers", "📋 Historical Cohort Dataset"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Student Academic & Engagement Indicators")
        
        persona = st.selectbox(
            "⚡ Quick Student Persona Preset",
            ["Custom Student Profile", "🌟 Disciplined Scholar (Regular Attendance)", "⚠️ Chronic Tardy Student (High Late Count)", "⚽ Extracurricular Athlete (Balanced)", "📉 Academically Disengaged Student"]
        )
        
        if persona == "🌟 Disciplined Scholar (Regular Attendance)":
            def_study, def_assign, def_prev, def_marks, def_extra, def_late = 4.5, 95.0, 92.0, 88.0, 3.0, 0
        elif persona == "⚠️ Chronic Tardy Student (High Late Count)":
            def_study, def_assign, def_prev, def_marks, def_extra, def_late = 1.5, 55.0, 62.0, 50.0, 6.0, 9
        elif persona == "⚽ Extracurricular Athlete (Balanced)":
            def_study, def_assign, def_prev, def_marks, def_extra, def_late = 2.5, 80.0, 78.0, 72.0, 14.0, 2
        elif persona == "📉 Academically Disengaged Student":
            def_study, def_assign, def_prev, def_marks, def_extra, def_late = 1.0, 40.0, 52.0, 42.0, 4.0, 7
        else:
            def_study, def_assign, def_prev, def_marks, def_extra, def_late = 3.5, 85.0, 82.0, 75.0, 4.0, 1
            
        with st.form("attendance_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                study_hours = st.slider("Daily Study Hours", 0.0, 12.0, float(def_study), step=0.25)
                assignment_completion = st.slider("Assignment Submission Rate (%)", 0.0, 100.0, float(def_assign), step=1.0)
                previous_attendance = st.slider("Prior Semester Attendance (%)", 0.0, 100.0, float(def_prev), step=0.5)
            with col_b:
                internal_marks = st.slider("Continuous Assessment Marks (/100)", 0.0, 100.0, float(def_marks), step=1.0)
                extracurricular_hours = st.slider("Weekly Extracurricular Hours", 0.0, 25.0, float(def_extra), step=0.5)
                late_count = st.slider("Late Arrivals / Tardy Count", 0, 20, int(def_late))
                
            submit_btn = st.form_submit_button("🚀 Predict Attendance Standing", use_container_width=True)
            
    with col_result:
        st.subheader("Attendance Status & Intervention Guidance")
        if submit_btn:
            sample = pd.DataFrame([{
                "study_hours_per_day": study_hours,
                "assignment_completion_percent": assignment_completion,
                "previous_attendance_percent": previous_attendance,
                "internal_marks": internal_marks,
                "extracurricular_hours": extracurricular_hours,
                "late_count": late_count
            }])
            
            pred = model.predict(sample)[0]
            prob_arr = model.predict_proba(sample)[0]
            classes = list(model.classes_)
            reg_idx = classes.index(1) if 1 in classes else 1
            prob_regular = prob_arr[reg_idx]
            
            if pred == 1:
                st.markdown(f"""
                <div class="badge-regular">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Projected Status</div>
                    <div class="metric-banner">🎓 LIKELY TO ATTEND REGULARLY</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Regularity Confidence: {prob_regular:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="badge-irregular">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Early-Warning Notice</div>
                    <div class="metric-banner">⚠️ RISK OF IRREGULAR ATTENDANCE</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Risk Severity: {(1 - prob_regular):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.markdown("### 📊 Regularity Probability")
            st.progress(float(prob_regular))
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Regular Attendance Likelihood", f"{prob_regular:.1%}")
            col_m2.metric("Absence / Shortage Risk", f"{(1 - prob_regular):.1%}")
            
            st.markdown("### 🧭 Academic Advisor Guidance")
            if previous_attendance < 75.0:
                st.error("🚨 **75% Mandatory Threshold Alert:** Prior attendance is below University minimum. Student is currently on attendance probation.")
            if late_count >= 5:
                st.warning("⏰ **Frequent Latecomer Warning:** Schedule a time management counseling session with mentor.")
            if assignment_completion < 60.0:
                st.info("📚 **Academic Disengagement:** Incomplete homework/assignments correlate strongly with impending lecture truancy.")
            if pred == 1 and previous_attendance >= 80.0:
                st.success("✅ **Good Standing:** Student exhibits positive academic engagement and dependable attendance habits.")
                
            with st.expander("🔍 Model Input Payload"):
                st.json(sample.to_dict(orient="records")[0])
        else:
            st.info("👈 Set student indicators and click **'Predict Attendance Standing'**.")

with tab2:
    st.subheader("Predictive Drivers for Regular Attendance")
    if chart_path.exists():
        st.image(str(chart_path), caption="Top Features Influencing Regular Student Attendance", use_container_width=True)
    else:
        st.info("Feature importance plot will appear after running train_model.py")

with tab3:
    st.subheader("Historical Student Cohort Dataset (student_attendance.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students Tracked", f"{len(df):,}")
        col2.metric("Regular Attendance Rate", f"{df['will_attend_regularly'].mean():.1%}")
        col3.metric("Avg Previous Attendance", f"{df['previous_attendance_percent'].mean():.1f}%")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Dataset not found.")
