import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ----------------------------------------
# 🔧 Page Configuration
# ----------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# ----------------------------------------

# 📌 Sidebar: About Me & Contact
# ----------------------------------------
st.sidebar.title("👤 About Me")
st.sidebar.info("""
**Mahi this side!**

🎓 I'm pursuing **BCA in AI & ML** from **Baba Farid Group of Institutions** (2nd Year).

🛠️ This project was built as part of my **Summer Training**.

🔧 Technologies Used:
- Python
- Pandas
- Streamlit
- XGBoost
""")

st.sidebar.title("📞 Contact Me")
st.sidebar.markdown("""
- 📱 **Phone**: 9988650994  
- 📧 **Email**: vinaygaba267@gmail.com
""")

# ----------------------------------------
# 🧠 Title & Description
# ----------------------------------------
st.title("💓 Heart Disease Prediction App")
st.markdown("### Powered by XGBoost — Fast, Accurate & Reliable")

# ----------------------------------------
# 📊 Load & Prepare Data
# ----------------------------------------
df = pd.read_csv(r"heart.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------
# 🤖 Train the Model
# ----------------------------------------
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ----------------------------------------
# 📈 Show Accuracy
# ----------------------------------------
st.markdown(f"**🔍 Model Accuracy:** `{accuracy * 100:.2f}%`")

# ----------------------------------------
# 📝 User Input
# ----------------------------------------
st.markdown("### 👤 Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=29, max_value=77, value=54)
    sex = st.radio("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options={
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }.items(), format_func=lambda x: x[1])

with col2:
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 245)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", options={
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    }.items(), format_func=lambda x: x[1])

with col3:
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.radio("Exercise Induced Angina?", ["No", "Yes"])
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope of ST Segment", {
        0: "Upsloping", 1: "Flat", 2: "Downsloping"
    }.items(), format_func=lambda x: x[1])
    ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", {
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }.items(), format_func=lambda x: x[1])

# ----------------------------------------
# 📦 Prepare Data for Prediction
# ----------------------------------------
user_input = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp[0],
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "restecg": restecg[0],
    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": slope[0],
    "ca": ca,
    "thal": thal[0]
}

input_df = pd.DataFrame([user_input])

# ----------------------------------------
# 🧪 Make Prediction
# ----------------------------------------
st.markdown("---")
if st.button("🔍 Predict Heart Disease"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error("⚠️ **Heart Disease Detected!**")
        st.markdown(f"**Confidence:** `{prob * 100:.2f}%`")
    else:
        st.success("✅ **No Heart Disease Detected.**")
        st.markdown(f"**Confidence:** `{prob * 100:.2f}%`")

# ----------------------------------------
# 📌 Disclaimer
# ----------------------------------------
st.markdown("---")
st.caption("Note: This prediction is based on historical data and should not replace professional medical advice.")

