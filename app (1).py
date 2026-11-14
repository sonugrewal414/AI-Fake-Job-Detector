
import os
import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "fake_job_model.pkl"
)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
tfidf = model_data["tfidf"]

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    text-align: center;
    color: #17365D;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 17px;
}
.stButton>button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
    height: 48px;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">🛡️ AI Fake Job Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect fraudulent job and internship offers using Artificial Intelligence</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# Information
st.subheader("🔍 Analyze a Job Posting")

st.write(
    "Enter the job information below to check whether "
    "the job posting is genuine or potentially fraudulent."
)

# Input fields
title = st.text_input("Job Title")

company_profile = st.text_area(
    "Company Profile",
    placeholder="Enter information about the company..."
)

description = st.text_area(
    "Job Description",
    placeholder="Enter the complete job description...",
    height=180
)

requirements = st.text_area(
    "Job Requirements",
    placeholder="Enter the required skills and qualifications..."
)

benefits = st.text_area(
    "Benefits",
    placeholder="Enter salary, incentives and other benefits..."
)

st.write("")

# Detection button
if st.button("🔍 Detect Fake Job"):

    if not description.strip():
        st.warning("Please enter the job description first.")

    else:
        # Combine text in the SAME order as training
        job_text = (
            title + " " +
            company_profile + " " +
            description + " " +
            requirements + " " +
            benefits
        )

        # Convert text into numerical features
        transformed_text = tfidf.transform([job_text])

        # Make prediction
        prediction = model.predict(transformed_text)[0]

        # Calculate probability
        probabilities = model.predict_proba(transformed_text)[0]

        fake_index = list(model.classes_).index(1)
        fake_probability = probabilities[fake_index] * 100

        st.divider()
        st.subheader("📊 Detection Result")

        if prediction == 1:

            st.error("⚠️ This job posting is predicted to be FAKE!")

            st.metric(
                "Estimated Fake Probability",
                f"{fake_probability:.2f}%"
            )

            st.warning(
                "Be careful. Verify the company and offer "
                "before sharing personal information or making payments."
            )

        else:

            st.success("✅ This job posting is predicted to be GENUINE!")

            st.metric(
                "Estimated Fake Probability",
                f"{fake_probability:.2f}%"
            )

            st.info(
                "The model classified this posting as genuine. "
                "Always verify the employer independently."
            )

st.divider()

# Footer
st.markdown(
    "<center>AI Fake Job Detector | Machine Learning Project</center>",
    unsafe_allow_html=True
)
