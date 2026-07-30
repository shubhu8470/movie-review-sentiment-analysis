import streamlit as st
import pandas as pd
import os

try:
    import joblib
except ImportError:
    st.error(
        "joblib is not installed. "
        "Please make sure requirements.txt contains 'joblib>=1.3.2'."
    )
    st.stop()
# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# =====================================
# Load Model
# =====================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("❌ Model files not found!")

    st.info(
        """
Please train the model first.

Open your terminal and run:

python train.py

After training, run:

streamlit run app.py
"""
    )
    st.stop()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# =====================================
# Header
# =====================================
st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    """
Predict whether a movie review is **Positive 😊** or **Negative 😞**
using Machine Learning (TF-IDF + Logistic Regression).
"""
)

# =====================================
# Input
# =====================================
review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: The movie was absolutely fantastic!",
    height=180,
)

# =====================================
# Predict
# =====================================
if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        confidence = model.predict_proba(review_vector).max()

        if prediction == 1:
            st.success("😊 Positive Review")
            st.balloons()
        else:
            st.error("😞 Negative Review")

        st.subheader("Confidence")

        st.progress(float(confidence))

        st.write(f"**{confidence*100:.2f}%**")

# =====================================
# Examples
# =====================================
st.markdown("---")

st.subheader("Try these examples")

examples = [
    "The movie was absolutely fantastic.",
    "I loved the acting and storyline.",
    "The film was boring and a waste of time.",
    "The direction was amazing.",
    "Worst movie I have ever watched."
]

for ex in examples:
    st.code(ex)

# =====================================
# Sidebar
# =====================================
st.sidebar.title("About Project")

st.sidebar.write("""
### Technologies

- Python
- Streamlit
- Scikit-Learn
- TF-IDF
- Logistic Regression
- Joblib

### Developer

Movie Review Sentiment Analysis

Machine Learning Project
""")