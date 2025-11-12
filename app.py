import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle

# ------------------------------
# Load model and scaler
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Fake Instagram Profile Detector",
    page_icon="🤖",
    layout="centered"
)

# ------------------------------
# Custom CSS Styling
# ------------------------------
st.markdown("""
    <style>
    /* General page style */
    .main {
        background-color: #121212;
        color: #e0e0e0;
        font-family: "Segoe UI", sans-serif;
    }

    /* Input widget styling */
    .stTextInput > div > div > input, 
    .stNumberInput input, 
    .stTextArea textarea, 
    .stSelectbox > div > div > div {
        background-color: #1e1e1e !important;
        color: #f5f5f5 !important;
        border-radius: 8px;
        border: 1px solid #444;
        padding: 8px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        transform: scale(1.02);
    }

    /* Titles and headers */
    h1, h2, h3, h4 {
        color: #f3f4f6;
    }

    /* Subheader box styling */
    .result-box {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Title
# ------------------------------
st.title("🤖 Fake Instagram Profile Detection")
st.caption("A dark-themed AI-powered app to predict if an Instagram account is fake or real.")

# ------------------------------
# Input Form
# ------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        username = st.text_input("👤 Username", value="john_doe_123")
        fullname = st.text_input("🧾 Full Name", value="John Doe")
        description = st.text_area("📝 Profile Description / Bio", value="Travel lover ✈️ | Coffee addict ☕")
        profile_pic = st.selectbox("🖼️ Profile Picture Present?", ["Yes", "No"])
        external_url = st.selectbox("🔗 External URL Present?", ["Yes", "No"])
        private = st.selectbox("🔒 Is Profile Private?", ["Yes", "No"])

    with col2:
        posts = st.number_input("📸 Number of Posts", min_value=0, max_value=10000, value=50)
        followers = st.number_input("👥 Number of Followers", min_value=0, max_value=10000000, value=1000)
        follows = st.number_input("➡️ Number of Following", min_value=0, max_value=10000, value=500)

    submit = st.form_submit_button("🔍 Predict")

# ------------------------------
# Backend Feature Calculation
# ------------------------------
if submit:

    def count_numbers_ratio(text):
        text = str(text)
        return sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0

    def count_words(text):
        return len(str(text).strip().split())

    # Derived text-based features
    nums_len_username = count_numbers_ratio(username)
    fullname_words = count_words(fullname)
    nums_len_fullname = count_numbers_ratio(fullname)
    name_eq_username = 1 if fullname.lower().replace(" ", "") == username.lower() else 0
    description_len = len(description)

    # Convert dropdowns to binary
    def to_binary(x): return 1 if x == "Yes" else 0
    profile_pic = to_binary(profile_pic)
    external_url = to_binary(external_url)
    private = to_binary(private)

    # Ratios
    eps = 1e-6
    followers_to_follows = followers / (follows + eps)
    posts_per_1k_followers = posts / ((followers + eps) / 1000)
    log1p_followers = np.log1p(followers)
    log1p_follows = np.log1p(follows)
    log1p_posts = np.log1p(posts)

    # DataFrame for prediction
    input_data = pd.DataFrame({
        'profile pic': [profile_pic],
        'nums/length username': [nums_len_username],
        'fullname words': [fullname_words],
        'nums/length fullname': [nums_len_fullname],
        'name==username': [name_eq_username],
        'description length': [description_len],
        'external URL': [external_url],
        'private': [private],
        'posts': [posts],
        'followers': [followers],
        'follows': [follows],
        'followers_to_follows': [followers_to_follows],
        'posts_per_1k_followers': [posts_per_1k_followers],
        'log1p_followers': [log1p_followers],
        'log1p_follows': [log1p_follows],
        'log1p_posts': [log1p_posts]
    })

    # Scale numeric columns
    num_cols = input_data.select_dtypes(include=[np.number]).columns
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display Results
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.subheader("🧩 Auto-Calculated Features")
    st.write({
        "Numbers/Length in Username": round(nums_len_username, 3),
        "Full Name Words": fullname_words,
        "Numbers/Length in Full Name": round(nums_len_fullname, 3),
        "Description Length": description_len,
        "Followers/Follows Ratio": round(followers_to_follows, 3),
        "Posts per 1K Followers": round(posts_per_1k_followers, 3)
    })

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This profile is predicted to be **FAKE** with probability {probability:.2f}")
    else:
        st.success(f"✅ This profile is predicted to be **REAL** with probability {1 - probability:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Model: Logistic Regression trained on Kaggle Fake Instagram Profile dataset")
