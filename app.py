
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# UI Setup
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #ff6347;'>🧠 Fake News Detector 🔍</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a news article below to check if it's real or fake!</p>", unsafe_allow_html=True)

# Input text
user_input = st.text_area("📝 Paste News Article Text Here", height=300)

# Predict button
if st.button("Check News ✅"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]
        
        if prediction == "FAKE":
            st.error("❌ This news is **Fake**!", icon="🚨")
        else:
            st.success("✅ This news appears to be **Real**.", icon="📰")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
