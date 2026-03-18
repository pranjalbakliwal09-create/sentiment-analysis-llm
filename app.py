import streamlit as st
from transformers import pipeline

# Load your trained model
classifier = pipeline("sentiment-analysis", model="sentiment-model")

# UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖")

st.title("🤖 AI Sentiment Analyzer")
st.write("Enter text and see if it's Positive or Negative!")

text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']

        if label == "POSITIVE":
            st.success(f"Positive 😊 (Confidence: {score:.2f})")
        else:
            st.error(f"Negative 😠 (Confidence: {score:.2f})")