import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

st.set_page_config(page_title="Eco Lifestyle Agent 🌱", layout="centered")
st.title("🌱 Eco Lifestyle Agent")
st.markdown("Ask me anything about sustainable living, eco-friendly habits, and green tips!")

# Input from user
user_input = st.text_input("You:", placeholder="How can I reduce water waste?")

if user_input:
    with st.spinner("Thinking eco-friendly thoughts..."):
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(response)
