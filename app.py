import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import torch.nn.functional as F

# Load tokenizer and model
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return tokenizer, model

# Improved Question Answering Function
def answer_question(question, context, tokenizer, model):
    try:
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding="max_length", max_length=384)

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        start_probs, end_probs = F.softmax(start_logits, dim=1), F.softmax(end_logits, dim=1)

        # Get top start and end indices
        start_index = torch.argmax(start_probs).item()
        end_index = torch.argmax(end_probs).item() + 1  

        # Filter out low-confidence predictions
        if start_probs[0, start_index] < 0.3 or end_probs[0, end_index - 1] < 0.3:
            return "I'm not confident in my answer."

        # Avoid long spans that include unnecessary text
        if end_index - start_index > 10:  # Adjust max answer length
            return "The answer is too uncertain to extract precisely."

        answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index], skip_special_tokens=True)

        return answer.strip()
    
    except Exception as e:
        return f"Error: {str(e)}"

# Load fine-tuned model and tokenizer
MODEL_PATH = "C:/Users/manis/LASTPROJECT/best_fine_tuned_model"
tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

# Streamlit App
st.title("Question Answering System")
st.write("Provide a context and ask a question to get the answer.")

# User Inputs
context = st.text_area("Enter the context:", height=200)
question = st.text_input("Enter your question:")

# Generate Answer
if st.button("Get Answer"):
    if not context.strip() or not question.strip():
        st.warning("Please provide both a context and a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = answer_question(question, context, tokenizer, model)
            if answer:
                st.success(f"Answer: {answer}")
            else:
                st.error("Sorry, I couldn't find an answer to your question.")

# Sidebar
st.sidebar.title("About")
st.sidebar.write("This app uses a fine-tuned Hugging Face model for QA.")