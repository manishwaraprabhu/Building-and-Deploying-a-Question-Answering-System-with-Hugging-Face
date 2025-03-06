import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load tokenizer and model
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return tokenizer, model

# Function for question answering
def answer_question(question, context, tokenizer, model):
    try:
        # Tokenize the input question and context
        inputs = tokenizer(
            question, 
            context, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=512  # Adjust if needed
        )
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the start and end positions of the answer
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely start and end positions
        start_index = torch.argmax(start_scores).item()
        end_index = torch.argmax(end_scores).item() + 1

        # Validate indices
        if start_index >= len(inputs['input_ids'][0]) or end_index > len(inputs['input_ids'][0]) or start_index >= end_index:
            return "Sorry, I couldn't find an answer to your question."

        # Decode the answer
        answer_tokens = inputs['input_ids'][0][start_index:end_index]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Return the answer
        return answer.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

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
            # Call the function to get the answer
            answer = answer_question(question, context, tokenizer, model)
            if answer:
                st.success(f"Answer: {answer}")
            else:
                st.error("Sorry, I couldn't find an answer to your question.")

st.sidebar.title("About")
st.sidebar.write(
    "This application uses a fine-tuned model for Question Answering built with Hugging Face Transformers."
)
