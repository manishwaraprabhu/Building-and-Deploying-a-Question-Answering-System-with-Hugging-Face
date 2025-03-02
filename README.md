**Project Overview**
This project implements a Question Answering (QA) System using Hugging Face's Transformers and Streamlit. We fine-tuned a DistilBERT model for extractive question answering using the SQuAD dataset and deployed it as an interactive Streamlit web application.

**Project Workflow**
✅ Step 1: Dataset Preparation
🔹 Loaded and preprocessed the SQuAD dataset for fine-tuning.
🔹 Extracted context, questions, and answers, ensuring correct start-end token mappings.
🔹 Converted the dataset into the Hugging Face dataset format for training.
🔹 Saved the processed dataset as train_data.csv and validation_data.csv for further use.

✅ Step 2: Baseline Model (Pretrained DistilBERT Evaluation)
🔹 Loaded train_data.csv and validation_data.csv for model evaluation.
🔹 Loaded pretrained DistilBERT (distilbert-base-cased) for zero-shot question answering.
🔹 Evaluated the model on the validation dataset before fine-tuning.
🔹 Measured initial Exact Match (EM) and F1 scores as benchmarks.
🔹 Observed that the pretrained model struggled with domain-specific questions.
🔹 Saved the baseline model as baseline_model for further fine-tuning.

✅ Step 3: Fine-Tuning the DistilBERT Model
🔹 Loaded the previously saved baseline_model for fine-tuning.
🔹 Fine-tuned DistilBERT on the SQuAD dataset using the Hugging Face Trainer API.
🔹 Used AdamW optimizer and learning rate scheduling for optimal training.
🔹 Set batch size, epochs, and gradient accumulation to balance performance.
🔹 Saved the best fine-tuned model as best_fine_tuned_model after training.

✅ Step 4: Model Evaluation
🔹 Compared the baseline vs fine-tuned model performance.
🔹 Evaluated both models using SQuAD metrics (EM & F1 score).
🔹 Achieved improved EM: 74.50% and F1: 83.07% after fine-tuning.
🔹 Saved incorrect predictions for further error analysis & improvements.
🔹 Visualized performance gains using matplotlib bar charts.
🔹 Generated an evaluation report (evaluation_report.json) with key metrics.

✅ Step 5: Deployment via Streamlit
🔹 Built a Streamlit web application for real-time question answering.
🔹 Loaded the fine-tuned model (best_fine_tuned_model) for inference.
🔹 Implemented context input, question input, and answer retrieval.
🔹 Handled edge cases (invalid inputs, model errors, and incorrect answers).
🔹 Tested the app with real-world queries, ensuring usability & performance.

**Final Results: Performance Comparison**   	
Baseline Model	 -   Exact Match (EM): 71.75%,	   F1 Score: 80.65%
Fine-Tuned Model -   Exact Match (EM): 74.50%,	   F1 Score: 83.07%
🔹 Fine-tuning improved model accuracy by ~3% (EM) and ~2.5% (F1 Score).
🔹 Deployment successfully enables real-time QA interaction via a web UI.

**Conclusion**
This project demonstrates the end-to-end process of training, fine-tuning, evaluating, and deploying a Question Answering model using Hugging Face Transformers, PyTorch, and Streamlit. By fine-tuning DistilBERT on SQuAD, we successfully improved its performance and built an interactive QA system for real-time inference. 
