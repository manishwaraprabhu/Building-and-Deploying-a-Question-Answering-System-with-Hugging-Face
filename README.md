# 🧠 Building and Deploying a Question Answering System with Hugging Face

## 📌 Project Overview  
This project implements an **Extractive Question Answering (QA) System** using Hugging Face Transformers and Streamlit. A pre-trained **DistilBERT** model was fine-tuned on the **SQuAD** dataset to improve accuracy in extracting answers from a given context. The final model was deployed as an interactive **Streamlit** web application for real-time user interaction.

---

## 🚀 Project Workflow

### ✅ Step 1: Dataset Preparation
- Loaded and preprocessed the **SQuAD dataset** for fine-tuning.
- Extracted **context, questions, and answers**, ensuring accurate start-end token mappings.
- Converted the data into Hugging Face `Dataset` format for training.
- Saved processed data as `train_data.csv` and `validation_data.csv` for reuse.

---

### ✅ Step 2: Baseline Model Evaluation (Pretrained DistilBERT)
- Loaded `train_data.csv` and `validation_data.csv` for model benchmarking.
- Used **DistilBERT (distilbert-base-cased)** for zero-shot QA evaluation.
- Evaluated performance before fine-tuning to establish baseline metrics.
- Observed limitations in handling domain-specific or complex queries.
- Saved the baseline model as `baseline_model`.

---

### ✅ Step 3: Fine-Tuning the DistilBERT Model
- Loaded the `baseline_model` and fine-tuned it using the Hugging Face **Trainer API**.
- Used **AdamW optimizer** with learning rate scheduling.
- Configured batch size, epochs, and gradient accumulation for optimal performance.
- Trained the model on SQuAD dataset and saved the best version as `best_fine_tuned_model`.

---

### ✅ Step 4: Model Evaluation
- Compared **baseline vs fine-tuned** model performance.
- Evaluated using **Exact Match (EM)** and **F1 Score**.
- Achieved significant improvements post fine-tuning:
  - **EM**: 74.50%
  - **F1**: 83.07%
- Generated `evaluation_report.json` summarizing all key metrics.
- Visualized improvement using bar charts via `matplotlib`.
- Saved incorrect predictions for further error analysis.

---

### ✅ Step 5: Deployment via Streamlit
- Developed a **Streamlit web app** for real-time QA interaction.
- Integrated `best_fine_tuned_model` to power the backend inference.
- Enabled users to input both **context** and **question** for on-the-fly answers.
- Implemented robust error handling for invalid inputs and edge cases.
- Tested thoroughly with real-world examples for stability and usability.

---

## 📊 Final Results: Performance Comparison

| Model            | Exact Match (EM) | F1 Score  |
|------------------|------------------|-----------|
| Baseline Model   | 71.75%           | 80.65%    |
| Fine-Tuned Model | **74.50%**       | **83.07%**|

✅ Fine-tuning improved **EM by ~3%** and **F1 Score by ~2.5%**  
✅ Real-time QA interaction successfully enabled through the deployed UI

---

## ✅ Conclusion  
This project demonstrates the **complete lifecycle of building an NLP solution**—from **dataset preparation** and **model fine-tuning** to **deployment and evaluation**. Using Hugging Face Transformers and PyTorch, we improved a DistilBERT model’s performance on SQuAD and deployed it via **Streamlit**, making it accessible for real-time use cases such as knowledge assistants, intelligent search engines, and chatbot integrations.

---

## 🛠️ Technologies & Tools
- Hugging Face Transformers
- PyTorch
- Hugging Face Datasets
- Streamlit
- SQuAD Dataset
- Matplotlib, Pandas
- JSON, CSV, Tokenizers

---

## 📂 Project Structure

├── data/
│   ├── train_data.csv
│   └── validation_data.csv
├── models/
│   ├── baseline_model/
│   └── best_fine_tuned_model/
├── evaluation/
│   ├── evaluation_report.json
│   └── performance_plots.png
├── app.py  # Streamlit Application
├── train.py  # Fine-tuning script
├── requirements.txt
└── README.md

---

## 📖 Sample Dataset Entry
```json
{
  "context": "The Amazon rainforest is one of the world's most biodiverse habitats. It plays a critical role in regulating the global climate.",
  "question": "What role does the Amazon rainforest play in the climate?",
  "answer": "regulating the global climate"
}

---

## 📚 Learning Outcomes

- Fine-tuning pre-trained transformer models for extractive QA  
- Understanding tokenization and label alignment for QA  
- Evaluating NLP models using EM and F1 metrics  
- Building and deploying interactive ML apps using Streamlit  
- Exposure to Hugging Face's ecosystem and Trainer API
