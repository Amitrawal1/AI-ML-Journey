# 📝 Text Summarizer using T5 Transformer

A deep learning based **Abstractive Text Summarization** application built using **T5 (Text-To-Text Transfer Transformer)**, **PyTorch**, **Hugging Face Transformers**, and **FastAPI**. The project fine-tunes the **T5-Small** model on the **SAMSum** dataset and provides an interactive web interface for generating concise summaries from long conversations or text. :contentReference[oaicite:0]{index=0}

---

## 🎥 Demo

| Web Interface | Generated Summary |
|---------------|-------------------|
| Paste long conversations or articles | Generate concise summaries using the trained T5 model |

> Add screenshots or GIFs of your application here.

---

# ✨ Features

- 🤖 Fine-tuned T5-Small Transformer model
- 🧹 Automatic text preprocessing
- ✂️ Abstractive text summarization
- ⚡ FastAPI REST API
- 🎨 Responsive HTML/CSS frontend
- 📊 Compression percentage indicator
- 📋 One-click summary copy
- 💻 Apple Silicon (MPS), CUDA and CPU support
- 💾 Load locally fine-tuned model
- 🔄 Real-time inference using Fetch API

---

# 📂 Project Structure

```text
TEXTSUMMERIZER/
│
├── data/
│   ├── samsum-train.csv
│   └── samsum-validation.csv
│
├── saved_summary_model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── generation_config.json
│   └── model.safetensors
│
├── app.py                  # FastAPI Backend
├── index.html              # Frontend UI
├── txt_summarize.py         # Model Training
├── requirements.txt
└── README.md
```

---

# 🧠 Model Architecture

The project uses **T5 (Text-to-Text Transfer Transformer)** for abstractive text summarization.

### Model Used

- T5-Small
- Encoder-Decoder Transformer
- Hugging Face Transformers
- PyTorch

---

# 📚 Dataset

The model is fine-tuned on the **SAMSum Dialogue Summarization Dataset**.

Dataset contains:

- Human conversations
- Corresponding abstractive summaries

Training Pipeline:

- Load dataset
- Clean text
- Tokenization
- Fine-tuning
- Save trained model
- Perform inference :contentReference[oaicite:1]{index=1}

---

# 🧹 Data Preprocessing

Before training, every dialogue passes through:

- Remove line breaks
- Remove HTML tags
- Convert to lowercase
- Remove unnecessary whitespace :contentReference[oaicite:2]{index=2}

---

# ⚙️ Training Configuration

Training is performed using Hugging Face Trainer.

Main parameters:

| Parameter | Value |
|-----------|------:|
| Model | T5-Small |
| Epochs | 6 |
| Batch Size | 8 |
| Max Input Length | 512 |
| Max Summary Length | 150 |
| Weight Decay | 0.01 |
| Warmup Steps | 500 |
| Evaluation Strategy | Every Epoch |
| Save Strategy | Every Epoch | :contentReference[oaicite:3]{index=3}

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/TEXTSUMMERIZER.git
```

Move into the project

```bash
cd TEXTSUMMERIZER
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

Run

```bash
python txt_summarize.py
```

The script will

- Load the SAMSum dataset
- Preprocess dialogues
- Tokenize data
- Fine-tune T5-Small
- Save the trained model locally :contentReference[oaicite:4]{index=4}

Saved model:

```text
saved_summary_model/
```

---

# ▶️ Run the Web Application

Start the FastAPI server

```bash
uvicorn app:app --reload
```

Open your browser

```
http://127.0.0.1:8000
```

---

# 🌐 API Endpoint

### Generate Summary

```
POST /summarize/
```

Request

```json
{
    "dialogue": "Your long conversation or text..."
}
```

Response

```json
{
    "summary": "Generated summary."
}
```

---

# 🖥️ Device Support

The application automatically selects the best available hardware.

Priority

```
Apple MPS
↓
CUDA GPU
↓
CPU
```

No manual configuration is required. :contentReference[oaicite:5]{index=5}

---

# 🔄 Project Workflow

```text
Input Dialogue
        │
        ▼
Text Cleaning
        │
        ▼
Tokenization
        │
        ▼
Fine-Tuned T5 Model
        │
        ▼
Beam Search Generation
        │
        ▼
Decode Tokens
        │
        ▼
Generated Summary
        │
        ▼
Display on Web UI
```

---

# 🎨 Frontend Features

- Modern responsive interface
- Source text counter
- Summary counter
- Compression percentage
- Copy summary button
- AJAX-based summarization (no page reload)
- Loading state while generating summaries :contentReference[oaicite:6]{index=6}

---

# 📚 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- FastAPI
- Jinja2
- HTML5
- CSS3
- JavaScript (Fetch API)
- Pandas
- Regular Expressions

---

# 📈 Future Improvements

- Support larger T5 variants (Base / Large)
- ROUGE score evaluation
- Batch summarization
- PDF & DOCX summarization
- Upload text files
- Multi-language summarization
- Streaming responses
- Docker deployment
- Hugging Face Spaces deployment
- User authentication

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ If you found this project helpful, consider giving it a star!