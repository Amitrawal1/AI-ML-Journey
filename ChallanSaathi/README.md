# ⚖️ ChallanSaathi — Indian Motor Vehicle Law Assistant

Build an AI-powered **Indian Motor Vehicle Law Assistant** using **Hybrid Retrieval-Augmented Generation (RAG)**. NyayaDrive retrieves relevant provisions from real Indian motor vehicle law documents using **Vector Search + BM25** and generates simple, source-grounded answers using a local **LLM with Ollama**.

---

## 🚗 Demo

| Legal Question | NyayaDrive |
|----------------|------------|
| Ask a motor vehicle law question | Retrieve relevant legal provisions |
| Ask about Haryana / UP rules | Retrieve state-specific rules |
| Ask about a Rule / Section | Find the relevant legal provision |
| Ask in simple language | Generate a simple answer with citations |

> Add your screenshots or demo GIF here.

---

## ✨ Features

- ⚖️ Indian Motor Vehicle Law Question Answering
- 🔎 Hybrid Retrieval using Vector Search + BM25
- 🧠 Semantic Search using embeddings
- 📚 Real motor vehicle law documents in PDF format
- 🇮🇳 State-specific legal retrieval
- 🏷️ Metadata-based filtering
- 📑 Rule and Section-aware retrieval
- 🤖 Local LLM inference using Ollama
- 📌 Source and page-level citations
- 🛡️ Grounded answers using retrieved legal sources
- 🚫 Prevents the LLM from inventing legal provisions
- 🧩 Modular RAG pipeline

---

## 📂 Project Structure

```text
NyayaDrive/
│
├── data/
│   └── raw/
│       ├── HARYANA.pdf
│       └── UP.pdf
│
├── notebooks/
│   └── rag_pipeline.ipynb
│   
│
├── vectorstore/
│
├── requirements.txt
