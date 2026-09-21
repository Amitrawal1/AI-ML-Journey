# 🚀 AI-ML Journey

Welcome to my **AI/ML Journey** repository.

This repository documents my hands-on learning and development in:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Reinforcement Learning
- Natural Language Processing
- Generative AI
- Retrieval-Augmented Generation (RAG)
- AI Agents

The goal is simple:

- 📚 Learn by building
- 🧠 Understand concepts through implementation
- 🚀 Build practical and production-oriented AI projects
- 🌍 Keep projects open source and useful for the community

This repository represents my journey from learning individual AI/ML concepts to building complete, practical systems.

---

# 📂 Projects

| # | Project | Description | Technologies |
|---|---|---|---|
| 1 | 🔗 **[Flappy Bird AI](./Project-1-Flappy-Bird)** | Deep Reinforcement Learning agent that learns to play Flappy Bird autonomously using a Deep Q-Network (DQN). | Python, PyTorch, Gymnasium, DQN |
| 2 | 🔗 **[Optimal Crops Predicting](./Project-2-Optimal-Crops-Predicting)** | Machine Learning model that recommends suitable crops based on soil and environmental conditions. | Python, Scikit-Learn, Pandas |
| 3 | 🔗 **[Power Plant Prediction](./Project-3-Power-Plant-Prediction)** | Machine Learning model that predicts power plant energy output using supervised regression techniques. | Python, Scikit-Learn, Pandas |
| 4 | 🔗 **[CNN CIFAR-10](./Project-4-CNN-CIFAR10)** | Image classification system trained on the CIFAR-10 dataset using Convolutional Neural Networks. | Python, PyTorch, CNN, Deep Learning |
| 5 | 🔗 **[Text Summarizer](./Project-5-Text-Summarizer)** | NLP-based text summarization system that fine-tunes a T5 Transformer on the SAMSum dataset to generate concise conversational summaries. | Python, Hugging Face Transformers, NLP, T5 |
| 6 | 🔗 **[ChallanSaathi](./Project-6-ChallanSaathi)** | Hybrid RAG-based Indian Motor Vehicle Law Assistant that retrieves relevant provisions and generates grounded answers with rule/section references and source citations. | Python, RAG, BM25, Vector Search, Llama 3.1, Ollama |
| 7 | 🔗 **[Rocket Landing AI](./Project-7-Rocket-Landing-AI)** | Reinforcement Learning agent that learns to control and land a reusable rocket booster on a landing pad using a custom physics simulation. | Python, PPO, Gymnasium, PyTorch, Pygame |

---

# 🚀 Project 7 — Rocket Landing AI

A Reinforcement Learning project where an AI agent learns to control and land a reusable rocket booster.

The agent is not explicitly programmed with a landing procedure. Instead, it learns through repeated simulated flights using rewards and penalties.

The simulation models important rocket-flight concepts including:

- Gravity
- Engine thrust
- Fuel consumption
- Air drag
- Rocket orientation
- Grid fins
- Landing legs
- Horizontal and vertical velocity
- Landing conditions

The agent controls:

- 🚀 Engine throttle
- ↔️ Rocket lean
- 🦿 Landing-leg deployment

The project uses **PPO (Proximal Policy Optimization)** for reinforcement learning and curriculum learning to gradually increase the difficulty of training. :contentReference[oaicite:1]{index=1}

### 🎥 Demo

[▶️ Watch the Rocket Landing AI Demo](https://www.youtube.com/watch?v=f0YVHjock84)

### 📊 Results

The trained model was evaluated on **200 randomized landing attempts**.

| Result | Performance |
|---|---:|
| 🎯 Landed on landing pad | **65%** |
| 🛬 Landed safely beside pad | **32.5%** |
| 💥 Crashed | **2.5%** |
| 📐 Landing tilt | **~1°** |

The final evaluation used a 90 m landing pad with randomized starting conditions. :contentReference[oaicite:2]{index=2}

### 🧠 Learning Approach

The training uses a curriculum-learning strategy:

```text
Easy Landing
     ↓
Large Landing Pad
     ↓
Increase Difficulty
     ↓
Smaller Landing Pad
     ↓
Fine-Tuning
     ↓
More Difficult Landings