# 🐦 Flappy Bird AI using Deep Q-Network (DQN)

Train an AI agent to play Flappy Bird using **Deep Reinforcement Learning (DQN)** with **PyTorch** and **Gymnasium**. The project also includes a manual mode where you can play the game using the **Spacebar**.

---

## 🎮 Demo

| AI Training | Manual Gameplay |
|-------------|-----------------|
| Train a DQN agent to maximize rewards | Play Flappy Bird yourself using the keyboard |

> Add your gameplay GIF or screenshots here.

---

## ✨ Features

- 🧠 Deep Q-Network (DQN) implementation
- 🎯 Experience Replay Memory
- 🔄 Target Network Synchronization
- 📉 Epsilon-Greedy Exploration Strategy
- ⚡ GPU (CUDA), Apple Silicon (MPS), and CPU support
- 💾 Automatic best model saving
- 🎮 Manual gameplay mode using keyboard
- ⚙️ Configurable hyperparameters using YAML

---

## 📂 Project Structure

```
Flappy_Bird_AI/
│
├── agent.py                # Train/Test DQN Agent
├── dqn.py                  # Neural Network Architecture
├── experience_replay.py    # Replay Memory Buffer
├── manual_play.py          # Play Flappy Bird manually
├── parameters.yaml         # Hyperparameters
├── runs/                   # Saved models & logs
│   ├── *.pt
│   └── *.log
│
├── requirements.txt
└── README.md
```

---

## 🧠 Reinforcement Learning Algorithm

The agent learns using the **Deep Q-Network (DQN)** algorithm.

### Techniques Used

- Deep Q-Learning
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
- Bellman Equation
- Mean Squared Error (MSE) Loss
- Adam Optimizer

---

## 🏗️ Neural Network

Input:
- Environment State (Observation)

Output:
- Q-values for each action

Actions:

| Action | Description |
|--------|-------------|
| 0 | Do Nothing |
| 1 | Flap |

---

## ⚙️ Hyperparameters

All training parameters are stored inside:

```
parameters.yaml
```

Example:

```yaml
alpha: 0.001
gamma: 0.99

epsilon_init: 1.0
epsilon_min: 0.05
epsilon_decay: 0.995

replay_memory_size: 100000
mini_batch_size: 64

reward_threshold: 1000
network_sync_rate: 1000
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Flappy_Bird_AI.git
```

Move into the project

```bash
cd Flappy_Bird_AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the AI

```bash
python agent.py default --train
```

The agent will

- Explore the environment
- Store experiences
- Learn from replay memory
- Save the best model automatically

Saved model:

```
runs/default.pt
```

Training log:

```
runs/default.log
```

---

## 🎮 Test the Trained AI

```bash
python agent.py default
```

The trained model will play Flappy Bird automatically.

---

## 🎮 Play Manually

Run

```bash
python manual_play.py
```

Controls

| Key | Action |
|-----|--------|
| Space | Flap |
| Close Window | Exit Game |

---

## 🖥️ Device Support

The project automatically selects the best available device.

Priority:

```
Apple MPS
↓
CUDA GPU
↓
CPU
```

No configuration is required.

---

## 📊 Training Workflow

```
Environment
      │
      ▼
 Current State
      │
      ▼
 Policy Network
      │
      ▼
Choose Action
      │
      ▼
Take Action
      │
      ▼
Receive Reward
      │
      ▼
Store Experience
      │
      ▼
Replay Memory
      │
      ▼
Sample Mini Batch
      │
      ▼
Update Policy Network
      │
      ▼
Sync Target Network
      │
      ▼
Repeat
```

---

## 📚 Technologies Used

- Python
- PyTorch
- Gymnasium
- flappy-bird-gymnasium
- PyGame
- YAML

---

## 📈 Future Improvements

- Double DQN (DDQN)
- Dueling DQN
- Prioritized Experience Replay
- Rainbow DQN
- TensorBoard Integration
- Model Evaluation Metrics
- Training Reward Graphs

---

## 🤝 Contributing

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

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ If you found this project helpful, consider giving it a star!
