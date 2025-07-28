# Connect 4 RL Bot 🔴🟡  
*A hybrid AI agent that plays a modified version of Connect 4 with hidden cells, combining symbolic search and deep reinforcement learning.*

## 🎯 Goal  
To build a reinforcement learning (RL) agent that can **beat a traditional Minimax algorithm** in a **partially observable Connect 4 environment** (i.e., where certain cells are blocked or hidden), introducing a new framework for decision-making under uncertainty.

## 🕹️ What is Connect 4?  
A classic 2-player game played on a 7×6 board. Players alternate dropping colored discs into a column, and the first to connect four in a row—horizontally, vertically, or diagonally—wins.

### 🧩 Our Twist:
- **Blocked cells** are introduced randomly each game
- Players must adapt to an **imperfectly observable** environment
- Traditional deterministic strategies (e.g., Minimax) are no longer optimal

## 📚 Background & Prior Work

### Classical Methods:
- **Minimax with Alpha-Beta Pruning**: Strong in perfect-information settings  
  - Assumes full visibility of the board  
  - Proven optimal in standard Connect 4 (Allis, 1994)

### Deep Reinforcement Learning:
- **AlphaZero (DeepMind)**: MCTS + Deep CNNs via self-play  
  - Excellent in games with fixed rules and visibility  
  - Not designed for dynamic environments like ours

### Imperfect Information Strategies:
- **Libratus (Poker AI)**: Counterfactual regret minimization  
  - Inapplicable to spatial board games like Connect 4  
  - Strong in bluff-based and betting scenarios

## 💡 Motivation & Contribution  
- Most real-world tasks involve **partial observability** and **uncertainty**  
- We combine **probabilistic search** with **policy-based RL**  
- First known attempt to apply this hybrid model to Connect 4 with occluded cells

### Use Cases:
- Autonomous systems  
- Financial markets  
- Game AI for multiplayer or hidden-information games

## 🧠 Agents Overview

### 🧮 Minimax Agent
- Classical search algorithm enhanced with:
  - Win detection  
  - Center control  
  - Piece height heuristics  
  - Opponent move blocking  
- Modified to estimate cell probabilities when blocked
- Strengths:
  - Deterministic and reliable
- Limitations:
  - Computationally expensive with deeper trees  
  - No ability to learn from experience

### 🧠 Neural Network Agent
- Board encoded as 3-channel image:
  - Channel 1: Player 1 tokens  
  - Channel 2: Player 2 tokens  
  - Channel 3: Hidden (blocked) cells
- CNN trained using **REINFORCE (Policy Gradient)**:
  - Rewards: +1 (win), −1 (loss), 0 (draw)
  - Trained via self-play vs. random opponents
- Strengths:
  - Learns over time  
  - Generalizes across layouts
- Limitations:
  - Slow convergence  
  - Sensitive to architecture and hyperparameters

## 🛠️ Project Timeline (Sprints)
1. Build core Connect 4 engine with support for blocked cells  
2. Implement Minimax with alpha-beta pruning  
3. Adapt Minimax to estimate hidden cell values  
4. Train CNN on masked game states  
5. Run tournaments and evaluate robustness  
6. Finalize results and compile report/presentation

## 📈 Baseline Performance  
- **Minimax vs Random Agent** (with 4 blocked cells):  
  **Win Rate: 80%**

## 🧪 Future Work
- Train RL agent against Minimax directly  
- Increase complexity (e.g., dynamic or probabilistically shifting occlusions)  
- Benchmark against other POMDP algorithms  
- Expand to multiplayer or variable-grid environments

## 💻 Requirements
- Python ≥ 3.8  
- NumPy  
- PyTorch  
- Matplotlib

## ✍️ Authors  
Venora Furtado  
Tejaswi Tripathi  
