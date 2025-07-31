# Genetic Mutation Environment Reinforcement Learning 🧬🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stable Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![Pygame](https://img.shields.io/badge/pygame-2.0+-red.svg)](https://www.pygame.org/)

A comprehensive reinforcement learning project that optimizes agent behavior in a simulated genetic mutation testing environment. The agent must efficiently explore a spatial grid representing mutations with varying quality scores while managing a limited budget for testing.

## 🎯 Project Overview

This project addresses the challenge of **balancing exploration and exploitation** in a constrained environment where:
- An agent moves on a 5×5 grid of genetic mutations
- Each mutation has a quality score (EVO2) between 0-1
- The agent has a limited budget (initially 100) for testing
- Testing high-quality mutations yields higher rewards
- The goal is to maximize cumulative rewards through strategic exploration

## 🚀 Key Features

- **Multiple RL Algorithms**: DQN, PPO, and A2C implementations
- **Enhanced Visualization**: Real-time pygame rendering with animations
- **Comprehensive Analysis**: Performance metrics, convergence studies, and generalization testing
- **Professional Documentation**: Detailed reports and analysis
- **Demo Capabilities**: GIF generation and interactive demonstrations

## 🏗️ Environment Architecture

### Agent Capabilities
- **Movement**: Navigate in 4 directions (up, down, left, right)
- **Testing**: Analyze mutations at current position (consumes budget)
- **Budget Management**: Strategic resource allocation

### Action Space (Discrete)
- `0`: Test Mutation (uses budget, yields reward based on quality)
- `1`: Move Up (2 units)
- `2`: Move Down (2 units)
- `3`: Move Left (2 units)
- `4`: Move Right (2 units)

### State Space (Continuous)
- Current mutation's EVO2 score (0-1)
- Normalized remaining budget (0-1)
- Mutation type placeholder (extensible)

### Reward Structure
```python
Testing Rewards:
  High-value mutation (EVO2 > 0.8): +10
  Medium-value mutation (EVO2 > 0.5): +5
  Low-value mutation (≤ 0.5): -2
  
Movement: 0 reward
Budget depletion: -10 (episode termination)
```

## 📊 Algorithm Performance

| Algorithm | Final Performance | Convergence Episodes | Generalization |
|-----------|------------------|---------------------|----------------|
| **PPO** | **869.1 ± 36.9** | **5** | **88.8%** |
| DQN | 575.8 ± 256.2 | 52 | 75.9% |
| A2C | 453.8 ± 138.6 | 20 | 78.9% |

**🏆 PPO achieved the best overall performance** with fastest convergence and superior generalization capabilities.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
```bash
pip install stable-baselines3[extra]
pip install gymnasium
pip install pygame
pip install numpy matplotlib
pip install imageio Pillow  # For GIF generation
```

### Quick Install
```bash
git clone <repository-url>
cd RL_formative
pip install -r requirements.txt  # If available
```

## 🎮 Usage

### 1. Environment Demonstration

**Interactive Demo:**
```bash
python3 implementations/demo_random_agent.py
# Choose option 2 for live visualization
```

**Generate GIF:**
```bash
python3 implementations/demo_random_agent.py
# Choose option 1 to create animated GIF
```

### 2. Training Agents

**Train PPO (Recommended):**
```python
from stable_baselines3 import PPO
from genetic_mutation_env import GeneticMutationEnv

env = GeneticMutationEnv()
model = PPO("MlpPolicy", env, learning_rate=0.0003, batch_size=64)
model.learn(total_timesteps=100000)
model.save("ppo_genetic_mutation")
```

**Train DQN:**
```python
from stable_baselines3 import DQN

model = DQN("MlpPolicy", env, learning_rate=0.0003, buffer_size=10000)
model.learn(total_timesteps=50000)
```

**Train A2C:**
```python
from stable_baselines3 import A2C

model = A2C("MlpPolicy", env, learning_rate=0.0007, n_steps=5)
model.learn(total_timesteps=10000)
```

### 3. Testing Trained Models

```python
# Load and test a trained model
model = PPO.load("ppo_genetic_mutation")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
```

### 4. Generate Analysis Visualizations

```bash
# Generate comprehensive analysis
python3 visualizations_scripts/generate_visualizations.py
python3 visualizations_scripts/convergence_generalization_analysis.py
```

## 📁 Project Structure

```
RL_formative/
├── README.md                           # This file
├── report.md                           # Comprehensive project report
├── Environment_Visualization_Report.md # Visualization documentation
├── main.py                             # Main entry point
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
│
├── environments/                       # Environment implementations
│   ├── genetic_env.py                 # Basic genetic mutation environment
│   └── enhanced_genetic_env.py        # Enhanced environment with visualization
│
├── implementations/                    # Training implementations
│   ├── demo_random_agent.py           # Interactive demonstration script
│   ├── rendering.py                   # Rendering utilities
│   └── training/                      # Training scripts and utilities
│       ├── [training scripts]         # Algorithm-specific training files
│       └── models/                    # Additional model storage
│
├── SB3/                               # Stable Baselines3 models and results
│   └── models/                        # Trained model storage
│       ├── dqn/
│       │   └── genetic_dqn.zip       # Trained DQN model
│       ├── pg/                        # Policy Gradient models
│       │   ├── ppo.zip               # Trained PPO model
│       │   └── a2c.zip               # Trained A2C model
│       └── random_agent_demo.gif     # Demo visualization
│
├── visualizations_scripts/            # Analysis and plotting scripts
│   ├── generate_visualizations.py     # Main visualization generator
│   ├── convergence_generalization_analysis.py # Convergence analysis
│   ├── create_simulation_gif.py       # GIF creation utilities
│   └── plots.py                       # Basic plotting functions
│
├── artifacts_&_images/                # Generated analysis artifacts
│   ├── cumulative_rewards.png         # Reward progression plots
│   ├── training_stability.png         # Training stability analysis
│   ├── convergence_analysis.png       # Convergence comparison
│   └── generalization_analysis.png    # Generalization performance
│
├── logs/                              # Training logs and documentation
│   ├── training_log.md                # DQN training log
│   ├── ppo_training_log.md            # PPO training log
│   └── a2c_training_log.md            # A2C training log
│
├── testing_models/                    # Model testing and analysis
│   ├── test.py                        # Basic model testing
│   └── analyze_convergence_generalization.py # Advanced analysis
│
└── new_env/                           # Python virtual environment (excluded)
```

## 🔧 Optimal Hyperparameters

### PPO (Best Performance)
```python
PPO(
    policy="MlpPolicy",
    learning_rate=0.0003,
    batch_size=64,
    gamma=0.99,
    total_timesteps=100000
)
```

### DQN
```python
DQN(
    policy="MlpPolicy",
    learning_rate=0.0003,
    buffer_size=10000,
    batch_size=32,
    exploration_final_eps=0.1
)
```

### A2C
```python
A2C(
    policy="MlpPolicy",
    learning_rate=0.0007,
    n_steps=5,
    gamma=0.99
)
```

## 📈 Visualization Features

- **Real-time Environment Rendering**: 5×5 grid with color-coded mutations
- **Agent Animation**: Pulsing blue agent with smooth transitions
- **Statistics Panel**: Live budget, score, and action tracking
- **Professional UI**: Clean interface with step-by-step logging
- **GIF Export**: Generate shareable animated demonstrations

## 🧪 Experimental Results

### Key Findings
1. **PPO Superior Performance**: Achieved highest rewards (869.1) with fastest convergence (5 episodes)
2. **Sample Efficiency**: PPO required only 31 total episodes vs 93 for A2C
3. **Generalization**: PPO maintained 88.8% performance on unseen states
4. **Stability**: Low variance (±36.9) indicates consistent learning

### Performance Insights
- **Learning rates** of 0.0003-0.0007 optimal for this environment
- **High discount factors** (0.99) crucial for budget management
- **Batch size scaling** with algorithm complexity improves stability
- **Budget constraints** favor policy gradient methods over value-based

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Algorithmic Enhancements**: Implement hierarchical RL, curiosity-driven learning
- **Environment Extensions**: Larger grids, dynamic landscapes, multi-objective optimization
- **Analysis Tools**: Statistical significance testing, ablation studies
- **Performance Optimization**: Training speed improvements, hyperparameter tuning

## 📄 License

[Add your license information here]

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{genetic_mutation_rl,
  title={Genetic Mutation Environment Reinforcement Learning},
  author={[Your Name]},
  year={2024},
  howpublished={\url{[Repository URL]}}
}
```

## 📞 Contact

m.bonyu@alustudent.com

---

**🔬 Explore the intersection of reinforcement learning and genetic analysis with our comprehensive mutation testing environment!**
