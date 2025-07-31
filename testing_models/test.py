# Example evaluation script for a saved model
from stable_baselines3 import DQN
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.genetic_env import GeneticMutationEnv

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dqn', 'genetic_dqn')
model = DQN.load(model_path)

# Create environment
env = GeneticMutationEnv(render_mode="human")

# Test the trained model
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()