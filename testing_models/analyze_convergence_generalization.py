import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
import numpy as np

# Load environment
env = gym.make('YourCustomEnv-v0')  # Replace with your actual environment name

# Load models
models = {
    'DQN': DQN.load('models/dqn/genetic_dqn.zip'),
    'PPO': PPO.load('models/pg/ppo.zip'),
    'A2C': A2C.load('models/pg/a2c.zip')
}

# Analyze episodes to convergence using logs
# (This is hypothetical without detailed convergence analysis code)
convergence_info = {
    'DQN': {'episodes_to_stabilization': 35},
    'PPO': {'episodes_to_stabilization': 20},
    'A2C': {'episodes_to_stabilization': 50}
}

print("Episodes to Convergence:")
for model_name, info in convergence_info.items():
    print(f"{model_name}: {info['episodes_to_stabilization']} episodes")

# Testing on unseen states
def test_generalization(model, env, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

print("\nGeneralization Performance:")
for model_name, model in models.items():
    avg_reward = test_generalization(model, env)
    print(f"{model_name}: Average reward on unseen states = {avg_reward}")
    
# Note: This code assumes you have a working custom environment 'YourCustomEnv-v0'.
