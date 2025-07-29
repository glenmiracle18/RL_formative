from stable_baselines3 import DQN
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.genetic_env import GeneticMutationEnv

# Create environment
env = GeneticMutationEnv(render_mode="human")

# Create DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    buffer_size=10000,
    batch_size=32,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    exploration_fraction=0.1,
    target_update_interval=1000,
    learning_starts=1000
)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("models/dqn/genetic_dqn")

# Test the trained model
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()