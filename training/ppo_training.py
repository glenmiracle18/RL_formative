from stable_baselines3 import PPO
from environment.genetic_env import GeneticMutationEnv

# Create environment
env = GeneticMutationEnv(render_mode=None)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("models/pg/ppo")