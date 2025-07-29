from stable_baselines3 import A2C
from environment.genetic_env import GeneticMutationEnv

# Create environment
env = GeneticMutationEnv(render_mode=None)

# Create A2C model
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0007,
    n_steps=5
)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("models/pg/a2c")