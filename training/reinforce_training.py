from stable_baseline3 import PPO
from environments.genetic_env import GeneticMutationEnv

# create environment
env = GeneticMutationEnv(render_mode=None)

# create PPO model
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    exploration_rate=0.1,
)


# train the model
model.learn(total_timesteps=10000)

# save the model
model.save("models/pg/reinforce_pg")