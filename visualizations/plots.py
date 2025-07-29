import matplotlib.pyplot as plt

# Example data
timesteps = range(10000)
rewards_dqn = [np.random.rand() for _ in timesteps]
rewards_ppo = [np.random.rand() for _ in timesteps]
rewards_a2c = [np.random.rand() for _ in timesteps]

plt.plot(timesteps, rewards_dqn, label='DQN')
plt.plot(timesteps, rewards_ppo, label='PPO')
plt.plot(timesteps, rewards_a2c, label='A2C')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.legend()
plt.show()