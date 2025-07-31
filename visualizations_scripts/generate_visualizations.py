import matplotlib.pyplot as plt
import numpy as np

# Data from training logs
# DQN
dqn_rewards = [100, 73, 598, 683, 540, 400, -152, -20, 28, 760, 4, 400,
               232, 16, 750, 635, 1000, 720, 535, 1000, 543, 560, -20, 232,
               905, -93, 769, -8, 929, 965, 640, 925, 558, 80, -20, 640, 723,
               412, 510, 430, 510, 990, 124, 573, 700, 129, 448, 651, 615, 785, 848, 885]

# PPO
ppo_rewards = [755, 670, 569, 695, 767, 802, 724, 777, 868, 758,
               866, 816, 785, 817, 830, 870, 865, 875, 880, 761,
               751, 813, 845, 833, 863, 858, 868, 850, 923, 928, 910]

# A2C
a2c_rewards = [210, 447, 791, 871, 804, 821, 995, 891, 542, 830,
               985, 692, 822, 465, 705, 381, 493, 522, 500, 458, 374,
               52, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
               500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
               500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
               500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
               500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
               500, 500, 500, 38, 500]

# Calculate cumulative rewards
cumulative_dqn_rewards = np.cumsum(dqn_rewards)
cumulative_ppo_rewards = np.cumsum(ppo_rewards)
cumulative_a2c_rewards = np.cumsum(a2c_rewards)

# Plot cumulative rewards over episodes
plt.figure(figsize=(14, 7))

plt.plot(cumulative_dqn_rewards, label='DQN', linestyle='-', marker='o')
plt.plot(cumulative_ppo_rewards, label='PPO', linestyle='-', marker='x')
plt.plot(cumulative_a2c_rewards, label='A2C', linestyle='-', marker='s')

plt.title('Cumulative Rewards over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid(True)

plt.savefig('visualizations/cumulative_rewards.png')
plt.show()

# Placeholder data for DQN objective function
# In a real scenario, this should be actual data from training logs
# DQN Objective
dqn_objective = np.random.normal(loc=100, scale=5, size=52)

# PG Policy Entropy
# Placeholder for PG policy entropy; replace with actual data if available
ppo_entropy = np.random.normal(loc=0.7, scale=0.05, size=31)

# Plot objective function and entropy
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(dqn_objective, label='DQN Objective', color='g', linestyle='-', marker='o')
plt.title('DQN Objective Function over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Objective Value')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(ppo_entropy, label='PPO Policy Entropy', color='b', linestyle='-', marker='x')
plt.title('PPO Policy Entropy over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('visualizations/training_stability.png')
plt.show()
