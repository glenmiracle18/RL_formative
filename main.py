from environments.genetic_env import GeneticMutationEnv
from training.dqn_training import train_dqn

if __name__ == "__main__":
    # Test the environment with random actions
    env = GeneticMutationEnv(render_mode="human")
    obs, _ = env.reset()
    for _ in range(20):  # Run for 20 steps
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    # Train the DQN model. will soon add other models
    print("Starting DQN training...")
    train_dqn()