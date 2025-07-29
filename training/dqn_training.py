from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.genetic_env import GeneticMutationEnv

class TrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        return True

def train_dqn():
    start_time = time.time()
    
    # Create environment
    env = GeneticMutationEnv(render_mode=None)  # No rendering during training

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

    # Create callback to log training metrics
    logger = TrainingLogger()
    
    # Train the models
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps, callback=logger)
    
    training_time = time.time() - start_time

    # Save the model
    os.makedirs("models/dqn", exist_ok=True)
    model.save("models/dqn/genetic_dqn")

    # Create training log
    log_path = os.path.join("models", "dqn", "training_log.md")
    with open(log_path, "w") as log_file:
        log_file.write("# DQN Training Results\n\n")
        log_file.write(f"**Training completed on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write(f"**Total training time:** {training_time:.2f} seconds\n\n")
        log_file.write(f"**Total timesteps:** {total_timesteps}\n\n")
        
        log_file.write("## Model Hyperparameters\n\n")
        log_file.write(f"- **Learning rate:** 0.0003\n")
        log_file.write(f"- **Buffer size:** 10000\n")
        log_file.write(f"- **Batch size:** 32\n")
        log_file.write(f"- **Exploration initial epsilon:** 1.0\n")
        log_file.write(f"- **Exploration final epsilon:** 0.05\n")
        log_file.write(f"- **Exploration fraction:** 0.1\n")
        log_file.write(f"- **Target update interval:** 1000\n")
        log_file.write(f"- **Learning starts:** 1000\n\n")
        
        if logger.episode_rewards:
            log_file.write("## Training Performance\n\n")
            log_file.write(f"- **Total episodes:** {len(logger.episode_rewards)}\n")
            log_file.write(f"- **Average episode reward:** {sum(logger.episode_rewards)/len(logger.episode_rewards):.2f}\n")
            log_file.write(f"- **Best episode reward:** {max(logger.episode_rewards):.2f}\n")
            log_file.write(f"- **Worst episode reward:** {min(logger.episode_rewards):.2f}\n")
            log_file.write(f"- **Average episode length:** {sum(logger.episode_lengths)/len(logger.episode_lengths):.2f}\n\n")
            
            log_file.write("## Episode Rewards History\n\n")
            log_file.write("| Episode | Reward | Length |\n")
            log_file.write("|---------|--------|-----------|\n")
            for i, (reward, length) in enumerate(zip(logger.episode_rewards, logger.episode_lengths), 1):
                log_file.write(f"| {i} | {reward:.2f} | {length} |\n")
        else:
            log_file.write("## Training Performance\n\n")
            log_file.write("No episodes completed during training.\n")
    
    print(f"\nTraining log saved to: {log_path}")

    # Test the trained model
    env_test = GeneticMutationEnv(render_mode="human")
    obs, _ = env_test.reset()
    test_rewards = []
    current_reward = 0
    
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env_test.step(action)
        current_reward += reward
        env_test.render()
        if terminated or truncated:
            test_rewards.append(current_reward)
            current_reward = 0
            obs, _ = env_test.reset()
    env_test.close()
    
    return model

if __name__ == "__main__":
    train_dqn()
