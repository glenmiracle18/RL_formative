import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.genetic_env import GeneticMutationEnv
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

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

def train_a2c():
    start_time = time.time()
    
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

    # Create callback to log training metrics
    logger = TrainingLogger()
    
    # Train the model
    total_timesteps = 10000
    model.learn(total_timesteps=total_timesteps, callback=logger)
    
    training_time = time.time() - start_time

    # Save the model
    os.makedirs("models/pg", exist_ok=True)
    model.save("models/pg/a2c")

    # Create training log
    log_path = os.path.join("models", "pg", "a2c_training_log.md")
    with open(log_path, "w") as log_file:
        log_file.write("# A2C Training Results\n\n")
        log_file.write(f"**Training completed on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write(f"**Total training time:** {training_time:.2f} seconds\n\n")
        log_file.write(f"**Total timesteps:** {total_timesteps}\n\n")
        
        log_file.write("## Model Hyperparameters\n\n")
        log_file.write(f"- **Learning rate:** 0.0007\n")
        log_file.write(f"- **N steps:** 5\n")
        log_file.write(f"- **Policy:** MlpPolicy\n\n")
        
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
    
    print(f"\nA2C Training log saved to: {log_path}")
    return model

if __name__ == "__main__":
    train_a2c()
