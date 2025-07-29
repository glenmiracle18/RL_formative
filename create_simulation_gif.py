import sys
import os
import gymnasium as gym
import numpy as np
import pygame
import imageio

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.genetic_env import GeneticMutationEnv

def create_simulation_gif():
    env = GeneticMutationEnv(render_mode='human')
    obs, info = env.reset()
    frames = []
    
    # Simulate random actions in the environment
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Capture the frame
        frame = pygame.surfarray.array3d(env.window)
        frames.append(np.transpose(frame, (1, 0, 2)))  # Transpose to match image orientation
        
        if terminated or truncated:
            obs, info = env.reset()
        
    env.close()

    # Save frames as GIF
    gif_path = os.path.join('models', 'environment_simulation.gif')
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Simulation GIF saved to: {gif_path}")

if __name__ == "__main__":
    create_simulation_gif()
