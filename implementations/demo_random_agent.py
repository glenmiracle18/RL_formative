#!/usr/bin/env python3
"""
Enhanced Genetic Mutation Environment Demonstration
This script shows an agent taking random actions in the environment
without any training or model involved - pure visualization demo.
"""

import sys
import os
import time
import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add the environment path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from environments.enhanced_genetic_env import EnhancedGeneticMutationEnv

def create_demo_gif():
    """Create a GIF showing the agent taking random actions"""
    print("🧬 Starting Enhanced Genetic Mutation Environment Demo...")
    
    env = EnhancedGeneticMutationEnv(render_mode='human')
    obs, info = env.reset()
    
    frames = []
    step_count = 0
    episode_count = 1
    total_reward = 0
    
    print(f"📊 Episode {episode_count} started")
    
    # Run the simulation
    for i in range(200):  # Capture 200 frames
        # Take random action
        action = env.action_space.sample()
        action_names = ["Test", "Up", "Down", "Left", "Right"]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Print action info every 10 steps
        if i % 10 == 0:
            current_pos = env.agent_pos
            current_mutation = env.mutations[current_pos[0], current_pos[1]]
            print(f"  Step {step_count}: Action={action_names[action]}, "
                  f"Position=({current_pos[0]},{current_pos[1]}), "
                  f"Mutation={current_mutation:.3f}, "
                  f"Reward={reward:.1f}, "
                  f"Budget={env.budget}")
        
        # Capture frame for GIF
        if env.window is not None:
            # Get the pygame surface as array
            frame_array = pygame.surfarray.array3d(env.window)
            frame_array = np.transpose(frame_array, (1, 0, 2))  # Correct orientation
            frames.append(frame_array)
        
        # Reset if episode ends
        if terminated or truncated:
            print(f"  📈 Episode {episode_count} ended: Total Reward = {total_reward:.2f}, Steps = {step_count}")
            episode_count += 1
            if episode_count <= 3:  # Allow up to 3 episodes in the demo
                obs, info = env.reset()
                total_reward = 0
                step_count = 0
                print(f"  🔄 Episode {episode_count} started")
            else:
                break
        
        # Small delay for better visualization
        time.sleep(0.05)
    
    env.close()
    print(f"✅ Simulation completed. Captured {len(frames)} frames")
    
    # Save frames as GIF
    if frames:
        print("🎬 Creating GIF animation...")
        
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            pil_frame = Image.fromarray(frame.astype('uint8'), 'RGB')
            pil_frames.append(pil_frame)
        
        # Save as GIF
        gif_path = os.path.join('models', 'random_agent_demo.gif')
        os.makedirs('models', exist_ok=True)
        
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,  # 100ms per frame
            loop=0
        )
        
        print(f"🎯 Demo GIF saved to: {gif_path}")
        return gif_path
    else:
        print("❌ No frames captured")
        return None

def create_static_demo():
    """Create a static demonstration without GIF"""
    print("🧬 Running Static Demo (No GIF)...")
    
    env = EnhancedGeneticMutationEnv(render_mode='human')
    obs, info = env.reset()
    
    step_count = 0
    total_reward = 0
    
    try:
        # Run for 100 steps
        for i in range(100):
            action = env.action_space.sample()
            action_names = ["Test", "Up", "Down", "Left", "Right"]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if i % 20 == 0:
                current_pos = env.agent_pos
                current_mutation = env.mutations[current_pos[0], current_pos[1]]
                print(f"  Step {step_count}: Action={action_names[action]}, "
                      f"Position=({current_pos[0]},{current_pos[1]}), "
                      f"Mutation={current_mutation:.3f}, "
                      f"Reward={reward:.1f}, "
                      f"Budget={env.budget}")
            
            if terminated or truncated:
                print(f"  Episode ended: Total Reward = {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
            
            # Handle pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            
            time.sleep(0.1)  # Slow down for better observation
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    finally:
        env.close()
        print("✅ Static demo completed")

def main():
    """Main demonstration function"""
    print("=" * 60)
    print("🧬 ENHANCED GENETIC MUTATION ENVIRONMENT DEMO")
    print("=" * 60)
    print("This demo shows an agent taking RANDOM actions")
    print("(No training or model involved - pure visualization)")
    print("=" * 60)
    
    choice = input("Choose demo type:\n1. Create GIF (requires imageio)\n2. Static demo\nEnter choice (1/2): ").strip()
    
    if choice == "1":
        try:
            import imageio
            gif_path = create_demo_gif()
            if gif_path:
                print(f"✨ Success! GIF created at: {gif_path}")
            else:
                print("❌ Failed to create GIF")
        except ImportError:
            print("❌ imageio not available. Install with: pip install imageio")
            print("🔄 Running static demo instead...")
            create_static_demo()
    else:
        create_static_demo()

if __name__ == "__main__":
    main()
