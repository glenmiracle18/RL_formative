#!/usr/bin/env python3
"""
Enhanced Genetic Mutation Environment Demonstration
This script shows an agent taking random actions in the environment
without any training or model involved - pure visualization demo.
"""

import sys
import os
import time

# Add the environment path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from environments.enhanced_genetic_env import EnhancedGeneticMutationEnv
import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_demo_gif():
    """Create a GIF showing the agent taking random actions"""
    print("🧬 Starting Enhanced Genetic Mutation Environment Demo...")
    print("📺 Watch the pygame window for real-time action!")
    print("⏸️  Press SPACE to pause/unpause, ESC or Q to quit early")
    
    env = EnhancedGeneticMutationEnv(render_mode='human')
    obs, info = env.reset()
    
    frames = []
    step_count = 0
    episode_count = 1
    total_reward = 0
    paused = False
    
    print(f"📊 Episode {episode_count} started")
    
    # Run the simulation
    for i in range(200):  # Capture 200 frames
        # Handle pygame events for real-time interaction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n⏹️ User closed window")
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"\n{'⏸️ Paused' if paused else '▶️ Resumed'}")
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    print("\n⏹️ User quit early")
                    break
        
        # Handle pause
        if paused:
            time.sleep(0.1)
            continue
        
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
        
        # Ensure the display is updated
        pygame.display.flip()
        
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
        
        # Longer delay for better visualization
        time.sleep(0.15)
    
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
    print("📺 Watch the pygame window for real-time action!")
    print("⏸️  Press SPACE to pause/unpause, ESC or Q to quit, F for faster, S for slower")
    
    env = EnhancedGeneticMutationEnv(render_mode='human')
    obs, info = env.reset()
    
    step_count = 0
    total_reward = 0
    paused = False
    delay = 0.2  # Default delay
    running = True
    
    try:
        # Run indefinitely until user quits
        while running:
            # Handle pygame events for real-time interaction
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n⏹️ User closed window")
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"\n{'⏸️ Paused' if paused else '▶️ Resumed'}")
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        print("\n⏹️ User quit")
                        running = False
                        break
                    elif event.key == pygame.K_f:
                        delay = max(0.05, delay - 0.05)
                        print(f"\n⏩ Speed increased (delay: {delay:.2f}s)")
                    elif event.key == pygame.K_s:
                        delay = min(1.0, delay + 0.05)
                        print(f"\n⏪ Speed decreased (delay: {delay:.2f}s)")
            
            if not running:
                break
                
            # Handle pause
            if paused:
                time.sleep(0.1)
                continue
            
            action = env.action_space.sample()
            action_names = ["Test", "Up", "Down", "Left", "Right"]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print action info every 10 steps
            if step_count % 10 == 0:
                current_pos = env.agent_pos
                current_mutation = env.mutations[current_pos[0], current_pos[1]]
                print(f"  Step {step_count}: Action={action_names[action]}, "
                      f"Position=({current_pos[0]},{current_pos[1]}), "
                      f"Mutation={current_mutation:.3f}, "
                      f"Reward={reward:.1f}, "
                      f"Budget={env.budget}")
            
            if terminated or truncated:
                print(f"  📈 Episode ended: Total Reward = {total_reward:.2f}, Steps = {step_count}")
                print(f"  🔄 Starting new episode...")
                obs, info = env.reset()
                total_reward = 0
                step_count = 0
            
            # Ensure display is updated
            pygame.display.flip()
            
            time.sleep(delay)  # Adjustable delay
            
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
