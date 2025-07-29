import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import math
import time

class EnhancedGeneticMutationEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.width = 8
        self.height = 8
        self.cell_size = 80
        self.window_width = self.width * self.cell_size
        self.window_height = self.height * self.cell_size + 100  # Extra space for UI
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Test
        
        self.agent_pos = np.array([self.height//2, self.width//2])  # Starting position
        self.budget = 100
        self.mutations = self._load_mutations()
        self.current_mutation_idx = 0
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None
        
        # Animation and effects
        self.agent_pulse = 0
        self.test_animation = None
        self.last_reward = 0
        self.action_history = []
        
        # Colors
        self.colors = {
            'background': (25, 25, 40),
            'grid': (60, 60, 80),
            'agent': (100, 200, 255),
            'high_mutation': (255, 100, 100),
            'medium_mutation': (255, 200, 100),
            'low_mutation': (100, 255, 100),
            'text': (255, 255, 255),
            'ui_bg': (40, 40, 60)
        }

    def _load_mutations(self):
        # Create more interesting mutation patterns
        mutations = np.random.rand(self.height, self.width)
        
        # Add some hot spots (high-value mutations)
        for _ in range(3):
            center_x, center_y = np.random.randint(1, self.height-1), np.random.randint(1, self.width-1)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.height and 0 <= y < self.width:
                        mutations[x, y] = min(1.0, mutations[x, y] + 0.4)
        
        return mutations

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([self.height//2, self.width//2])
        self.budget = 100
        self.current_mutation_idx = 0
        self.mutations = self._load_mutations()
        self.last_reward = 0
        self.action_history = []
        
        self.state = np.array([
            self.mutations[int(self.agent_pos[0]), int(self.agent_pos[1])],
            self.budget / 100,
            0  # Mutation type placeholder
        ], dtype=np.float32)
        
        if self.render_mode == "human":
            self._render_setup()
        return self.state, {}

    def step(self, action):
        reward = 0
        terminated = False
        self.action_history.append(action)
        
        if len(self.action_history) > 20:
            self.action_history.pop(0)

        if action == 0:  # Test Mutation
            self.budget -= 1
            self.test_animation = time.time()
            
            if self.budget < 0:
                reward = -10
                terminated = True
            else:
                evo2_score = self.state[0]
                if evo2_score > 0.8:
                    reward = 10
                elif evo2_score > 0.5:
                    reward = 5
                else:
                    reward = -2
                    
        elif action == 1:  # Move Up
            new_pos = self.agent_pos + np.array([-1, 0])
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                reward = -0.1  # Small penalty for moving
                
        elif action == 2:  # Move Down
            new_pos = self.agent_pos + np.array([1, 0])
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                reward = -0.1
                
        elif action == 3:  # Move Left
            new_pos = self.agent_pos + np.array([0, -1])
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                reward = -0.1
                
        elif action == 4:  # Move Right
            new_pos = self.agent_pos + np.array([0, 1])
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                reward = -0.1

        self.last_reward = reward
        self.state = np.array([
            self.mutations[int(self.agent_pos[0]), int(self.agent_pos[1])],
            self.budget / 100,
            0
        ], dtype=np.float32)

        if self.budget <= 0:
            terminated = True

        if self.render_mode == "human":
            self.render()
            
        return self.state, reward, terminated, False, {}

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def render(self, mode='human'):
        if mode == 'human':
            if self.window is None:
                self._render_setup()
            self._draw_frame()

    def _render_setup(self):
        pygame.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Enhanced Genetic Mutation Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)

    def _draw_frame(self):
        self.window.fill(self.colors['background'])
        
        # Update animations
        self.agent_pulse += 0.3
        
        # Draw mutation grid with enhanced visuals
        for i in range(self.height):
            for j in range(self.width):
                x = j * self.cell_size
                y = i * self.cell_size
                
                mutation_value = self.mutations[i, j]
                
                # Color based on mutation strength
                if mutation_value > 0.8:
                    color = self.colors['high_mutation']
                elif mutation_value > 0.5:
                    color = self.colors['medium_mutation']
                else:
                    color = self.colors['low_mutation']
                
                # Add some transparency based on value
                alpha = int(100 + 155 * mutation_value)
                temp_surface = pygame.Surface((self.cell_size-2, self.cell_size-2))
                temp_surface.set_alpha(alpha)
                temp_surface.fill(color)
                self.window.blit(temp_surface, (x+1, y+1))
                
                # Draw grid lines
                pygame.draw.rect(self.window, self.colors['grid'], 
                               (x, y, self.cell_size, self.cell_size), 1)
                
                # Draw mutation value text
                value_text = self.font.render(f"{mutation_value:.2f}", True, self.colors['text'])
                text_rect = value_text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                self.window.blit(value_text, text_rect)

        # Draw agent with pulsing effect
        agent_x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        agent_y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
        
        pulse_size = 5 + 3 * math.sin(self.agent_pulse)
        agent_radius = int(15 + pulse_size)
        
        # Agent shadow
        pygame.draw.circle(self.window, (0, 0, 0), 
                         (agent_x + 2, agent_y + 2), agent_radius, 0)
        
        # Agent body
        pygame.draw.circle(self.window, self.colors['agent'], 
                         (agent_x, agent_y), agent_radius, 0)
        
        # Agent outline
        pygame.draw.circle(self.window, (255, 255, 255), 
                         (agent_x, agent_y), agent_radius, 2)
        
        # Test animation effect
        if self.test_animation and time.time() - self.test_animation < 0.5:
            animation_progress = (time.time() - self.test_animation) / 0.5
            effect_radius = int(agent_radius + 20 * animation_progress)
            effect_alpha = int(255 * (1 - animation_progress))
            
            effect_surface = pygame.Surface((effect_radius * 2, effect_radius * 2))
            effect_surface.set_alpha(effect_alpha)
            
            if self.last_reward > 0:
                pygame.draw.circle(effect_surface, (0, 255, 0), 
                                 (effect_radius, effect_radius), effect_radius, 3)
            else:
                pygame.draw.circle(effect_surface, (255, 0, 0), 
                                 (effect_radius, effect_radius), effect_radius, 3)
            
            self.window.blit(effect_surface, 
                           (agent_x - effect_radius, agent_y - effect_radius))

        # Draw UI panel
        ui_y = self.height * self.cell_size
        pygame.draw.rect(self.window, self.colors['ui_bg'], 
                        (0, ui_y, self.window_width, 100))
        
        # Budget display
        budget_text = self.large_font.render(f"Budget: {self.budget}", True, self.colors['text'])
        self.window.blit(budget_text, (10, ui_y + 10))
        
        # Current mutation value
        current_mutation = self.mutations[self.agent_pos[0], self.agent_pos[1]]
        mutation_text = self.font.render(f"Current Mutation Score: {current_mutation:.3f}", 
                                       True, self.colors['text'])
        self.window.blit(mutation_text, (10, ui_y + 50))
        
        # Last reward
        reward_color = (0, 255, 0) if self.last_reward > 0 else (255, 0, 0) if self.last_reward < 0 else self.colors['text']
        reward_text = self.font.render(f"Last Reward: {self.last_reward:.1f}", 
                                     True, reward_color)
        self.window.blit(reward_text, (10, ui_y + 75))
        
        # Action legend
        legend_x = self.window_width - 200
        legend_text = self.font.render("Actions: ↑↓←→ Test", True, self.colors['text'])
        self.window.blit(legend_text, (legend_x, ui_y + 10))
        
        # Position display
        pos_text = self.font.render(f"Position: ({self.agent_pos[0]}, {self.agent_pos[1]})", 
                                  True, self.colors['text'])
        self.window.blit(pos_text, (legend_x, ui_y + 35))

        pygame.display.flip()
        self.clock.tick(30)  # Higher FPS for smoother animation

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
