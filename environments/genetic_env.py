import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class GeneticMutationEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.width = 5
        self.height = 5
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Test
        self.agent_pos = np.array([2, 2])  # Starting position of the agent
        self.budget = 100  # Starting budget
        self.mutations = self._load_mutations()
        self.current_mutation_idx = 0
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _load_mutations(self):
        # Simulated EVO 2 scores for mutations
        return np.random.rand(self.width, self.height)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([2, 2])
        self.budget = 100
        self.current_mutation_idx = 0
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

        if action == 0:  # Test Mutation
            self.budget -= 1
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
            new_pos = self.agent_pos + np.array([-2, 0])
            self.agent_pos = new_pos if self._is_valid_position(new_pos) else self.agent_pos
        elif action == 2:  # Move Down
            new_pos = self.agent_pos + np.array([2, 0])
            self.agent_pos = new_pos if self._is_valid_position(new_pos) else self.agent_pos
        elif action == 3:  # Move Left
            new_pos = self.agent_pos + np.array([0, -2])
            self.agent_pos = new_pos if self._is_valid_position(new_pos) else self.agent_pos
        elif action == 4:  # Move Right
            new_pos = self.agent_pos + np.array([0, 2])
            self.agent_pos = new_pos if self._is_valid_position(new_pos) else self.agent_pos

        self.state = np.array([
            self.mutations[int(self.agent_pos[0]), int(self.agent_pos[1])],
            self.budget / 100,
            0  # Mutation type placeholder
        ], dtype=np.float32)

        if self.budget <= 0 or self.current_mutation_idx >= len(self.mutations):
            terminated = True

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, False, {}

    def _is_valid_position(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def render(self, mode='human'):
        if mode == 'human':
            if self.window is None:
                self._render_setup()
            self._draw_frame()

    def _render_setup(self):
        pygame.init()
        self.window = pygame.display.set_mode(((self.width * 50, self.height * 50)))
        pygame.display.set_caption("Genetic Mutation Environment")
        self.clock = pygame.time.Clock()

    def _draw_frame(self):
        self.window.fill((255, 255, 255))  # White background

        # Draw agent
        pygame.draw.rect(self.window, (0, 0, 255), (self.agent_pos[1] * 50, self.agent_pos[0] * 50, 50, 50))

        # Draw mutations
        for i in range(self.width):
            for j in range(self.height):
                color = (int(255 * self.mutations[i, j]), 0, 0)  # Red color based on EVO2 score
                pygame.draw.rect(self.window, color, (j * 50, i * 50, 50, 50))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()