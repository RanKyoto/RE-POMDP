import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pygame

from typing import Optional

class NoisyGridworldEnv(gym.Env):
    """
        A gridworld environment with noisy state observation.   
        States:  
        Type: Discrete(grid_size * grid_size)

        Actions:  
        Type: Discrete(5)  
        Note: Up, Down, Left, Right, Stay

        :args grid_size: the size of the gridworld. 
        :args p: probability of observing the true state
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        }
    def __init__(self, grid_size=4, p=0.4, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        
        self.grid_size = grid_size
        self.p = p  # Probability of observing the true state
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, Stay
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.misunderstood = False

        self.window_size = 400  # Size of the pygame window
        self.cell_size = self.window_size // self.grid_size
        self.screen = None
        self.font = None
        
        self.reset()

    def _get_random_start_pos(self):
        return (np.random.randint(self.grid_size), np.random.randint(self.grid_size))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = {}):
        super().reset(seed=seed)
        if options.get("start_pos") is None:
            self.start_pos = self._get_random_start_pos()
        else:
            self.start_pos = options.get("start_pos")
        self.current_pos = self.start_pos
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.noisy_pos = self.get_noisy_pos()  # Update noisy observation
        if self.render_mode == "human":
            self.render()
        true_state = self.current_pos[0] * self.grid_size + self.current_pos[1]
        noisy_state = self.noisy_pos[0] * self.grid_size + self.noisy_pos[1]
        if self.misunderstood:
            return noisy_state, {"true_state":true_state}
        else:
            return true_state, {"noisy_state":noisy_state}

    def reward(self, current_pos, action):
        reward = 0
        observation = current_pos[0] * self.grid_size + current_pos[1] + 1
        if observation in {7,10,12,15}:
            reward += 10
        
        if action == 4:
            reward += 0 # stay
        else:
            reward -= 1 # move

        return reward

    def step(self, action):
        if self.misunderstood:
            reward = self.reward(self.noisy_pos,action)
        else:
            reward = self.reward(self.current_pos,action)

        if action == 0:  # Up
            next_pos = (self.current_pos[0] - 1, self.current_pos[1])
        elif action == 1:  # Down
            next_pos = (self.current_pos[0] + 1, self.current_pos[1])
        elif action == 2:  # Left
            next_pos = (self.current_pos[0], self.current_pos[1] - 1)
        elif action == 3:  # Right
            next_pos = (self.current_pos[0], self.current_pos[1] + 1)
        elif action == 4:  # Stay (do nothing)
            next_pos = self.current_pos

        # Check if next position is within grid boundaries
        if 0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size:
            self.current_pos = next_pos

        
        terminated, truncated = False, False

        # Update noisy observation
        self.noisy_pos = self.get_noisy_pos()

        if self.render_mode == "human":
            self.render()

        true_state = self.current_pos[0] * self.grid_size + self.current_pos[1]
        noisy_state = self.noisy_pos[0] * self.grid_size + self.noisy_pos[1]
        if self.misunderstood:
            return noisy_state, reward, terminated, truncated, {"ture_state":true_state}
        else:
            return true_state, reward, terminated, truncated, {"noisy_state":noisy_state}

    def get_noisy_pos(self):
        if np.random.rand() < self.p:
            return self.current_pos
        
        neighbors = self.get_neighbors(self.current_pos)
        # Randomly select one of the neighbors
        noisy_pos = neighbors[np.random.randint(len(neighbors))]
    
        return noisy_pos
    
    def get_Psi(self):
        """ Obtain the observation probability matrix Psi"""
        num_state = self.grid_size ** 2
        Psi = np.zeros((num_state, num_state))
        for idx in range(num_state):
            Psi[idx,idx]= self.p
            i,j = idx // self.grid_size, idx % self.grid_size
            neighbors = self.get_neighbors((i,j))
            num_neighbors = len(neighbors)
            for neighbor_pos in neighbors:
                Psi[idx,neighbor_pos[0] * self.grid_size + neighbor_pos[1]] = (1-self.p)/num_neighbors
        return Psi
    
    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption('Gridworld')
            self.font = pygame.font.SysFont(None, 24)

        self.screen.fill((255, 255, 255))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                state_num = x * self.grid_size + y + 1  # Starting from 1
                if (x, y) == self.current_pos:
                    pygame.draw.rect(self.screen, (173, 255, 47), rect)  # Current position in light green
                elif (x, y) in self.get_neighbors(self.current_pos):
                    pygame.draw.rect(self.screen, (255, 255, 200), rect)  # Neighbors in yellow      
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Empty cell
                
                if (x, y) == self.noisy_pos:
                    pygame.draw.circle(self.screen, (255, 165, 0), rect.center, self.cell_size // 4)  # Noisy observation 
                text = self.font.render(f'S{state_num}', True, (0, 0, 0))
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)

        pygame.display.flip()
        if mode == "human":
            return None
        else:
            return pygame.surfarray.array3d(self.screen)

    def get_neighbors(self, pos):
        neighbors = []
        x, y = pos

        # Determine the neighbors based on position
        if x > 0:  # Up
            neighbors.append((x - 1, y))
        if x < self.grid_size - 1:  # Down
            neighbors.append((x + 1, y))
        if y > 0:  # Left
            neighbors.append((x, y - 1))
        if y < self.grid_size - 1:  # Right
            neighbors.append((x, y + 1))

        return neighbors

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# Testing the environment
if __name__ == "__main__":
    env = NoisyGridworldEnv(grid_size=4,p=0.4)
    Phi = env.get_Psi()
    observation, info = env.reset()
    env.render()

    # Main loop to test the rendering and keyboard control
    running = True
    epi_reward = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                reward = 0
                if event.key == pygame.K_w:
                    observation, reward, _, _ , info = env.step(0)  # Up
                elif event.key == pygame.K_s:
                    observation, reward, _, _ , info = env.step(1)  # Down
                elif event.key == pygame.K_a:
                    observation, reward, _, _ , info = env.step(2)  # Left
                elif event.key == pygame.K_d:
                    observation, reward, _, _ , info = env.step(3)  # Right
                elif event.key == pygame.K_SPACE:
                    observation, reward, _, _ , info = env.step(4)  # Stay
                epi_reward += reward
                print(f"Observed State: S{observation + 1}[S{info['noisy_state']+1}]")  # Print the observed state
                print(f"Reward:{epi_reward}")
                env.render()

        pygame.time.wait(100)  # Wait for 100 milliseconds before checking for the next event

    env.close()
