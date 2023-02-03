"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

from typing import Tuple, Dict, Optional, Iterable, Callable

import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from gym.utils import seeding
import numpy as np

import pygame
from pygame import gfxdraw

import matplotlib.pyplot as plt
import time
from IPython import display

class BucketEnv3(gym.Env):
    """
    Description:
       

    Observation:
        

    Actions:
        

    Récompense:
        

    Etat de départ:
        

    Conditionns d'arrêt:
        
       

    """

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 20
    }

    def __init__(self):
        self.X = 3
        self.Y = 5
        self.C = self.X * self.Y // 2
        self.box = np.zeros(shape=(self.X, self.Y + 2), dtype=int)
        self.actions = {
            # (ori, x, y)
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),            
         }
        self.dimensions = {
            0: (1, 2),
            1: (2, 1)
        }
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([self.Y + 2 for _ in range(self.X)])
        
        self.seed()
        self.state = None

        self.screen = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _prevY(self, x):
        y = self.Y - 1
        while y > -1 and self.box[x][y] == 0:
            y -= 1
        return y + 1    
    
    def step(self, action: int):
        self.C -= 1
        ori, x = self.actions[action]
        dx, dy = self.dimensions[ori]
        if x >= 0 and x + dx <= self.X:
            if dx == 1:
                # check x and 'fill' to compute y
                y = self._prevY(x)
            else:
                # check x and x+1 and 'fill' to compute y
                y = max(self._prevY(x), self._prevY(x + 1))
        else:
            y = 0
        # check bounds on (x,y)
        done = x < 0 or x + dx > self.X or y + dy > self.Y
        # update box
        area = []
        for i in range(x, x + dx):
            if i >= self.X:
                # overload
                continue
            for j in range(y, y + dy):
                self.box[i][j] = 1
                if j >= self.Y:
                    # overload
                    continue
        self.state = [self._prevY(l) for l in range(self.X)]
        last = np.sum(self.box[:,self.Y-1])
        if not done:
            reward = 1.0
        else:
            reward = 0.0
        return tuple(self.state), reward, done, {}
    
    def simulate_step(self, state: Tuple[int, int], action: int):
        ori, x = self.actions[action]
        dx, dy = self.dimensions[ori]
        next_state = list(state)
        if dx == 1:
            # check x and 'fill' to compute y
            y = state[x]
            next_state[x] += 2
        else:
            # check x and x+1 and 'fill' to compute y
            y = max(state[x], state[x+1])
            next_state[x] += 1
            next_state[x+1] += 1
        # check bounds on (x,y)
        done = x + dx > self.X or y + dy > self.Y
        if not done:
            reward = 1.0
        else:
            reward = 0.0
        return tuple(next_state), reward, done, {}
    

    def reset(self):
        self.box = np.zeros(shape=(self.X, self.Y + 2), dtype=int)
        self.state = [0 for _ in range(self.X)]

        self.C = self.X * self.Y // 2
        self.steps_beyond_done = None
        return tuple(self.state)

    def render(self, mode='rgb_array'):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_size = 600
        #scale = screen_size // 5
        blocksize = 70
        cx = screen_size // 2
        cy = screen_size - (blocksize * 6)
        rx = cx - self.X // 2 * blocksize
        ry = cy - self.Y // 2 * blocksize
        

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_size, screen_size))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((screen_size, screen_size))

        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((143, 152, 178))
        #surf.fill((255, 255, 255))

        greyed = [(0, 5), (1, 5), (2, 5), (0, 6), (1, 6), (2, 6)]
        for (x, y), value in np.ndenumerate(self.box):
            # Add the geometry of the boxes
            c = 255 if self.box[x,y] else 192 if (x,y) in greyed else 0
                
            col = tuple([c for _ in range(3)])
                
            gfxdraw.filled_polygon(surf, [(rx + x * blocksize, ry + y * blocksize),
                                          (rx + (x + 1) * blocksize, ry + y * blocksize),
                                          (rx + (x + 1) * blocksize, ry + (y + 1) * blocksize),
                                          (rx + x * blocksize, ry + (y + 1) * blocksize),
                                          (rx + x * blocksize, ry + y * blocksize)],
                                   col)

        # Add the geometry of the matrix
        (r, g, b) = (129, 132, 203)
        for x in range(self.X + 1):
            gfxdraw.line(surf, rx + (x * blocksize),
                         ry,
                         rx + (x * blocksize),
                         ry + (self.Y +2) * blocksize,
                         (r, g, b))

        for y in range(self.Y + 3):
            gfxdraw.line(surf, rx,
                         ry + y * blocksize,
                         rx + self.X * blocksize,
                         ry + y * blocksize,
                         (255,0,0) if y == self.Y else (r, g, b))
                    
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

            
def plot_stats(stats,smooth=10):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[i-smooth:i+smooth]) for i in range(smooth, len(vals)-smooth)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()  
    
def show_render(img, render,sleep=0.1):
    img.set_data(render) 
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    time.sleep(sleep)    
    
def rendering(env, policy: Callable, episodes = 2):
    plt.figure(figsize=(8, 8))
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        img = plt.imshow(env.render(mode='rgb_array')) 
        while not done:
            p = policy(state)
            if isinstance(p, np.ndarray):
                action = np.random.choice(5, p=p)
            else:
                action = p
            
            #action = np.argmax(action_values[state])
            next_state, reward, done, _ = env.step(action)
            show_render(img, env.render(mode='rgb_array')) 
            state = next_state
            
def testing(env, action_values):
    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        step += 1
        action = np.argmax(action_values[state])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        if not done:
            state = next_state
        else:
            print(f"Episode finished after {step} timesteps, earn {total_reward}")