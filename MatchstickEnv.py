# Classe MatchstickEnv représentant l'environnement d'un jeu d'allumettes. 

# La méthode reset initialise l'état du jeu à num_matches.
# La méthode reset retourne cet état. 

# La méthode step permet de prendre une action, qui consiste à enlever 1, 2 ou 3 allumettes.
# La méthode step retourne :
# 1. l'état suivant, 
# 2. la récompense (1 si le jeu est terminé, 0 sinon), 
# 3. un indicateur de terminaison 
# 4. un dictionnaire vide. 

# La méthode _check_win vérifie si un joueur a gagné.

# La classe hérite de gym.Env et définit les espaces d'observation et d'action en utilisant les classes spaces.Discrete de gym. 
# Les erreurs sont également gérées en utilisant les méthodes standard de gym.Env

import gym
from gym import spaces

class MatchstickEnv(gym.Env):
    def __init__(self, n_matchsticks):
        self.n_matchsticks = n_matchsticks
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(n_matchsticks + 1)
        self.reset()
        
    def reset(self):
        self.state = self.n_matchsticks
        self.turn = 1
        return self.state
    
    def step(self, action):
        if action not in range(1, 4):
            raise ValueError("Invalid action")
        if self.state - action < 0:
            raise ValueError("Not enough matchsticks")
        self.state -= action
        done = self._check_win()
        reward = 1 if done and self.turn == 1 else -1 if done else 0
        self.turn = 1 if self.turn == 2 else 2
        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        print("Player {} turn, remaining matchsticks: {}".format(self.turn, self.state))
    
    def _check_win(self):
        return self.state == 0
