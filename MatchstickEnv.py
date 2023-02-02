class MatchstickEnv:
    def __init__(self, num_matches):
        self.num_matches = num_matches
        self.action_space = [1, 2, 3]
        self.observation_space = [i + 1 for i in range(num_matches)]
        self.reset()

    def reset(self):
        self.state = self.num_matches
        return self.state

    def step(self, action):
        if action not in self.action_space:
            raise ValueError("Invalid action")
        if self.state - action < 0:
            raise ValueError("Not enough matches to take")
        self.state -= action
        done = self.state == 0
        reward = 1 if done else 0
        return self.state, reward, done, {}
