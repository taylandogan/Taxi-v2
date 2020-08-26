import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - lr: learning rate
        - nA: number of actions available to the agent
        """
        self.lr = 0.02
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def get_epsilon_probs(self, state, i_episode):
        epsilon = 1.0 / i_episode
        policy_eps = np.ones(self.nA) * epsilon / self.nA
        policy_eps[np.argmax(self.Q[state])] = 1 - epsilon + (epsilon / self.nA)
        return policy_eps

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_eps = self.get_epsilon_probs(state, i_episode)
        action = np.random.choice(self.nA, p=policy_eps)
        return action

    def step(self, state, action, reward, next_state, next_action, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            self.Q[state][action] += self.lr * (reward + 0 - self.Q[state][action])
            return

        self.Q[state][action] += self.lr * (reward + self.Q[next_state][next_action] - self.Q[state][action])

