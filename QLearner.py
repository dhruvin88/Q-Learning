"""
QLearner
CoreLogic - AI Coding Challenge
This class assumes that the states and actions are integer values.
"""
import numpy as np
import random as rand

class QLearner:

    def __init__(self,
                 num_states=100,
                 num_actions=3,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 verbose=False
                 ):

        self.num_states = num_states  # number of states in the environment
        self.num_actions = num_actions  # number of actions, should be 3
        self.alpha = alpha  # this is the learning rate
        self.gamma = gamma  # this is the weight of the future rewards
        self.rar = rar  # this is the initial random value for exploration
        self.radr = radr  # this is the amount of decay for the exploration
        self.verbose = verbose  # use this if you want to print logs to help debug

        # initialize the q table
        self.Q = np.zeros((num_states, num_actions))
        self.s = 0
        self.a = 0

    def init_state(self, s):
        """
        This function will set the state, and return best action at the given state.
        :param s: The stating state of the q-learner.
        :return: A valid action to take.
        """
        # TODO: Write code to initialize the state of the q-learner
        # Hint: return optimal action here
        self.a = np.argmax(self.Q[s,:]);
        self.s = s
        return self.a
    
    def query(self, s):
        """
        This function will return the best action at the given state without changing the learner.
        :param s: The stating state of the q-learner.
        :return: A valid action to take.
        """
        # TODO: Write code to return the best action for the current state.
        # Hint: return optimal action here
        return np.argmax(self.Q[s,:]);
    
    def step(self, s_prime, reward):
        """
        This function will cause the q-learner to take a step,
        and use the next state (s_prime) and the reward.
        :param s_prime: The new state.
        :param reward: The reward.
        :return: The next action to take.
        """
        # TODO: Write the code to update q-values, get best action, and update internal data.
        # You will want to do the following:

        # TODO: Update Q value using Q Learning Bellman update equation
        self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + self.alpha * (reward + self.gamma * self.Q[s_prime, self.query(s_prime)])
        
        # TODO: Update internal variables
        self.s = s_prime
        #self.a = self.query(self.s)
        
        # TODO: return best actions for current state# TODO: Get action to return explore or exploit?
        random = np.random.rand()
        if random < self.rar:
            # Take rand action
            self.a = np.random.randint(0,3)
        else:
            # choice optimum action
            self.a = self.query(self.s)
        self.rar *= self.radr
        
        return self.a