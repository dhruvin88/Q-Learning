"""
Agent
CoreLogic - AI Coding Challenge
This class will take in pandas data and write a tades file for the suggested trades.
"""
from Indicators import Indicators as IndUtil
from QLearner import QLearner
import collections
import numpy as np

class Agent:

    def __init__(self, max_position=100, starting_cash=100000):
        self.indutil = IndUtil()
        self.epochMax = 200
        self.epochMin = 20
        self.converged = False
        self.learner = QLearner(
            num_states=self.indutil.get_state_size(),
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            verbose=False
        )
        self.convergedata = collections.deque(maxlen=10)
        self.numshares = max_position
        self.starting_cash = float(starting_cash)

    """
    This will determine if the algorithm has converged (internal use only)
    It will look for less than a 5% change in the average of the last 10 port values
    """
    def has_converged(self, reward, epochs):
        self.convergedata.append(reward)
        if self.epochMin > epochs:
            return False
        elif abs(100*(np.average(self.convergedata)-reward)/reward) < 1:
            return True
        elif self.epochMax >= epochs:
            return True
        else:
            return False

    """
    This method is an internal method used by training to keep track of action effects
    """
    def take_action(self, df, symbol, index, action, cash, position, last_port_value):
        # 0 for hold, 1 for buy, 2 for sell
        price = df.loc[index][symbol]

        if action == 1 and position < self.numshares and cash >= self.numshares * price:
            position += self.numshares
            cash -= self.numshares * price
        elif action == 2 and position > 0:
            cash += position * price
            position = 0

        reward = ((cash + position*price)/last_port_value)-1
        port_value = cash + position*price
        return cash, position, reward, port_value

    """
    This will train the agents q-learner
    """
    def train(self, df, symbol):
        states = self.indutil.compute_indicators(df, symbol)
        epoch = 0
        last_epoch_value = self.starting_cash

        while not self.has_converged(last_epoch_value, epoch):
            position = 0
            cash = self.starting_cash
            last_port_value = cash

            action = self.learner.init_state(states.iloc[0].values[0])
            cash, position, reward, port_value \
                = self.take_action(df, symbol, states.index[0], action, cash, position, last_port_value)
            for index, state in states[1:].iterrows():
                action = self.learner.step(state.values[0], reward)
                last_port_value = port_value
                cash, position, reward, port_value \
                    = self.take_action(df, symbol, index, action, cash, position, last_port_value)

            price = df.iloc[-1][symbol]
            last_epoch_value = position * price + cash
            epoch += 1

    """
    This method will be called to test the learned policy.
    """
    def test(self, df, symbol):
        states = self.indutil.compute_indicators(df, symbol)
        trades = df[[symbol, ]]
        trades.values[:, :] = 0
        position = 0
        cash = self.starting_cash

        for index, state in states.iterrows():
            action = self.learner.query(state.values[0])
            price = df.loc[index][symbol]
            if action == 1 and cash >= price*self.numshares and position < self.numshares:
                trades.loc[index][symbol] = self.numshares
                position = self.numshares
                cash -= price*self.numshares
            elif action == 2 and position > 0:
                trades.loc[index][symbol] = -position
                cash += price * position
                position = 0

        return trades
