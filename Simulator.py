"""
Simulator
CoreLogic - AI Coding Challenge
This class will take in an orders file and simulate those trades.
"""
import numpy as np
import matplotlib.pyplot as plt

class Simulator:

    def __init__(self, max_position, starting_cash):
        self.max_position = max_position
        self.stating_cash = starting_cash

    def simulate(self, df, symbol, orders, save_graph=False):
        cash = self.stating_cash
        position = 0
        tmp_prices = df[[symbol, ]]
        prices = tmp_prices.loc[tmp_prices.index, :]
        prices['QLearner Value'] = np.nan

        prices = prices.sort_index()

        for date, pricedata in prices.iterrows():
            current_price = pricedata[symbol]
            shares = orders.loc[date][symbol]
            # ensure trade is valid
            if 0 <= position + shares <= self.max_position \
                    and 0 <= current_price*shares+cash:
                position += shares
                cash -= current_price*shares

            prices.loc[date]['QLearner Value'] = cash + current_price * position

        buyholdCash = self.stating_cash - prices.iloc[0][symbol]*self.max_position
        prices['BuyHold Value'] = prices[symbol]*self.max_position + buyholdCash

        if save_graph:
            values = prices[['BuyHold Value', 'QLearner Value']]
            values.plot()

            plt.title("QLearning vs Buy Hold Strategy")
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')

            plt.savefig("QLearningPerformance.png")

        return prices.iloc[-1]['QLearner Value'], prices.iloc[-1]['BuyHold Value']
