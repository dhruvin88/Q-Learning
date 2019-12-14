"""
main
CoreLogic - AI Coding Challenge
This is the main file that needs to be run to read in data, setup an agent, and simulate trading.
"""
import numpy as np
import pandas as pd
import datetime as dt
from Agent import Agent
from Simulator import Simulator as Sim

def get_data():
    """
    This is a simple method is to read in a csv file and get the desired dataframe.
    """
    start_date = dt.datetime(2017, 1, 1)
    end_date = dt.datetime(2018, 12, 31)
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    tmp_df = pd.read_csv('./TSLA.csv', index_col='Date', parse_dates=True,
                         usecols=['Date', 'Adj Close'], na_values=['nan'])
    tmp_df = tmp_df.rename(columns={'Adj Close': 'TSLA'})

    df = df.join(tmp_df)
    df = df.dropna(subset=["TSLA"])

    return df


if __name__ == "__main__":
    """
    This is the main method to get data, setup an agent, and simulate trading.
    """
    qlearning_values = []
    buyholdvalue = 0
    rounds = 10
    for i in range(0, rounds):
        print('Starting round: ' + str(i+1) + '/' + str(rounds))
        starting_cash = 100000
        max_position = 100
        symbol = 'TSLA'
        df = get_data()

        # Train agent, then get trades for target timeframe
        agent = Agent(max_position=max_position, starting_cash=starting_cash)
        agent.train(df, symbol)
        orders = agent.test(df, symbol)

        # simulate trades
        simulator = Sim(max_position=max_position, starting_cash=starting_cash)
        qvalue, buyholdvalue = simulator.simulate(df, symbol, orders, i==9)  # graph last round
        qlearning_values.append(qvalue)

    print("Policy has earned and average of: $" + str(round(np.average(qlearning_values)-starting_cash, 2)))
    print("BuyHold has earned: $" + str(round(buyholdvalue - starting_cash, 2)))
