"""
Indicators
CoreLogic - AI Coding Challenge
This class will take in a pandas dataframe and return the indicators used as states.
"""
import pandas as pd
import numpy as np

class Indicators:

    def __init__(self, SMA_Days=10, Bin_Size=32):
        self.SMA_Days = SMA_Days
        self.Bin_Size = Bin_Size
    
    def discretize(self, df, column):
        df.sort_values(column, inplace=True)
        df[column] = pd.cut(df[column], self.Bin_Size, labels=False)
        df.sort_index(inplace=True)
        return df[[column]]

    def compute_indicators(self, df, symbol='TSLA'):
        #Momentum
        momentum = []
        for i in range(len(df[symbol])):
            if i > 10:
                momentum.append(df[symbol][i] - df[symbol][(i-self.SMA_Days)])
            else:
                momentum.append(0)
                
        momentum = momentum / np.linalg.norm(momentum)
        df["Momentum"] = momentum
        
        #Simple Moving Avgerage
        sma = []
        for i in range(len(df[symbol])):
            if i > 10:
                sma.append((df[symbol][i] / df[symbol][i-self.SMA_Days: i].mean()) - 1)
            else:
                sma.append(0)
        sma = sma / np.linalg.norm(sma)
        df["SMA"] = sma
        
        #Bollinger Bands
        bb = []
        for i in range(len(df[symbol])):
            if i > 10:
                bb.append((df[symbol][i] - df[symbol][i-self.SMA_Days: i].mean()) / (2 *  np.std(df[symbol][i-self.SMA_Days: i])))
            else:
                bb.append(0)
        bb = bb / np.linalg.norm(bb)        
        df["BollingerBands"] = bb
        
        #Discretizing indicators to State 
        df["Momentum"] = self.discretize(df, "Momentum")
        df["SMA"] = self.discretize(df, "SMA")
        df["BollingerBands"] = self.discretize(df, "BollingerBands")
        
        #df["State"] = momentum # TODO: replace this with some useful indicator, and add others
        df["State"] = df.apply(func=lambda x: int(int(x.iloc[1]) << 6 | int(x.iloc[2])), axis=1)
        df.sort_values("State", inplace=True)
        df["State"] = pd.cut(df["State"], self.Bin_Size, labels=False)
        df.sort_index(inplace=True)

        return df[["State"]]

    """
    This will return the max size of the state
    The size will be the size of the cross product of the indicators.
    """
    def get_state_size(self):
        # TODO: Change this if you add more indicators
        return self.Bin_Size * 64
    
        
