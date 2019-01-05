# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 13:00:43 2017

@author: academy
"""
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import newton
from scipy.stats import norm

from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn
from scipy import array, newaxis

# Fetching data and initializing the IV column
nifty_data = pd.read_csv("data/data.csv")
nifty_data['IV'] = 0
print(nifty_data.head())


def black_shcoles_call(S, K, T, r, sigma):

    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))

    return call

def min_func(sigma, S, K, T, r, C):
    if sigma < 0:
        return np.inf
    bs_val = black_shcoles_call(S, K, T, r, sigma)
    return abs(bs_val - C)


def implied_sig(S, K, r, T, C):
    return newton(
        min_func,
        0.02,
        args = (
            S,
            K,
            T,
            r,
            C
        ),
        maxiter=10000,
    )

implied_vols = {}
for date in list(set(nifty_data['Date'])):
    implied_vols[date] = []


# Computing implied volatilities
for row in range(0,len(nifty_data)):
    underlyingPrice = nifty_data.iloc[row]['Underlying Value']
    strikePrice = nifty_data.iloc[row]['Strike Price']
    interestRate = 0
    daysToExpiration = nifty_data.iloc[row]['Time to Expiry']
    callPrice = nifty_data.iloc[row]['LTP']

    if float(callPrice) == 0.0:
        continue

    result = implied_sig(
        underlyingPrice,
        strikePrice,
        interestRate,
        daysToExpiration,
        callPrice
    )
    implied_vols[nifty_data.iloc[row]['Date']].append({
        "strikePrice" : strikePrice,
        "impliedVol" : result,
        "daysToExpiration" : daysToExpiration
    })


# Plotting the volatility smile
def Plot_smile(date):
    option_data = nifty_data[nifty_data['Date'] == date]
    plt.plot(option_data['Strike Price'],option_data['IV'])
    plt.legend(option_data['Date'])
    plt.ylabel('Implied Volatility')
    plt.xlabel('Strike Price')
    plt.show()

# Taking input date and calling the Plot_smile() function
def Take_input():
    smile_date = input("Enter the date for plotting Volatility Smile in the format dd-mm-yyyy: ")
    date_check = 0
    print(nifty_data['Date'])
    for date in nifty_data['Date']:
        print(date, smile_date)
        if date == smile_date:
            Plot_smile(smile_date)
            return
    print("\nKindly enter a valid date.\n")
    Take_input()

def plot_2D():
    dates = []
    for date, vols in implied_vols.items():
        x = [vol["strikePrice"] for vol in vols]
        y = [vol["impliedVol"] for vol in vols]
        plt.plot(x, y)
        plt.ylabel('Implied Volatility')
        plt.xlabel('Strike Price')
        dates.append(str(date) + ", T = " + str(vols[0]["daysToExpiration"]))
    plt.legend(dates)
    plt.show()

def plot_3D():
    X, Y, Z = [], [], []
    for date, vols in implied_vols.items():
        X += [vol["strikePrice"] for vol in vols]
        Y += [vol["daysToExpiration"] for vol in vols]
        Z += [vol["impliedVol"] for vol in vols]

    """
    X, Y, Z = np.array(X),np.array(Y), np.array(Z)
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z)
    plt.show()
    """

    #ax.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    plt.xlabel("K - Strike Price [USD $] ")
    plt.ylabel("T - Days to Expiration [Days] ")
    #plt.zlabel("$/sigma$ - Implied Volatility ")
    plt.show()


# Calling the Take_input() function
plot_3D()