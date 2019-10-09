#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:49:53 2019

@author: med-pvb
"""

# Reinforcement Learning

# Upper confidence bound (UCB)

# Import Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Import data
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Random selection (Just for reference)
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Implementing UCB
N = 10000
d = 10
ads_selected = []
n_selections = [0] * d # vector of d zeros
sum_rewards = [0] * d 
total_reward = 0
for n in range(0, N):
    ad = 0
    max_ub = 0
    for i in range(0, d):
        if (n_selections[i] > 0):
            average_reward = sum_rewards[i]/n_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / n_selections[i]) 
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_ub:
            max_ub = upper_bound
            ad = i
    ads_selected.append(ad)
    n_selections[ad] = n_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_rewards[ad] = sum_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualize results
plt.hist(ads_selected)
plt.title('Ads selections')
plt.xlabel('Ads')
plt.ylabel('Selection fraction')
plt.show()