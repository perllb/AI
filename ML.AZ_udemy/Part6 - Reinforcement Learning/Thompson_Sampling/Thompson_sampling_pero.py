#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:17:09 2019

@author: med-pvb
"""

# Reinforcement Learning

# Thompson Sampling

# Import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Get data
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Random selection (Just for reference)
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

# Implementing Thompson Sampling
N = 10000
d = 10
ads_selected = []
n_rewards_1 = [0] * d  # vector of d zeros 
n_rewards_0 = [0] * d 
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(n_rewards_1[i] + 1, n_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        n_rewards_1[ad] = n_rewards_1[ad] + 1 
    else:
        n_rewards_0[ad] = n_rewards_0[ad] + 1 
    total_reward = total_reward + reward

# Visualize results
plt.hist(ads_selected)
plt.title('Ads selections')
plt.xlabel('Ads')
plt.ylabel('Selection fraction')
plt.show()