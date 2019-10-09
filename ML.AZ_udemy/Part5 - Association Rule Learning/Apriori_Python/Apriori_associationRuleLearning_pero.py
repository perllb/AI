#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:23:04 2019

@author: med-pvb
"""

#%reset -f

# Association Rule Learning
# Apriori

# Import libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# Training apriori model
from apyori import apriori 
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, max_length = 2)    

# Visualizing Results
results = list(rules)


# graphs only work well with very few rules
subrules2 <- sample(rules, 10)
plot(subrules2, method="graph")
## igraph layout generators can be used (see ? igraph::layout_)
plot(subrules2, method="graph", control=list(layout=igraph::in_circle()))
plot(subrules2, method="graph", control=list(
  layout=igraph::with_graphopt(spring.const=5, mass=50)))

plot(subrules2, method="graph", control=list(type="itemsets"))
## try: plot(subrules2, method="graph", interactive=TRUE)
## try: plot(subrules2, method="graph", control=list(engine="graphviz"))

