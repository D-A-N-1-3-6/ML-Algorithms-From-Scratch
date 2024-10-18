import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SEED = 42
# Intuition
    # -> y = mX + b

# Creating the data

X, y = make_regression(n_samples=100, n_features=1, 
                       noise=1, random_state=SEED)

print(X.shape, y.shape)
# Imma do it in a jupyter notebook