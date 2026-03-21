import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, y = make_blobs(n_samples=300, n_features=1, centers=3, cluster_std=1.0, random_state=42)