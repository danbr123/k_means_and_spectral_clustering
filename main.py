import argparse

import matplotlib
import pandas as pd
import numpy as np
import random
import mykmeanssp as alg
from sklearn.datasets import make_blobs
from kmeans_init import k_means_pp
from output import outputFile, graphic


KUPPER2 = 10
KUPPER3 = 5
NUPPER2 = 1000
NUPPER3 = 500


def main():
    K, N, d, r, r_n, MAX_ITER = initialize()
    if K >= N:
        print("Error in data and/or parameters")
        return 0
    dataMatrix, y = make_blobs(n_samples=N, centers=K, n_features=d, random_state = r_n)
    dataList = dataMatrix.tolist()

    # First is the Spectral-Clustering-Final-Project

    Kfirstcentroids = k_means_pp(dataMatrix, K).tolist()
    Clusters_Spectral = alg.Kmeans(K, N, d, MAX_ITER, dataList, Kfirstcentroids)

    # Second is the K-means Algorithm
    Kfirstcentroids = k_means_pp(dataMatrix, K).tolist()
    Clusters_Kmeans = alg.Kmeans(K, N, d, MAX_ITER, dataList, Kfirstcentroids)

    outputFile(dataList, y, Clusters_Spectral, Clusters_Kmeans, K, N)
    graphic(dataMatrix, y, N, K, d, Clusters_Spectral, Clusters_Kmeans)

def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("r")
    args = parser.parse_args()
    K = int(args.k)
    N = int(args.n)
    r = args.r
    r_n = None
    if r == "False":
        r = False
    else:
        r_n = int(r)
        r = True
    d = 3  # random.randint(2, 3)
    MAX_ITER = 300
    if r:
        if d == 2:
            K = random.randint(KUPPER2 // 2, KUPPER2)
            N = random.randint(NUPPER2 // 2, NUPPER2)
        else:
            K = random.randint(KUPPER3 // 2, KUPPER3)
            N = random.randint(NUPPER3 // 2, NUPPER3)
    return K, N, d, True, 1, MAX_ITER # K, N, d, r, r_n, MAX_ITER

main()