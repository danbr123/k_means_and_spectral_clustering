import argparse
import pandas as pd
import numpy as np
import random
import mykmeanssp as alg
from sklearn.datasets import make_blobs
KUPPER2 = 10
KUPPER3 = 5
NUPPER2 = 1000
NUPPER3 = 500

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("k", type=int)
    # parser.add_argument("n", type=int)
    # parser.add_argument("r")
    # args = parser.parse_args()
    # K = int(args.k)
    # N = int(args.n)
    # r = args.r
    r = False
    d = 3 #random.randint(2, 3)
    K = 5
    N = 500
    MAX_ITER = 300
    if r:
        if d==2:
            K = random.randint(KUPPER2/2, KUPPER2)
            N = random.randint(KUPPER2/2, KUPPER2)
        else:
            K = random.randint(KUPPER3 / 2, KUPPER3)
            N = random.randint(NUPPER3 / 2, NUPPER3)
    dataMetrix, y = make_blobs(n_samples=N, centers=K, n_features=d,random_state = 0)
    dataList = dataMetrix.tolist()

    # First is the Spectral-Clustering-Final-Project
    Kfirstcentroids = k_means_pp(dataMetrix, K).tolist()
    Clusters_Spectral = np.full((K, d), 1)
    print(Clusters_Spectral)
        #alg.Kmeans(K, N, d, MAX_ITER, dataList, Kfirstcentroids)

    # Second is the K-means Algorithm
    Kfirstcentroids = dataList[0:K:]
    Clusters_Kmeans = alg.Kmeans(K, N, d, MAX_ITER, dataList, Kfirstcentroids)

    outputFile(dataList, y, Clusters_Spectral, Clusters_Kmeans, K, N, d)

def k_means_pp(data, k):
    return np.array([1])

def outputFile(dataList, y, Clusters_Spectral, Clusters_Kmeans, K, N, d):
    dataFile = open('data.txt', 'w')
    clustersFile = open('clusters.txt', 'w')
    for i in range(N):
        dataFile.write(point_to_string(dataList[i]) + "," + str(y[i])+"\n")
    clustersFile.write(str(K)+"\n")



def point_to_string(lst):
    str_point = ""
    for t in lst:
        str_point += str(np.float64(t)) + ","
    return str_point[:-1]
main()