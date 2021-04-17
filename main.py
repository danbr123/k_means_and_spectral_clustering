import argparse
import random
import mykmeanssp as alg
import sys
from sklearn.datasets import make_blobs
from kmeans_init import k_means_pp
from output import outputTextFiles, graphic
from spectral_clustering import spectral_clustering

KUPPER = 100
MAX_CAPACITY2 = 200  # 400 #on Nova - 390 - 395, on Mac 460
MAX_CAPACITY3 = 200  # 400 #on Nova - 395 - 400, on Mac 460

'''
    main method in the main module
    beginning of the software  
    call all other modules methods and in order to create the software product
'''
def main():
    K, N, d, r, MAX_ITER = initialize()
    if K >= N or K <= 0:
        print("Error in data and/or parameters")
        return 0
    dataMatrix, y = make_blobs(n_samples=N, centers=K, n_features=d,
                               random_state=0)  # TODO replace 1 in random_state = 1 with None
    dataList = dataMatrix.tolist()

    # Normalized Spectral Clustering Algorithm
    r = True  # TODO delete this line
    T, new_K = spectral_clustering(dataMatrix, r, K)
    Kfirstcentroids = k_means_pp(T, new_K).tolist()
    TList = T.tolist()
    Clusters_Spectral = alg.Kmeans(new_K, N, new_K, MAX_ITER, TList,
                                   Kfirstcentroids)  # TODO #is it intentional that we pass d as new_k

    # K-means Algorithm
    Kfirstcentroids = k_means_pp(dataMatrix, new_K).tolist()
    Clusters_Kmeans = alg.Kmeans(new_K, N, d, MAX_ITER, dataList, Kfirstcentroids)

    outputTextFiles(dataList, y, Clusters_Spectral, Clusters_Kmeans, new_K, N)  # TODO check if new_K or K
    graphic(dataMatrix, y, N, K, new_K, d, Clusters_Spectral, Clusters_Kmeans)

'''
    initialize method to read variables from the user (pass from task.py)
    initialize software starter variables - K, N, d, r, MAX_ITER
'''
def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("r")
    args = parser.parse_args()
    K = int(args.k)
    N = int(args.n)
    r = args.r
    if r == "True":
        r = True
    else:
        r = False
    d = 3  # random.randint(2, 3) Todo: remove #
    MAX_ITER = 300
    if r:
        if d == 2:
            K = random.randint(KUPPER // 2, KUPPER)
            N = random.randint(K + 1, MAX_CAPACITY2)  # TODO: #N = random.randint(NUPPER2 // 2, NUPPER2)
        else:
            K = random.randint(KUPPER // 2, KUPPER)
            N = random.randint(K + 1, MAX_CAPACITY3)  # TODO: #N = random.randint(NUPPER3 // 2, NUPPER3)
    return K, N, d, r, MAX_ITER

main()
