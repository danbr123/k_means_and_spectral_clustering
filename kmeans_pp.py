import argparse
import pandas as pd
import numpy as np
import mykmeanssp as alg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("argv", nargs='+')
    res = parser.parse_args()
    args = vars(res)["argv"]
    K = int(args[0])
    N = int(args[1])
    d = int(args[2])
    MAX_ITER = int(args[3])
    filename = args[4]
    dataPD = pd.read_csv(filename, names=[i for i in range(d)]) #read file
    dataNp = dataPD.values #create a double numpy array (class 'numpy.float64')
    dataList = dataPD.values.tolist();

    if dataNp.size == 0 or np.isnan(dataNp).any() or np.size(dataNp, 0) != N or np.size(dataNp, 1) != d\
            or K >= N or len(args) != 5:
        print("Error in data and/or parameters")
    else:
        Kfirstcentroids = k_means_pp(dataNp, K).tolist()
        ClustersPoint = alg.Kmeans(K,N,d,MAX_ITER, dataList, Kfirstcentroids)
        for i in range(K):
            print(point_to_string(ClustersPoint[i*d:(i+1)*d]))


def k_means_pp(data, k):
    np.random.seed(0)  # use a specific seed to compare with the tester
    d = data.shape[1]
    n = data.shape[0]
    centroids_array = np.empty(shape=(k, d), dtype=np.float64)  # initialize list of centroids
    centroids_cnt = 0  # the current number of centroids in centroids_array
    di_array = np.empty(n, dtype=np.float64)
    centroid_idx = np.random.choice(n)
    centroids_array[0] = data[centroid_idx]  # initial centroid

    #  print idx of the first centroid, newline if there is only one
    if k != 1:
        print(centroid_idx, end=",")
    else:
        print(centroid_idx)

    centroids_cnt += 1
    for i in range(1, k):
        di_array = calc_di(data, centroids_array, centroids_cnt, di_array)
        probability_array = update_prob_array(data, di_array)
        centroid_idx = np.random.choice(n, p=probability_array)
        centroids_array[i] = data[centroid_idx]

        # printing the idx of the centroids to the first line. if its the last centroid create new line.
        if i < k-1:
            print(centroid_idx, end=",")
        else:
            print(centroid_idx)
        centroids_cnt += 1
    return centroids_array


def calc_di(data, centroids, count, di_array):
    if count == 1:
        table2 = np.tile(centroids[count - 1], (data.shape[0], 1))
        return ((data - table2) ** 2).sum(axis=1)

    table2 = np.tile(centroids[count-1], (data.shape[0], 1))
    row_sum = ((data - table2)**2).sum(axis=1)
    row_bigger = (di_array < row_sum)*1
    return di_array * row_bigger + (1-row_bigger) * row_sum

def update_prob_array(data, di_array):  # update the array of probabilities with current centroids
    # calc sum of D_i for all points
    di_sum = di_array.sum()
    # update probability array
    return di_array/di_sum

def point_to_string(lst):
    str_point = ""
    for t in lst:
        str_point += str(np.float64(t)) + ","
    return str_point[:-1]

main()

