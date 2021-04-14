import numpy as np

def k_means_pp(data, k):#TODO: print error if randon data generated contain less then k distinct points in space
    np.random.seed(0)  # use a specific seed to compare with the tester
    d = data.shape[1]
    n = data.shape[0]
    centroids_array = np.empty(shape=(k, d), dtype=np.float64)  # initialize list of centroids
    centroids_cnt = 0  # the current number of centroids in centroids_array
    di_array = np.empty(n, dtype=np.float64)
    centroid_idx = np.random.choice(n)
    centroids_array[0] = data[centroid_idx]  # initial centroid

    centroids_cnt += 1
    for i in range(1, k):
        di_array = calc_di(data, centroids_array, centroids_cnt, di_array)
        probability_array = update_prob_array(di_array)
        centroid_idx = np.random.choice(n, p=probability_array)
        centroids_array[i] = data[centroid_idx]
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

def update_prob_array(di_array):  # update the array of probabilities with current centroids
    # calc sum of D_i for all points
    di_sum = di_array.sum()
    if di_sum == 0:
        print("Error: Division by zero in the K-means++ initialization algorithm")
        print("Reason: There might be enough close samples to one another that results in K be bigger than the number "
              "of different samples in space")

    # update probability array
    return di_array/di_sum