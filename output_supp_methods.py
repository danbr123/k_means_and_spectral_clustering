import numpy as np

''' 
    input is a list of points indexes. character '-1' is used to separate cluster belonging
    create a string of points indexes. each line represent different cluster.
'''
def clusterReader(Clusters_List):
    s = ""
    for i in Clusters_List:
        if i != -1:
            s = s + str(i) + ","
        else:
            s = s[:-1] + "\n"
    return s


''' 
    input is a list that represent a point
    return a string represent the point in the right format
'''
def point_to_string(lst):
    str_point = ""
    for t in lst:
        str_point += str(np.round(t, 8)) + ","
    return str_point[:-1]


''' 
    input is a list of points indexes. character '-1' is used to separate cluster belonging
    create a list of points indexes for each cluster. wrap all lists with another single list.
'''
def RowIdxList(K, Clusters_List):
    lst = []
    for i in range(K):
        idx = Clusters_List.index(-1) # -1 char is used to separate points between clusters in the 1'd list that
        # returned from C-API
        rowidx = np.array(Clusters_List[:idx])
        Clusters_List = Clusters_List[idx + 1:]
        lst.append(rowidx)
    return lst


''' 
    input is:
        - y array - y[i] contain cluster index for the i'th point of the generated data
        - lst list - lst[i] contain numpy array with points that was classified to i'th cluster.
                     points are represent as indexes as they appear in the generated data.
                     
    calculate Jaccard Measure for cluster classification of a certain algorithm
'''
def JaccardMeasure(y, lst, N, K):
    sum_std = 0  # std - for standard classification to cluster from make.blob
    sum_alg = 0  # ald - for algorithm classification to cluster
    nC2 = np.frompyfunc(lambda n: int(n * (n - 1) / 2), 1, 1)  # n choose 2 function

    # We will use matrix multiplication to check overlapping points
    mat_std = np.zeros(shape=(K, N))
    mat_alg = np.zeros(shape=(K, N))
    for i in range(K):
        arr_std = np.array(y[:] == i) * 1. # 1 if point belong to i'th cluster in y vector, otherwise 0
        mat_std[i] = arr_std # Update standard matrix
        for j in lst[i]: # 1 if point belong to i'th cluster - directly from lst[i] vector, otherwise 0
            mat_alg[i][j] = 1 # Update standard matrix

        # Sum pairs in each cluster for standard and algorithm clustering separately
        sum_std += nC2(np.sum(arr_std))
        sum_alg += nC2(np.sum(mat_alg[i]))

    mat_both = mat_std @ mat_alg.T # Matrix multiplication leave us with the overlapping points
    mat_sum = np.sum(nC2(mat_both))  # Sum all pairs both in standard and algorithm clustering

    # Upon return we sum pairs in the standard clustering and the algorithm clustering and subtract the shared pairs
    # of the two in order to avoid counting twice shared pairs.
    if mat_sum == 0: # If mat_sum is 0 may lead to division by zero Error
        return 0
    return mat_sum / (sum_std + sum_alg - mat_sum)
