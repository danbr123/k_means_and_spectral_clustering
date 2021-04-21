import numpy as np
import matrix_tools as mt

EPSILON = 0.0001

'''
convert generated data into Weighted Adjacency Matrix
'''


def array_to_adj_mat(arr):
    arr_size = arr.shape[0]
    W = np.empty(shape=(arr_size, arr_size), dtype=np.float64)
    for i in range(arr_size):
        arr1 = np.zeros_like(arr, dtype=np.float64) + arr[i]  # contain arr[i] in every row
        # calculate weight for each i,j with numpy efficiency
        arr2 = (arr1 - arr) ** 2
        arr3 = np.sqrt(arr2.sum(axis=1))
        arr4 = np.exp(-arr3 / 2.)
        W[i] = arr4
        W[i][i] = 0  # cancel inner loop
    return W


'''
find the index correspond to the largest Eigen-gap 
'''


def calc_k(eigenval_mat):
    size = eigenval_mat.shape[0]
    max = 0
    k = 0
    eigvals = np.sort(np.diag(eigenval_mat))  # extract the sorted eigen values from the matrix calculated in QR_iter

    for i in range(1, int(size / 2) + 1):
        eigendiff = abs(eigvals[i] - eigvals[i - 1])
        if max < eigendiff:
            max = eigendiff
            k = i
    return k


'''
calculate the matrix T, uses a vector of the original indexes of each eigenval
in the sorted eigenval array to sort the eigen vectors accordingly
'''


def find_t(vec_mat, k, vec_idx_arr):
    size = vec_mat.shape[0]
    T = np.zeros(shape=(size, k))
    for i in range(size):
        norm_sum = np.sqrt((np.sum(vec_mat[i][vec_idx_arr.tolist()] ** 2)))  # calculating the value required to
        # normalize the row
        if norm_sum > EPSILON:
            T[i] = vec_mat[i][vec_idx_arr.tolist()] / norm_sum  # normalizing U by the norm_sum to get T
    return T


'''
process the data to calculate T and K, and return them
'''


def spectral_clustering(data, r, K):
    # Calculate Weighted Adjacency Matrix
    W = array_to_adj_mat(data)

    # Calculate Normalized Graph Laplacian
    Ln = mt.find_NGL(W)

    # Calculate A, Q from the QR iteration algorithm
    A, Q = mt.QR_iter(Ln)

    # Calculate new K from the eigen-gap
    k = K
    if r:
        k = calc_k(A)

    # calculate the matrix T
    vec_idx = np.argsort(np.diag(A))[0:k]  # Array of the original idx of the sorted eigenvalues
    T = find_t(Q, k, vec_idx)
    return T, k
