import numpy as np
import matrix_tools as mt
EPSILON = 0.0001
from sklearn.datasets import make_blobs
import time


def array_to_adj_mat2(array):
    arr_size = array.shape[0]
    W = np.empty(shape=(arr_size, arr_size), dtype=np.float64)
    for i in range(arr_size):
        arr1 = np.zeros_like(array) + array[i]
        arr2 = (arr1 - array)**2
        arr3 = arr2.sum(axis = 1)
        arr4 = np.exp(-arr3/2.)
        W[i] = arr4
    return W

def array_to_adj_mat(array):
    arr_size = array.shape[0]
    W = np.empty(shape=(arr_size, arr_size), dtype=np.float64)
    # TODO: currently implemented with loops - change it to calculate for the entire matrix
    for i in range(arr_size):
        for j in range(arr_size):
            W[i][j] = calc_weight(array[i],array[j])
    return W

def find_lnorm(array):
    return mt.find_NGL(array_to_adj_mat(array))

def calc_weight(p1, p2):
    w = np.exp(-mt.eucledian_norm(p1-p2)**2/2)
    return w if w > EPSILON else 0


def calc_k(eigenval_mat):
    size = eigenval_mat.shape[0]
    max = 0
    k = 0
    eigvals = np.sort(np.diag(eigenval_mat))
    for i in range(1, int(size/2) + 1):
        eigendiff = abs(eigvals[i] - eigvals[i-1])
        if max < eigendiff:
            max = eigendiff
            k = i
    return k


def find_t(vec_mat, k):
    size = vec_mat.shape[0]
    T = np.zeros(shape=(size, k))
    for i in range(size):
        norm_sum = np.sqrt((np.sum(vec_mat[i][0:k]**2)))
        T[i] = vec_mat[i][0:k]/norm_sum
    return T


def spectral_clustering(data):
    W = array_to_adj_mat2(data)
    Ln = find_lnorm(W)
    A, Q = mt.QR_iter(Ln)
    k = calc_k(A)
    T = find_t(Q,k)
    # TODO - call normal kmeans from here with T as the data
    #temporary:
    return T

# TESTS ###################################################################
dataMatrix, y = make_blobs(n_samples=1000, centers=5, n_features=3, random_state = 0)

arr = np.array([[2,1,4],[1,1,1],[2,2,3]], dtype=np.float64)
# arr2 = np.array([[ 0.55794805, -0.25573179, -0.30606806],[-0.25573179,  0.53705136, -0.26732382],
#                 [-0.30606806, -0.26732382,  0.5636308]], dtype=np.float64)
#
# print(arr)
#
start = time.time()
# # W = array_to_adj_mat(arr)
res = spectral_clustering(arr)
end = time.time()
print(end - start)
print(res)