import numpy as np
import matrix_tools as mt
EPSILON = 0.0001
from sklearn.datasets import make_blobs
import time
from scipy.sparse import csgraph

def array_to_adj_mat2(arr):
    arr_size = arr.shape[0]
    W = np.empty(shape=(arr_size, arr_size), dtype=np.float64)
    for i in range(arr_size):
        arr1 = np.zeros_like(arr, dtype=np.float64) + arr[i]
        arr2 = (arr1 - arr)**2
        arr3 = np.sqrt(arr2.sum(axis = 1))
        arr4 = np.exp(-arr3/2.)
        W[i] = arr4
    return np.round(W, 5)

def array_to_adj_mat(arr):
    arr_size = arr.shape[0]
    W = np.empty(shape=(arr_size, arr_size), dtype=np.float64)
    # TODO: currently implemented with loops - change it to calculate for the entire matrix
    for i in range(arr_size):
        for j in range(arr_size):
            W[i][j] = calc_weight(arr[i],arr[j])
    return W

def find_lnorm(arr):
    return mt.find_NGL2(arr) + np.identity(arr.shape[0]) #TODO: decide if to remove the identity matrix

def calc_weight(p1, p2):
    w = np.exp(-mt.eucledian_norm(p1-p2)/2)
    return w if w > EPSILON else 0


def calc_k(eigenval_mat,Ln):
    size = eigenval_mat.shape[0]
    max = 0
    k = 0
    eigvals = np.sort(np.diag(eigenval_mat))
    # print(np.sum(eigvals-np.sort(np.linalg.eigvals(Ln))))
    for i in range(1, int(size/2) + 1):
        eigendiff = abs(eigvals[i] - eigvals[i-1])
        if max < eigendiff:
            max = eigendiff
            k = i
    return k


def find_t(vec_mat, k, vec_idx_arr):
    size = vec_mat.shape[0]
    T = np.zeros(shape=(size, k))
    for i in range(size):
        norm_sum = np.sqrt((np.sum(vec_mat[i][vec_idx_arr.tolist()]**2)))
        T[i] = vec_mat[i][vec_idx_arr.tolist()]/norm_sum
    return T


def spectral_clustering(data,r,K):
    start = time.time()
    W = array_to_adj_mat2(data)
    end = time.time()
    print("W:",end - start)

    start = time.time()
    Ln = find_lnorm(W)
    end = time.time()
    print("Ln:",end - start)

    start = time.time()
    A, Q = mt.QR_iter(Ln)

    end = time.time()
    print("AQ:",end - start)

    start = time.time()
    k = K
    if r:
        k = calc_k(A,Ln)
        print(k)
    end = time.time()
    print("k:",end - start)
    vec_idx = np.argsort(np.diag(A))[0:k]  # array of the original idx of the sorted eigenvals

    start = time.time()
    T = find_t(Q, k, vec_idx)
    end = time.time()
    print("T:",end - start)
    return T, k

# TESTS ###################################################################
# dataMatrix, y = make_blobs(n_samples=30, centers=5, n_features=3, random_state = None)
#
#
#
# arr = np.array([[2,1,4],[1,1,1],[2,2,3]], dtype=np.float64)
# arr = dataMatrix
# # arr2 = np.array([[ 0.55794805, -0.25573179, -0.30606806],[-0.25573179,  0.53705136, -0.26732382],
# #                 [-0.30606806, -0.26732382,  0.5636308]], dtype=np.float64)
# #
# # print(arr)
# #
# start = time.time()
# # # W = array_to_adj_mat(arr)
# res ,k= spectral_clustering(arr,0)
# end = time.time()
# print(end - start)
# print (res)
# print(arr)
# print(arr[:,0])
#
# arr = np.array([[2,1,4],[1,1,1],[2,2,3]], dtype=np.float64)
#
# Q,R = mt.MGSA(arr)
# print(Q)
# print(R)
# Q2,R2 = mt.MGSA2(arr)
# print(Q2)
# print(R2)