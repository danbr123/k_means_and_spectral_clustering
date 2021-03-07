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
    return mt.find_NGL2(arr)

def calc_weight(p1, p2):
    w = np.exp(-mt.eucledian_norm(p1-p2)/2)
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


def spectral_clustering(data,k):
    K = k
    W = array_to_adj_mat2(data)
    Ln = find_lnorm(W)
    A, Q = mt.QR_iter(Ln)
    if k == 0:
        K = calc_k(A)
    T = find_t(Q,K)
    return T, K

# TESTS ###################################################################
dataMatrix, y = make_blobs(n_samples=10, centers=1, n_features=3, random_state = None)
#
arr = np.array([[-2,1,4],[1,1,1],[2,2,3]], dtype=np.float64)
arr2 = np.array([np.diag([1,4,5,6,7,8])])
arr3 = np.array([[0,1],[2,3]], dtype=np.float64)




# arr2 = np.array([[-2,1],[2,3]], dtype=np.float64)

# arr2 = np.array([[ 0.55794805, -0.25573179, -0.30606806],[-0.25573179,  0.53705136, -0.26732382],
#                 [-0.30606806, -0.26732382,  0.5636308]], dtype=np.float64)
#
# print(arr)
#
arr = dataMatrix
start = time.time()
# # W = array_to_adj_mat(arr)
res ,k= spectral_clustering(arr,0)
print(k)
end = time.time()
print(end - start)
# print(res)
# W = array_to_adj_mat(arr)
# W2 = array_to_adj_mat2(arr2)
# Ln = find_lnorm(W)
# Ln2 = find_lnorm(W2)
# A, Q = mt.QR_iter(arr)
# print(A)
# print(W2)
# print(Ln2)
# print(dataMatrix)

# arr = find_lnorm(W2)
# print(arr)
# evenVal = np.linalg.eigvals(arr)
# # print(evenVal)
# R,Q = np.linalg.qr(arr)
# # print(Q@R)
# A, Q = mt.QR_iter(arr)
# # print(A)
# Ln3 = csgraph.laplacian(W2, normed=True)
# print()
k=1
for i in range(10):
    dataMatrix, y = make_blobs(n_samples=30, centers=10, n_features=3, random_state = None)
    res ,k= spectral_clustering(dataMatrix,0)
    print(k)
