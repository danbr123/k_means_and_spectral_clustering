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


# TESTS ###################################################################
dataMatrix, y = make_blobs(n_samples=1000, centers=5, n_features=3, random_state = 0)

arr = np.array([[2,1,4],[1,1,1],[2,2,3]], dtype=np.float64)
# arr2 = np.array([[ 0.55794805, -0.25573179, -0.30606806],[-0.25573179,  0.53705136, -0.26732382],
#                 [-0.30606806, -0.26732382,  0.5636308]], dtype=np.float64)
#
arr= dataMatrix
# print(arr)
#
start = time.time()
# # W = array_to_adj_mat(arr)
W2 = array_to_adj_mat2(arr)
end = time.time()
print(end - start)
# # print(W)
# # print(W2)
start = time.time()
#
# # print("___")
# # ln = mt.find_NGL(W)
ln2 = mt.find_NGL2(W2)
end = time.time()
print(end - start)
start = time.time()
#
# # print("____")
# # Q,R = mt.QR_iter(ln)
# Q2,R2 = mt.QR_iter(ln2)
end = time.time()
print(end - start)
start = time.time()

# print(Q)
# print(R)
# eval, evec = np.linalg.eig(ln2)
# print(np.dot(evec[0], evec[2]))
# print(np.dot(Q[0],Q[1]))
# print("________________")
# print(ln)
Qg, Rg = mt.MGSA(ln2)
end = time.time()
print(end - start)
# print("arr2")
# print(arr2)
# Qg2, Rg2 = mt.MGSA(arr2)
# print("test")
# print (Qg-Qg2)
# print (Rg-Rg2)
# print(Qg[0]@Qg[2])
# print(Qg @ Qg.T)
# print(Qg.T @ Qg)
# print(eval)
# print(evec)
# # print(Q)
# print(Q2)
# print("________________")
# # print(R)
# print(R2)



#arr = np.random.rand(500, 3)
#res = np.random.rand(1000, 1000)

# res = array_to_adj_mat(arr)
# res = find_lnorm(res)
# print("___")
# print(res)
# print("___")
# Q,R = mt.QR_iter(res)
# print(Q)
# print("___")
# print(R)
# print("___")
# print(calc_k(R))
# print (np.linalg.eig(res))
#
# print("_____")
# print (Q,R)
# print("_____")
# print(np.dot(np.transpose(Q),Q))
# print("_____")

dataMatrix, y = make_blobs(n_samples=5, centers=5, n_features=3, random_state = 0)
# a = (dataMatrix[0][0] - dataMatrix[1][0])**2 + (dataMatrix[0][1] - dataMatrix[1][1])**2
# +(dataMatrix[0][2] - dataMatrix[1][2])**2
# a = np.exp(-a/2)
# print(a)
# print(dataMatrix)
# arr1 = array_to_adj_mat(dataMatrix)
# print(arr1)
# start = time.time()
# arr2 = array_to_adj_mat2(dataMatrix)
# end = time.time()
# print(arr2)
# # print(end - start)



