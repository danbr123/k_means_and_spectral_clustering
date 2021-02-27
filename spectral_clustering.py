import numpy as np
import matrix_tools as mt
EPSILON = 0.0001


def array_to_adj_mat(array):
    arr_size = array.shape[0]
    W = np.empty(shape=(arr_size, arr_size), dtype=np.float64)
    # TODO: currently implemented with loops - change it to calculate for the entire matrix
    for i in range(arr_size):
        for j in range(arr_size):
            W[i][j] = calc_weight(arr[i],arr[j])
    return W

def find_lnorm(array):
    return mt.find_NGL(array_to_adj_mat(array))

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


arr = np.array([[0,1,0],[1,1,1],[0,2,0],[0,0,0],[0,1,4],[70,70,70],[75,75,75],[80,80,80]])
#arr = np.random.rand(500, 3)
res = find_lnorm(arr)
#res = np.array([[1,3,5],[2,4,6],[3,3,4]], dtype=np.float64)
#res = np.random.rand(1000, 1000)

print(res)
Q,R = mt.QR_iter(res)
print(Q)
print("___")
print(R)
print(calc_k(R))
#
# print("_____")
# print (Q,R)
# print("_____")
# print(np.dot(np.transpose(Q),Q))
# print("_____")
