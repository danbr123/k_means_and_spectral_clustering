import numpy as np
import copy
EPSILON = 0.001

def find_DDM(mat):
    size = mat.shape[0]
    D = np.zeros(shape=(size, size), dtype=np.float16)
    D_sq = np.zeros_like(D)
    for i in range(size):
        D[i][i] = np.sum(mat[i])
        D_sq[i][i] = 1/np.sqrt(D[i][i]) # TODO: add exception
    return D, D_sq


def find_NGL(mat):
    size = mat.shape[0]
    D_sq = find_DDM(mat)[1]
    L_norm = np.identity(size) - np.dot(np.dot(D_sq, mat),D_sq)
    return L_norm


def MGSA(mat):  # Modified Gram-Schmidt Algorithm
    size = mat.shape[0]
    U = copy.deepcopy(mat)
    R = np.zeros_like(mat)
    Q = np.zeros_like(mat)
    for i in range(size):
        R[i][i] = eucledian_norm(U[:,i])
        Q[:,i] = U[:,i]/R[i][i]
        print("U[i]:",U[:,i])
        print("R[i][i]:", R[i][i])
        for j in range(i+1, size):
            R[i][j] = np.dot(Q[:,i], U[:,j])
            U[:,j] = U[:,j] - R[i][j]*Q[:,i]
    return Q, R

def QR_iter(mat):
    size = mat.shape[0]
    A_bar = copy.deepcopy(mat)
    Q_bar = np.identity(size)
    for i in range(size):
        Q,R = MGSA(A_bar)
        A_bar = np.dot(R,Q)
        if (abs(np.linalg.det(Q_bar) - np.linalg.det(np.dot(Q_bar,Q))) <= EPSILON):
            return A_bar, Q_bar
        Q_bar = np.dot(Q_bar,Q)
    return A_bar, Q_bar


def calc_det(mat): # TODO: finish
    size = mat.shape[0]
    if (size == 2):
        return mat[0][0]*mat[1][1] - mat[1][0]*mat[0][1]



def eucledian_norm(arr):
    return np.linalg.norm(arr, ord=2)  # TODO: write the formula instead



