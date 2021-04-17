import numpy as np
import copy, time

EPSILON = 0.001

def find_DDM2(mat):
    size = mat.shape[0]
    I = np.identity(size, dtype=np.float64)
    D_sum = mat.sum(axis=1) # sum columns
    return 1 / np.sqrt(D_sum) * I

def find_DDM(mat):
    size = mat.shape[0]
    D = np.zeros(shape=(size, size), dtype=np.float64)
    D_sq = np.zeros_like(D)
    for i in range(size):
        D[i][i] = np.sum(mat[i])
        D_sq[i][i] = 1 / np.sqrt(D[i][i])  # TODO: add exception
    return D, D_sq


def find_NGL2(mat):
    size = mat.shape[0]
    D_sq = find_DDM2(mat)
    L_norm = np.identity(size, dtype=np.float64) - np.dot(np.dot(D_sq, mat), D_sq)
    #arr_round = np.array(0.001 < abs(L_norm)) * 1.
    return L_norm# * arr_round


def find_NGL(mat):
    size = mat.shape[0]
    D_sq = find_DDM(mat)[1]
    L_norm = np.identity(size, dtype=np.float64) - np.dot(np.dot(D_sq, mat), D_sq)
    return np.round(L_norm, 5)


def MGSA2(mat): # Modified Gram-Schmidt Algorithm
    size = mat.shape[0]
    U = copy.deepcopy(mat)
    R = np.zeros_like(mat, dtype=np.float64)
    Q = np.zeros_like(mat, dtype=np.float64)

    # we will save in temporarily variables
    for i in range(size):
        col_u = U[:, i]
        r_ii = eucledian_norm(col_u)
        R[i][i] = r_ii
        col_q = col_u / r_ii
        Q[:, i] = col_q
        sub_matU = U[:,i+1:]
        row_r_ij = np.dot(col_q, sub_matU)
        R[i][i+1:] = row_r_ij # equals to j: from j=i+1 to j=n
        U[:,i+1:] = sub_matU - (col_q[:,np.newaxis]* row_r_ij) # equals to j: from j=i+1 to j=n
    return Q, R


def MGSA(mat, sumInit, sumQ,sumR,sumU):  # Modified Gram-Schmidt Algorithm
    size = mat.shape[0]
    U = copy.deepcopy(mat)
    R = np.zeros_like(mat, dtype=np.float64)
    Q = np.zeros_like(mat, dtype=np.float64)
    for i in range(size):
        R[i][i] = eucledian_norm(U[:, i])
        Q[:, i] = U[:, i] / R[i][i]
        for j in range(i + 1, size):
            R[i][j] = np.dot(Q[:, i], U[:, j])
            U[:, j] = U[:, j] - R[i][j] * Q[:, i]
    return Q, R, sumInit, sumQ,sumR,sumU


def QR_iter(mat):
    size = mat.shape[0]
    A_bar = copy.deepcopy(mat)
    Q_bar = np.identity(size, dtype=np.float64)
    for i in range(size):
        Q, R = MGSA2(A_bar)
        A_bar = np.dot(R, Q)
        if np.amax(abs(Q_bar - np.dot(Q_bar, Q))) <= EPSILON:
            return A_bar, Q_bar
        Q_bar = np.dot(Q_bar, Q)
    return A_bar, Q_bar


def eucledian_norm(arr):
    return np.sqrt(np.sum(arr ** 2))
