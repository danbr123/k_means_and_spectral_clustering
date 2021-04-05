import numpy as np
import copy

EPSILON = 0.001


def find_DDM2(mat):
    size = mat.shape[0]
    I = np.identity(size, dtype=np.float64)
    D_sum = mat.sum(axis=1)
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
    arr_round = np.array(0.001 < abs(L_norm)) * 1.
    return L_norm * arr_round


def find_NGL(mat):
    size = mat.shape[0]
    D_sq = find_DDM(mat)[1]
    L_norm = np.identity(size, dtype=np.float64) - np.dot(np.dot(D_sq, mat), D_sq)
    return np.round(L_norm, 5)


def MGSA2(mat):  # Modified Gram-Schmidt Algorithm
    size = mat.shape[0]
    T = np.triu(np.ones(shape=(size, size)), 0)
    U = copy.deepcopy(mat)
    R = np.zeros_like(mat, dtype=np.float64)
    Q = np.zeros_like(mat, dtype=np.float64)
    On = np.ones(mat.shape[0])
    for i in range(size):
        R[i][i] = eucledian_norm(U[:, i])
        Q[:, i] = U[:, i] / R[i][i]
        R[i] = np.dot(Q[:, i], U) * T[i, :]
        U = U - ((Q[:, i].reshape(mat.shape[0],1) * R[i]) * On)
        On[i] = 0
    return Q, R


def MGSA(mat):  # Modified Gram-Schmidt Algorithm
    size = mat.shape[0]
    U = copy.deepcopy(mat)
    R = np.zeros_like(mat, dtype=np.float64)
    Q = np.zeros_like(mat, dtype=np.float64)
    for i in range(size):
        R[i][i] = eucledian_norm(U[:, i])
        Q[:, i] = U[:, i] / R[i][i]
        for j in range(i + 1, size):
            R[i][j] = np.dot(Q[:, i].T, U[:, j])
            U[:, j] = U[:, j] - R[i][j] * Q[:, i]
    return Q, R


def QR_iter(mat):
    size = mat.shape[0]
    A_bar = copy.deepcopy(mat)
    Q_bar = np.identity(size, dtype=np.float64)
    for i in range(size):
        Q, R = MGSA2(A_bar)
        A_bar = np.dot(R, Q)
        # if i%10==0:
        #     print(i, np.amax(abs(Q_bar - np.dot(Q_bar,Q))))
        if (np.amax(abs(Q_bar - np.dot(Q_bar, Q))) <= EPSILON):
            return A_bar, Q_bar
        Q_bar = np.dot(Q_bar, Q)
    return A_bar, Q_bar


def eucledian_norm(arr):
    # return np.linalg.norm(arr)
    return np.sqrt(np.sum(arr ** 2))
