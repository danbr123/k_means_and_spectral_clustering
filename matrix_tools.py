import numpy as np
import copy, time

EPSILON = 0.001


def find_DDM(mat):  # calculate the square root of the Diagonal Degree Matrix
    size = mat.shape[0]
    I = np.identity(size, dtype=np.float64)
    D_sum = mat.sum(axis=1)  # sum columns
    try:
        return 1 / np.sqrt(D_sum) * I
    except ZeroDivisionError:
        print("Error: Division by Zero while calculating the Diagonal Degree Matrix")
        exit()  # TODO ???

# def find_DDM(mat):  # OLD
#     size = mat.shape[0]
#     D = np.zeros(shape=(size, size), dtype=np.float64)
#     D_sq = np.zeros_like(D)
#     for i in range(size):
#         D[i][i] = np.sum(mat[i])
#         D_sq[i][i] = 1 / np.sqrt(D[i][i])
#     return D, D_sq


def find_NGL(mat):  # Normalized Graph Laplacian
    size = mat.shape[0]
    D_sq = find_DDM(mat)
    L_norm = np.identity(size, dtype=np.float64) - np.dot(np.dot(D_sq, mat), D_sq)
    # arr_round = np.array(0.001 < abs(L_norm)) * 1.
    return L_norm  # * arr_round


#
# def find_NGL(mat):  # OLD
#     size = mat.shape[0]
#     D_sq = find_DDM(mat)[1]
#     L_norm = np.identity(size, dtype=np.float64) - np.dot(np.dot(D_sq, mat), D_sq)
#     return np.round(L_norm, 5)


def MGS(mat):  # Modified Gram-Schmidt Algorithm
    size = mat.shape[0]
    # Initialize matrices
    U = copy.deepcopy(mat)
    R = np.zeros_like(mat, dtype=np.float64)
    Q = np.zeros_like(mat, dtype=np.float64)

    # we will save in temporarily variables
    for i in range(size):
        col_u = U[:, i]  # i'th column of U
        r_ii = euclidean_norm(col_u)  # set the diagonals of R to be the euclidean norm of col_u
        R[i][i] = r_ii
        try:
            col_q = col_u / r_ii
        except ZeroDivisionError:
            print("Error: Division by zero in MGS. input matrix column ", i, " is empty")
            print("Input Matrix: ", U)
            exit()  # TODO ???
            return
        Q[:, i] = col_q
        sub_matU = U[:, i + 1:]  # contains columns of U only from column i+1 to n=size(U)
        row_r_ij = np.dot(col_q, sub_matU)  # calculating R_ij for all j>i (value not affected by the updating of U
                                            # because it only uses one column at a time
        R[i][i + 1:] = row_r_ij  # equals to j: from j=i+1 to j=n
        U[:, i + 1:] = sub_matU - (col_q[:, np.newaxis] * row_r_ij)  # equals to j: from j=i+1 to j=n
    return Q, R


# def MGSA(mat, sumInit, sumQ,sumR,sumU):  # old MGS
#     size = mat.shape[0]
#     U = copy.deepcopy(mat)
#     R = np.zeros_like(mat, dtype=np.float64)
#     Q = np.zeros_like(mat, dtype=np.float64)
#     for i in range(size):
#         R[i][i] = eucledian_norm(U[:, i])
#         Q[:, i] = U[:, i] / R[i][i]
#         for j in range(i + 1, size):
#             R[i][j] = np.dot(Q[:, i], U[:, j])
#             U[:, j] = U[:, j] - R[i][j] * Q[:, i]
#     return Q, R, sumInit, sumQ,sumR,sumU


def QR_iter(mat):  # QR iteration algorithm
    # Initialize matrices
    size = mat.shape[0]
    A_bar = copy.deepcopy(mat)
    Q_bar = np.identity(size, dtype=np.float64)
    for i in range(size):
        Q, R = MGS(A_bar)  # Get Q,R from MGS
        A_bar = np.dot(R, Q)
        if np.amax(abs(Q_bar - np.dot(Q_bar, Q))) <= EPSILON:  # if the maximum difference between the old Q_bar and the
                                                               # new Q_bar is smaller than EPSILON, the algorithm stops
            return A_bar, Q_bar
        Q_bar = np.dot(Q_bar, Q)  # calculate the new Q_bar and continue
    return A_bar, Q_bar


def euclidean_norm(arr):  # calculate the euclidean norm l2 of an array
    return np.sqrt(np.sum(arr ** 2))
