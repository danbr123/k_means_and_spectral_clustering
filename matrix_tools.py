import numpy as np
import copy
import sys

EPSILON = 0.0001

'''
calculate the square root of the Diagonal Degree Matrix
'''


def find_DDM(mat):
    size = mat.shape[0]
    I = np.identity(size, dtype=np.float64)
    D_sum = mat.sum(axis=1)  # Sum columns
    if 0 in ((EPSILON < abs(np.sqrt(D_sum))) * 1.):  # Check if there's 0 on the diagonal
        print("Error: Division by Zero while calculating the Diagonal Degree Matrix")
        sys.exit()
    return (1 / np.sqrt(D_sum)) * I


'''
Normalized Graph Laplacian
'''


def find_NGL(mat):
    size = mat.shape[0]
    D_sq = find_DDM(mat)
    L_norm = np.identity(size, dtype=np.float64) - np.dot(np.dot(D_sq, mat), D_sq)
    return L_norm


'''
Modified Gram-Schmidt Algorithm
'''


def MGS(mat):
    size = mat.shape[0]
    # Initialize matrices
    U = copy.deepcopy(mat)
    R = np.zeros_like(mat, dtype=np.float64)
    Q = np.zeros_like(mat, dtype=np.float64)

    # We will save in temporarily variables
    for i in range(size):
        col_u = U[:, i]  # i'th column of U
        r_ii = euclidean_norm(col_u)  # Set the diagonals of R to be the euclidean norm of col_u
        R[i][i] = r_ii
        if r_ii == 0:
            print("Error: Division by zero in MGS. input matrix column ", i, " is empty")
            print("Input Matrix: ", U)
            sys.exit()
        col_q = col_u / r_ii
        Q[:, i] = col_q
        sub_matU = U[:, i + 1:]  # Contains columns of U only from column i+1 to n=size(U)
        row_r_ij = np.dot(col_q, sub_matU)  # Calculating R_ij for all j>i (value not affected by the updating of U
        # Because it only uses one column at a time
        R[i][i + 1:] = row_r_ij  # Equals to j: from j=i+1 to j=n
        U[:, i + 1:] = sub_matU - (col_q[:, np.newaxis] * row_r_ij)  # Equals to j: from j=i+1 to j=n
    return Q, R


'''
QR iteration algorithm
'''


def QR_iter(mat):
    # Initialize matrices
    size = mat.shape[0]
    A_bar = copy.deepcopy(mat)
    Q_bar = np.identity(size, dtype=np.float64)
    for i in range(size):
        Q, R = MGS(A_bar)  # Get Q,R from MGS
        A_bar = np.dot(R, Q)
        if np.amax(abs(Q_bar - np.dot(Q_bar, Q))) <= EPSILON:  # If the maximum difference between the old Q_bar and the
            # New Q_bar is smaller than EPSILON, the algorithm stops
            return A_bar, Q_bar
        Q_bar = np.dot(Q_bar, Q)  # Calculate the new Q_bar and continue
    return A_bar, Q_bar


'''
calculate the euclidean norm l2 of an array
'''


def euclidean_norm(arr):
    return np.sqrt(np.sum(arr ** 2))
