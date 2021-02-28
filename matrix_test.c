#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct Matrix Matrix;

struct Matrix{
    double** mat;
    int size;
};

Matrix empty_mat(size){
    int i = 0;
    Matrix mat;
    mat.mat = calloc(size, sizeof(double*));
    for (i=0; i<size; ++i){
        mat.mat[i] = calloc(size, sizeof(double));
    }
    mat.size = size;
    return mat;
}

void free_mat(Matrix mat){
    int i;
    for (i=0; i<mat.size; ++i){
        free (mat.mat[i]);
    }
    free(mat.mat);
}

Matrix copy_mat(Matrix original){
    int i,j;
    Matrix new = empty_mat(original.size);
    for(i=0; i<original.size; ++i) {
        for (j = 0; j < original.size; ++j) {
            new.mat[i][j] = original.mat[i][j];
        }
    }
    return new;
}

double* get_col(Matrix mat, int idx){
    int i;
    double* vec = calloc(mat.size, sizeof(double));
    for (i=0; i<mat.size; ++i){
        vec[i] = mat.mat[i][idx];
    }
    return vec;
}

Matrix dot(Matrix mat1, Matrix mat2){ /* multiply 2 square matrices with the same size */
    int i,j,k;
    int size = mat1.size;
    /* create an empty matrix for the result size*size */
    Matrix res =  empty_mat(size);
    res.size = size;
    /* matrix multiplication */
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            for (k = 0; k < size; ++k) {
                res.mat[i][j] += mat1.mat[i][k] * mat2.mat[k][j];
            }
        }
    }
    return res;
}

double vec_mul(const double* vec1, const double* vec2, int size){
    double res = 0;
    int i;
    for (i=0; i<size; ++i) {
        res = res + vec1[i] * vec2[i];
    }
    return res;
}

void display(Matrix mat) {
    int size = mat.size;
    printf("\nOutput Matrix:\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%f  ", mat.mat[i][j]);
            if (j == size - 1)
                printf("\n");
        }
    }
}

double pseudorand(double max, int positive)
{
    if (positive) {
        return (max / RAND_MAX) * rand();
    }
    return (rand() > RAND_MAX / 2 ? -1 : 1) *(max / RAND_MAX) * rand();
}

Matrix gen_mat(int size, int max, int positive){
    int i,j;
    Matrix mat = empty_mat(size);
    for(j = 0; j<size; j++)
        for(i = 0; i<size; i++) {
            mat.mat[j][i] = pseudorand(max, positive);
        }
    mat.size = size;
    return mat;
}

double l2norm(double* vec, int size){
    int i;
    double res = 0;
    for (i=0; i<size; ++i){
        res += pow(vec[i], 2);
    }
    res = sqrt(res);
    return res;
}

void MGS(Matrix A, Matrix* Q, Matrix* R){
    int i,j,k;
    int size = A.size;
    /* line 1 */
    Matrix U = copy_mat(A);
    /* line 2 */
    for (i=0; i<U.size; ++i){
        /* line 3 */
        double* col = get_col(U,i);
        R->mat[i][i] = l2norm(col, size);
        /* line 4 */
        for (j=0; j<size; ++j){
            Q->mat[j][i] = col[j]/R->mat[i][i];
        }
        /* line 5 */
        for (j=i+1; j<size; ++j){
            /* line 6 */
            double* col_q = get_col(*Q,i);
            double* col_u = get_col(U,j);
            R->mat[i][j] = vec_mul(col_q, col_u, size);
            /* line 7 */
            for (k=0; k<size; ++k) {
                U.mat[k][j] = U.mat[k][j] - R->mat[i][j] * col_q[k];
            }
            free (col_q);
            free (col_u);
        }
    }
    free_mat(U);
}

int main() {
    int i = 0;
    int j = 0;
    Matrix mat1 = gen_mat(10,5, 1);
    Matrix mat2 = gen_mat(10,5, 1);
    Matrix mat3 = dot(mat1, mat2);
    Matrix mat4 = gen_mat(1000, 10, 1);
    Matrix Q = empty_mat(1000);
    Matrix R = empty_mat(1000);
    MGS(mat4, &Q, &R);
    return 0;
}
