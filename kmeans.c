#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Cluster Cluster;
typedef struct Point Point;

/* methods decleration*/
double find_dist_from_cluster(Point * point, Cluster* cluster);
void add_point_to_cluster(Point * point, Cluster *c);
Cluster* create_cluster(Point * initial_point, int dim, int data_len);
void update_cluster_center(Cluster *c);
Cluster* find_min_dist_cluster(Cluster** cluster_array, Point* point, int k);
void cluster_step(Cluster** clusters, Point ** data, int points_count, int k);
void clear_cluster(Cluster** c, int K);
void remember_centers(Cluster** clusters, int K, int d, double ** prev_centers);
Cluster ** firstKClusters(int d, int K, int N, Point ** data);
int cluster_equal(Cluster ** clusters, double ** prev_centers, int K, int d);
void free_prev_centers(double ** prev_centers, int j);
void free_main(Point ** data, Point ** Kfirstcentroids, Cluster ** clusters,int K,int j, int h, int l,int m);

struct Cluster{
    double* center;  /*list of center coordinates*/
    Point ** points;  /*list of points, each point is a list of doubles*/
    int idx; /*first EMPTY index in the points list, represents the "length" of the points array*/
    int dim;  /*size of each point array and center array, must be the same size*/
};
struct Point{
    double* coordinates;  /*list of doubles*/
    int tag;  /*tag is the index of the point as created in the raw data*/
};

/*global variable*/
int memory_fault_flag = 0;

/*
 * Main function that get all the arguments and print K-clusters
 */
static Cluster** Cmain(int K, int N, int d, int MAX_ITER, Point ** data, Point ** Kfirstcentroids) {
    int iteration =0;
    int equal = 0;
    int i,j;
    Cluster **clusters;
    double **prev_centers;
    clusters = firstKClusters(d,K,N, Kfirstcentroids);
    if (memory_fault_flag ==1)
                return clusters;
    prev_centers = (double **)malloc(sizeof(double *)*(K));
    if (prev_centers == NULL){
        memory_fault_flag =1;
        return clusters;
    }
    for(i=0;i<K;i++)
    {
        prev_centers[i]= (double *)malloc(sizeof(double)*d);
        if (prev_centers[i] == NULL){
            free_prev_centers(prev_centers,i);
            memory_fault_flag = 1;
            return clusters;
        }
        for(j=0;j<d;j++){
            prev_centers[i][j] = clusters[i]->center[j];
        }
    }
    if (memory_fault_flag == 0){
        while (iteration < MAX_ITER && equal == 0){
            clear_cluster(clusters, K);
            remember_centers(clusters, K, d, prev_centers);
            cluster_step(clusters, data, N, K);
            equal = cluster_equal(clusters,prev_centers, K, d);
            iteration++;
        }
    }
    free_prev_centers(prev_centers,K);
    return clusters;
}

/*
 * This actually defines the Kmeans_main function using a wrapper C API function
 * The wrapping function needs a PyObject* self argument.
 * It has input PyObject *args from Python.
 */
static PyObject* Kmeans_main(PyObject *self, PyObject *args)
{
    int K, N, d, MAX_ITER, i, j,count=0;
    Cluster **clusters;
    PyObject *dataList, *KfirstcentroidsList, *dim0, *dim1, *Py_Clusters_List, *item;
    PyObject *mark = Py_BuildValue("i",-1);

    if(!PyArg_ParseTuple(args, "iiiiOO", &K, &N, &d, &MAX_ITER, &dataList, &KfirstcentroidsList)) {
        /* In the CPython API, a NULL value is never valid for a
                     PyObject* so it is used to signal that an error has occurred.*/
        printf("Error in data and/or parameters in the API stage\n");
        PyErr_SetString(PyExc_NameError,"Error in data and/or parameters in the API stage");
        PyErr_Occurred();
        return NULL;
        }

    Point ** data = (Point **)malloc(sizeof(Point*)*N);
    Point ** Kfirstcentroids = (Point **)malloc(sizeof(Point*)*K);
    if (data == NULL || Kfirstcentroids == NULL)
        memory_fault_flag =1;
    else{
        for (i=0; i < N; i++) {
            data[i] = (Point *)malloc(sizeof(Point));
            if (data[i] == NULL){
                free_main(data, Kfirstcentroids, NULL, K, i, i, i, i);
                memory_fault_flag =1;
                break;
            }
            data[i]->coordinates = (double*)calloc(d, sizeof(double));
            if (data[i]->coordinates == NULL){
                free_main(data, Kfirstcentroids, NULL, K, i+1, i, i, i);
                memory_fault_flag =1;
                break;
            }
            data[i]->tag = i;
            if(i<K){
                Kfirstcentroids[i] = (Point *)malloc(sizeof(Point));
                if (Kfirstcentroids[i] == NULL){
                    free_main(data, Kfirstcentroids, NULL, K, i+1, i+1, i, i);
                    memory_fault_flag =1;
                    break;
                }
                Kfirstcentroids[i]->coordinates = (double*)calloc(d, sizeof(double));
                if (Kfirstcentroids[i]->coordinates == NULL){
                    free_main(data, Kfirstcentroids, NULL, K, i+1, i+1, i+1, i);
                    memory_fault_flag =1;
                    break;
                }
                Kfirstcentroids[i]->tag = i;
            }
            for (j=0; j < d; j++){
                dim0 = PyList_GetItem(dataList,i);
                dim1 = PyList_GetItem(dim0,j);
                data[i]->coordinates[j] = PyFloat_AsDouble(dim1);
                if(i<K){
                    dim0 = PyList_GetItem(KfirstcentroidsList,i);
                    dim1 = PyList_GetItem(dim0,j);
                    Kfirstcentroids[i]->coordinates[j] = PyFloat_AsDouble(dim1);
                }
            }
        }
    }
    /*Call the C Kmean algorithm from task 1 */
    if (memory_fault_flag ==0)
        clusters = Cmain(K, N, d, MAX_ITER, data, Kfirstcentroids);

    /*Build a Python list from clusters */
    if (memory_fault_flag ==0){
        Py_Clusters_List = PyList_New(N+K);
        for (i=0; i<K; i++) {
            for(j=0;j<=clusters[i]->idx;j++){
                if(j<clusters[i]->idx){
                    item = Py_BuildValue("i",clusters[i]->points[j]->tag);
                    PyList_SetItem(Py_Clusters_List, count, item);
                }else
                    PyList_SetItem(Py_Clusters_List, count, mark);
                count++;
            }
        }
        /* free data, Kfirstcentroids and clusters*/
        free_main(data, Kfirstcentroids, clusters, K ,N ,N ,K ,K);
    }
    else{
        printf("Memory Allocation Error\n");
        PyErr_SetString(PyExc_NameError,"Memory Allocation Error");
        PyErr_Occurred();
        return NULL;
    }
    return Py_Clusters_List; /*  Py_BuildValue(...) returns a PyObject*  */
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef KmeansMethods[] = {
    {"Kmeans",                   /* the Python method name that will be used */
      (PyCFunction) Kmeans_main, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parametersaccepted for this function */
      PyDoc_STR("K-Means Algorithm")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    KmeansMethods /* the PyMethodDef array from before containing the methods of the extension */
};


PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}

/* -- Methods -- */
/*create a new cluster with a single point (for the first step of the algorithm). center is calculated as well*/
Cluster* create_cluster(Point * initial_point, int dim, int data_len){
    Cluster* c = malloc(sizeof(Cluster));
    if (c==NULL){
        memory_fault_flag =1;
        return c;
    }
    c->center = (double*)calloc(dim, sizeof(double));
    c->points = (Point **)malloc(sizeof(Point*)*data_len);
    c->idx = 0;
    c->dim = dim;
    if (c->center == NULL || c->points == NULL){
        memory_fault_flag =1;
        return c;
    }
    add_point_to_cluster(initial_point, c);
    update_cluster_center(c);
    return c;
}
/* add a point to a cluster and update idx field*/
void add_point_to_cluster(Point * point, Cluster *c){
    c -> points[c -> idx] = point;
    c -> idx++;
}

/*updates the cluster center according to the current points in it*/
void update_cluster_center(Cluster *c){
    int i,j;
    if(c->idx <= 0) return;
    for (i=0; i<c->dim; ++i){
        double new_val = 0;
        for (j=0; j<c->idx; ++j){
            new_val += c->points[j]->coordinates[i];
        }
        new_val = new_val/((double)c->idx);
        c->center[i] = new_val;
    }
}

/* set cluster idx field to 0 and by that clear cluster from points*/
void clear_cluster(Cluster** c, int K){
    int i;
    for (i=0; i<K; ++i){
        c[i]->idx = 0; /*clearing the idx instead of removing the points. points beyond the idx are not used*/
    }
}

/*calculate distance between a point and a cluster*/
double find_dist_from_cluster(Point * point, Cluster* cluster){
    double dist = 0;
    int i;
    for (i=0; i<cluster->dim; ++i){
        double diff = (point->coordinates[i] - cluster->center[i]);
        dist += diff * diff;
    }
    return dist;
}

/*find the cluster with minimum distance from a given point. k is the number of clusters.*/
Cluster* find_min_dist_cluster(Cluster** cluster_array, Point * point, int k){
    double min_dist = find_dist_from_cluster(point, cluster_array[0]);
    Cluster* min_cluster = cluster_array[0];
    int i;
    for (i=1; i<k; i++){
        double new_dist = find_dist_from_cluster(point, cluster_array[i]);
        if (new_dist < min_dist){
            min_dist = new_dist;
            min_cluster = cluster_array[i];
        }
    }
    return min_cluster;
}

/*assign each point to its closest cluster, and when all points are assigned, update the cluster centers
data is a list of pointers to point arrays, point count is the number of points, k is the number of clusters*/
void cluster_step(Cluster** clusters, Point ** data, int points_count, int k){
    int i, j;
    for (i=0; i<points_count; ++i){
        Cluster* min_c = find_min_dist_cluster(clusters, data[i], k);
        add_point_to_cluster(data[i], min_c);
    }
    for (j=0; j<k; ++j){
        update_cluster_center(clusters[j]);
    }
}

/* allocate Cluster in memory and fill them with each first point from 'Kfirstcentroids'*/
Cluster ** firstKClusters(int d, int K, int N, Point **data){
    int i,j;
    Cluster **clusters=(Cluster **)malloc(sizeof(Cluster*)*K);
    if (clusters == NULL)
        memory_fault_flag =1;
    else{
        for (i = 0; i<K; i++){
            clusters[i] = create_cluster(data[i], d, N);
            if (memory_fault_flag ==1){
                for (j = 0; j<i; j++){
                   free(clusters[j]->center);
                   free(clusters[j]->points);
                }
                free(clusters);
                return NULL;
            }
        }
    }
    return clusters;
}

/* deep copy from current clusters to temporary clusters*/
void remember_centers(Cluster** clusters, int K, int d, double ** prev_centers){
    int i,j;
    for(i=0;i<K;i++){
        for(j=0;j<d;j++){
            prev_centers[i][j] = clusters[i]->center[j];
        }
    }
}

/* check if clusters hasn't change between iteration - comparison by each coordinate of cluster center*/
int cluster_equal(Cluster ** clusters, double ** prev_centers, int K, int d) {
    double temp =0;
    int i,j;
    for (i = 0; i < K; i++) {
        for (j = 0; j < d; j++) {
            temp = prev_centers[i][j] - clusters[i]->center[j];
            if (temp > 0.0001 || temp < -0.0001)
                return 0;
        }
    }
    return 1;
}

/* free metods - free allocated memory on the heap*/
/* free method for temporary centers - 'prev_centers'*/
void free_prev_centers(double ** prev_centers, int j){
    int i;
    for(i=0;i<j;i++){
        free(prev_centers[i]);
    }
    free(prev_centers);
}
/* free method for original data and clusters  - 'data', 'Kfirstcentroids', 'clusters'*/
void free_main(Point ** data, Point ** Kfirstcentroids, Cluster ** clusters,int K,int j, int h, int l,int m){
    int i;
    for (i=0; i < j; i++) {
        if(i<h)
            free(data[i]->coordinates);
        free(data[i]);
        if(i<K && i<l){
            if(i<m)
                free(Kfirstcentroids[i]->coordinates);
            free(Kfirstcentroids[i]);
            }
    }
    free(data);
    free(Kfirstcentroids);
    if (clusters != NULL){
        for(i=0;i<K;i++)
        {
            free(clusters[i]->center);
            free(clusters[i]->points);
            free(clusters[i]);
        }
        free(clusters);
    }
}