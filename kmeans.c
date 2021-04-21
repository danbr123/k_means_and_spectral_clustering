#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Cluster Cluster;
typedef struct Point Point;

/* Methods declaration*/
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
void free_main(Point ** data, Point ** Kfirstcentroids, Cluster ** clusters,int K,int iter, int allocation_key);

/*Structs*/
struct Cluster{
    double* center;  /*List of center coordinates*/
    Point ** points;  /*List of points, each point is a list of doubles*/
    int idx; /*First EMPTY index in the points list, represents the "length" of the points array*/
    int dim;  /*Size of each point array and center array, must be the same size*/
};
struct Point{
    double* coordinates;  /*List of doubles*/
    int tag;  /*Tag is the index of the point as created in the raw data*/
};

/*Global variable*/
int memory_fault_flag = 0;

/*
 * Main function that get all the arguments and print K-clusters
 */
static Cluster** Cmain(int K, int N, int d, int MAX_ITER, Point ** data, Point ** Kfirstcentroids) {
    /*Variables and pointers*/
    int iteration =0;
    int equal = 0;
    int i,j;
    Cluster **clusters;
    double **prev_centers;

    /*Initialize K clusters*/
    clusters = firstKClusters(d,K,N, Kfirstcentroids);
    if (memory_fault_flag ==1) /*If there was memory fault*/
                return clusters;
    prev_centers = (double **)malloc(sizeof(double *)*(K));
    if (prev_centers == NULL){
        memory_fault_flag =1; /*Memory allocation failed - global update*/
        return clusters;
    }
    for(i=0;i<K;i++)
    {
        prev_centers[i]= (double *)malloc(sizeof(double)*d);
        if (prev_centers[i] == NULL){
            free_prev_centers(prev_centers,i);
            memory_fault_flag = 1; /*Memory allocation failed - global update*/
            return clusters;
        }
        for(j=0;j<d;j++){
            prev_centers[i][j] = clusters[i]->center[j];
        }
    }
    if (memory_fault_flag == 0){ /*If there wasn't memory fault*/
        /*Main algorithm loop*/
        while (iteration < MAX_ITER && equal == 0){
            clear_cluster(clusters, K); /*Clear cluster from previous points*/
            remember_centers(clusters, K, d, prev_centers); /*Prev_centers save cluster centers*/
            cluster_step(clusters, data, N, K); /*Step - calculate new cluster centers and distribute points to clusters*/
            equal = cluster_equal(clusters,prev_centers, K, d); /*Check if centers changed between iteration*/
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
    /*Variables and pointers*/
    int K, N, d, MAX_ITER, i, j,count=0;
    Cluster **clusters;
    PyObject *dataList, *KfirstcentroidsList, *dim0, *dim1, *Py_Clusters_List, *item;
    PyObject *mark = Py_BuildValue("i",-1);

    /*Python C-API - receive data from Python*/
    if(!PyArg_ParseTuple(args, "iiiiOO", &K, &N, &d, &MAX_ITER, &dataList, &KfirstcentroidsList)) {
        /* In the CPython API, a NULL value is never valid for a
                     PyObject* so it is used to signal that an error has occurred.*/
        printf("Error in data and/or parameters in the API stage\n");
        PyErr_SetString(PyExc_NameError,"Error in data and/or parameters in the API stage");
        PyErr_Occurred();
        return NULL;
        }

    /*Allocate memory and fill C arrays with data from Python*/
    Point ** data = (Point **)malloc(sizeof(Point*)*N);
    Point ** Kfirstcentroids = (Point **)malloc(sizeof(Point*)*K);
    if (data == NULL || Kfirstcentroids == NULL)
        memory_fault_flag =1; /*Memory allocation failed - global update*/
    else{
        for (i=0; i < N; i++) {
            data[i] = (Point *)malloc(sizeof(Point));
            if (data[i] == NULL){
                free_main(data, Kfirstcentroids, NULL, K, i, 0); /*0 is th allocation key -> memory fail in data[i]*/
                memory_fault_flag =1; /*Memory allocation failed - global update*/
                break;
            }
            data[i]->coordinates = (double*)calloc(d, sizeof(double));
            if (data[i]->coordinates == NULL){
                free_main(data, Kfirstcentroids, NULL, K, i, 1); /*1 is th allocation key -> memory fail in data[i]->coordinates*/
                memory_fault_flag =1; /*Memory allocation failed - global update*/
                break;
            }
            data[i]->tag = i;
            if(i<K){
                Kfirstcentroids[i] = (Point *)malloc(sizeof(Point));
                if (Kfirstcentroids[i] == NULL){
                    free_main(data, Kfirstcentroids, NULL, K, i, 2); /*2 is th allocation key -> memory fail in Kfirstcentroids[i]*/
                    memory_fault_flag =1; /*Memory allocation failed - global update*/
                    break;
                }
                Kfirstcentroids[i]->coordinates = (double*)calloc(d, sizeof(double));
                if (Kfirstcentroids[i]->coordinates == NULL){
                    free_main(data, Kfirstcentroids, NULL, K, i, 3); /*3 is th allocation key -> memory fail in Kfirstcentroids[i]->coordinates*/
                    memory_fault_flag =1; /*Memory allocation failed - global update*/
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
    if (memory_fault_flag ==0) /*If there wasn't memory fault*/
        clusters = Cmain(K, N, d, MAX_ITER, data, Kfirstcentroids);

    /*Build a Python list from clusters */
    if (memory_fault_flag ==0){ /*If there wasn't memory fault*/
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
        /* Free data, Kfirstcentroids and clusters*/
        free_main(data, Kfirstcentroids, clusters, K ,N ,0);
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
    {"Kmeans",                   /* The Python method name that will be used */
      (PyCFunction) Kmeans_main, /* The C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* Flags indicating parametersaccepted for this function */
      PyDoc_STR("K-Means Algorithm")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* Name of module */
    NULL, /* Module documentation, may be NULL */
    -1,  /* Size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    KmeansMethods /* The PyMethodDef array from before containing the methods of the extension */
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
/*Create a new cluster with a single point (for the first step of the algorithm). center is calculated as well*/
Cluster* create_cluster(Point * initial_point, int dim, int data_len){
    Cluster* c = malloc(sizeof(Cluster));
    if (c==NULL){
        memory_fault_flag =1; /*Memory allocation failed - global update*/
        return c;
    }
    c->center = (double*)calloc(dim, sizeof(double));
    c->points = (Point **)malloc(sizeof(Point*)*data_len);
    c->idx = 0;
    c->dim = dim;
    if (c->center == NULL || c->points == NULL){
        memory_fault_flag =1; /*Memory allocation failed - global update*/
        return c;
    }
    add_point_to_cluster(initial_point, c);
    update_cluster_center(c);
    return c;
}
/* Add a point to a cluster and update idx field*/
void add_point_to_cluster(Point * point, Cluster *c){
    c -> points[c -> idx] = point;
    c -> idx++;
}

/*Updates the cluster center according to the current points in it*/
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

/* Set cluster idx field to 0 and by that clear cluster from points*/
void clear_cluster(Cluster** c, int K){
    int i;
    for (i=0; i<K; ++i){
        c[i]->idx = 0; /*Clearing the idx instead of removing the points. points beyond the idx are not used*/
    }
}

/*Calculate distance between a point and a cluster*/
double find_dist_from_cluster(Point * point, Cluster* cluster){
    double dist = 0;
    int i;
    for (i=0; i<cluster->dim; ++i){
        double diff = (point->coordinates[i] - cluster->center[i]);
        dist += diff * diff;
    }
    return dist;
}

/*Find the cluster with minimum distance from a given point. k is the number of clusters.*/
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

/*Assign each point to its closest cluster, and when all points are assigned, update the cluster centers
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

/* Allocate Cluster in memory and fill them with each first point from 'Kfirstcentroids'*/
Cluster ** firstKClusters(int d, int K, int N, Point **data){
    int i,j;
    Cluster **clusters=(Cluster **)malloc(sizeof(Cluster*)*K);
    if (clusters == NULL)
        memory_fault_flag =1; /*Memory allocation failed - global update*/
    else{
        for (i = 0; i<K; i++){
            clusters[i] = create_cluster(data[i], d, N);
            if (memory_fault_flag ==1){ /*If there was memory fault*/
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

/* Deep copy from current clusters to temporary clusters*/
void remember_centers(Cluster** clusters, int K, int d, double ** prev_centers){
    int i,j;
    for(i=0;i<K;i++){
        for(j=0;j<d;j++){
            prev_centers[i][j] = clusters[i]->center[j];
        }
    }
}

/* Check if clusters hasn't change between iteration - comparison by each coordinate of cluster center*/
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

/* Free methods - free allocated memory on the heap*/
/* Free method for temporary centers - 'prev_centers'*/
void free_prev_centers(double ** prev_centers, int j){
    int i;
    for(i=0;i<j;i++){
        free(prev_centers[i]);
    }
    free(prev_centers);
}
/* Free method for original data and clusters  - 'data', 'Kfirstcentroids', 'clusters'*/
void free_main(Point ** data, Point ** Kfirstcentroids, Cluster ** clusters,int K,int iter, int allocation_key){
    int i;
    for (i=0; i < iter; i++) {
        free(data[i]->coordinates);
        free(data[i]);
        if(i<K && i<iter){
            free(Kfirstcentroids[i]->coordinates);
            free(Kfirstcentroids[i]);
        }
    }
    if(allocation_key == 1)
        free(data[iter]);
    if(allocation_key == 2){
        free(data[iter]->coordinates);
        free(data[iter]);
    }
    if(allocation_key == 3 && iter < K){
        free(data[iter]->coordinates);
        free(data[iter]);
        free(Kfirstcentroids[iter]);
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