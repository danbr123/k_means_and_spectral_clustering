#include <stdio.h>
#include <stdlib.h>

typedef struct Cluster Cluster;
double find_dist_from_cluster();
void add_point_to_cluster();
Cluster* create_cluster();
void print_point(double* point, int dim);
void print_cluster(Cluster* c);
void update_cluster_center(Cluster *c);
Cluster* find_min_dist_cluster(Cluster** cluster_array, double* point, int k);
void cluster_step(Cluster** clusters, double** data, int points_count, int k);
void clear_cluster(Cluster** c, int K);
void print_clusters_centers(Cluster ** clusters, int K, int d);
void remember_centers(Cluster** clusters, int K, int d, double ** prev_centers);
Cluster ** firstKClusters(int d, int K, int N, double **data);
double** read_from_file(int n, int d);
int cluster_equal(Cluster ** clusters, double ** prev_centers, int K, int d);


struct Cluster{
    double* center;  /*list of center coordinates*/
    double** points;  /*list of points, each point is a list of doubles*/
    int idx; /*first EMPTY index in the points list, represents the "length" of the points array*/
    int dim;  /*size of each point array and center array, must be the same size*/
};




int main(int argc, char* argv[]) {
    int K = atoi(argv[1]);
    int N = atoi(argv[2]);
    int d = atoi(argv[3]);
    int MAX_ITER = atoi(argv[4]);
    int iteration =0;
    int equal = 0;
    int i,j;
    Cluster **clusters;
    double **prev_centers;
    double **data = read_from_file(N, d);
    if (data == NULL || K>=N || argc != 5){
        printf("Error in data and/or parameters");
        return 0;
    }
    clusters = firstKClusters(d,K,N, data);
    prev_centers = (double **)malloc(sizeof(double *)*(K));

    for(i=0;i<K;i++)
    {
        prev_centers[i]=(double *)malloc(sizeof(double)*d);
        for(j=0;j<d;j++){
            prev_centers[i][j] = clusters[i]->center[j];
        }
    }
    while (iteration < MAX_ITER && equal == 0){
        clear_cluster(clusters, K);
        remember_centers(clusters, K, d, prev_centers);
        cluster_step(clusters, data, N, K);
        equal = cluster_equal(clusters,prev_centers, K, d);
        iteration++;
    }
    print_clusters_centers(clusters, K, d);
    return 0;
}


/*create a new cluster with a single point (for the first step of the algorithm). center is calculated as well*/
Cluster* create_cluster(double* initial_point, int dim, int data_len){
    Cluster* c = malloc(sizeof(Cluster));
    c->center = (double*)calloc(dim, sizeof(double));
    c->points = calloc(data_len, sizeof(double*));
    c->idx = 0;
    c->dim = dim;
    add_point_to_cluster(initial_point, c);
    update_cluster_center(c);
    return c;
}

void add_point_to_cluster(double* point, Cluster *c){
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
            new_val += c->points[j][i];
        }
        new_val = new_val/((double)c->idx);
        c->center[i] = new_val;
    }
}

void clear_cluster(Cluster** c, int K){
    int i;
    for (i=0; i<K; ++i){
        c[i]->idx = 0; /*clearing the idx instead of removing the points. points beyond the idx are not used*/

    }
}

/*calculate distance between a point and a cluster*/
double find_dist_from_cluster(double* point, Cluster* cluster){
    double dist = 0;
    int i;
    for (i=0; i<cluster->dim; ++i){
        double diff = (point[i] - cluster->center[i]);
        dist += diff * diff;
    }
    return dist;
}

/*find the cluster with minimum distance from a given point. k is the number of clusters.*/
Cluster* find_min_dist_cluster(Cluster** cluster_array, double* point, int k){
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
void cluster_step(Cluster** clusters, double** data, int points_count, int k){
    int i, j;
    for (i=0; i<points_count; ++i){
        Cluster* min_c = find_min_dist_cluster(clusters, data[i], k);
        add_point_to_cluster(data[i], min_c);
    }
    for (j=0; j<k; ++j){
        update_cluster_center(clusters[j]);
    }
}

/*internal use*/
void print_cluster(Cluster* c){
    int i;
    printf("\nCluster Dimension: %d\n ", c->dim);
    printf("\nPrinting Cluster Center: \n ");
    print_point(c->center, c->dim);
    printf("\nCurrent Index: %d\n ", c->idx);
    printf("\nPrinting Cluster Points: \n ");
    for (i=0; i< c->idx; ++i){
        printf("Point #%d: ", i);
        print_point(c->points[i], c->dim);
        printf("\n");

    }
}

void print_point(double* point, int dim){
    int i;
    for (i=0; i<dim; i++){
        printf("%f,", point[i]);
    }
}

double** read_from_file(int n, int d){
    int point_counter = 0;
    int i = 0;
    char c;
    double **point_array=(double **)malloc(sizeof(double *)*n);
    point_array[point_counter] = (double*)calloc(d, sizeof(double));
    while (scanf("%lf%c", &point_array[point_counter][i], &c) == 2){
        i++;
        if (c == '\n'){
            if (i != d) return NULL;
            i = 0;
            point_counter++;
            point_array[point_counter] = (double*)calloc(d, sizeof(double));
        }
        if (i >= d || point_counter > n){
            return NULL;
        }
    }
    if (c != '\n') point_counter++;  /* if we reach EOF, we wont increase pointer_counter in the loop
                                        so we do it now for the next condition */
    if (n > point_counter) return NULL;
    return point_array;
}



/*  -----OMRI's CODE-----   */


Cluster ** firstKClusters(int d, int K, int N, double **data){
    int i;
    Cluster **clusters=(Cluster **)malloc(sizeof(Cluster*)*K);
    for (i = 0; i<K; i++){
        clusters[i] = create_cluster(data[i], d, N);
    }
    return clusters;
}
void remember_centers(Cluster** clusters, int K, int d, double ** prev_centers){
    int i,j;
    for(i=0;i<K;i++){
        for(j=0;j<d;j++){
            prev_centers[i][j] = clusters[i]->center[j];
        }
    }
}
int cluster_equal(Cluster ** clusters, double ** prev_centers, int K, int d) {
    double temp =0;
    int i,j;
    for (i = 0; i < K; i++) {
        for (j = 0; j < d; j++) {
            temp = prev_centers[i][j] - clusters[i]->center[j];
            if (temp > 0.001 || temp < -0.001)
                return 0;
        }
    }
    return 1;
}
void print_clusters_centers(Cluster ** clusters, int K, int d){
    double val, rounded_down;
    int i,j;
    for (i = 0; i <K; i++) {
        for (j = 0; j < d; j++) {
            val = clusters[i]->center[j];
            rounded_down = (double)((int)(val * 100)) / 100;
            printf("%0.2f",rounded_down);
            if(j<d -1)
                printf("%s",",");
        }
        if(i<K -1)
            printf("%s","\n");
    }
}

