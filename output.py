import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_pdf import PdfPages

def outputFile(dataList, y, Clusters_Spectral, Clusters_Kmeans, K, N):
    dataFile = open('data.txt', 'w')
    clustersFile = open('clusters.txt', 'w')
    for i in range(N):
        dataFile.write(point_to_string(dataList[i]) + "," + str(y[i])+"\n")
    clustersFile.write(str(K)+"\n")
    clustersFile.write(clusterReader(Clusters_Spectral))
    clustersFile.write(clusterReader(Clusters_Kmeans))
    dataFile.close()

def graphic(dataMatrix, y, N, K, d, Clusters_Spectral, Clusters_Kmeans):
    lst_Spectral = RowIdxList(K, Clusters_Spectral)
    lst_Kmeans = RowIdxList(K, Clusters_Kmeans)
    if d == 2:
        fig1 = visualization2d(dataMatrix, K, lst_Spectral, "Normalized Spectral Clustering")
        fig2 = visualization2d(dataMatrix, K, lst_Kmeans, "K-means")
    else:
        fig1 = visualization3d(dataMatrix, K, lst_Spectral, "Normalized Spectral Clustering")
        fig2 = visualization3d(dataMatrix, K, lst_Kmeans, "K-means")

    # calculating the Jaccard-Measure
    jm_spectral = JaccardMeasure(y, lst_Spectral, N, K)
    jm_kmeans = JaccardMeasure(y, lst_Kmeans, N, K)

    # creating the PDF
    fig3 = summarize(N, K, jm_spectral, jm_kmeans)
    pp = PdfPages('clusters.pdf')
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.close()



def visualization2d(dataMatrix,K,lst,str):
    fig = plt.figure()
    for i in range(K):
        newMatrix = dataMatrix[lst[i]].T
        x_axes = newMatrix[0]
        y_axes = newMatrix[1]
        plt.scatter(x_axes, y_axes, c=[(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))])
    plt.grid()
    plt.title(str)
    #plt.show()
    return fig

def visualization3d(dataMatrix,K,lst,str):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # print(K) # TODO: for some K and N there could be a cluster without points which in turn will lead to IndexError in the loop over i
    # print(lst)
    for i in range(K):
        newMatrix = dataMatrix[lst[i]].T
        x_axes = newMatrix[0]
        y_axes = newMatrix[1]
        z_axes = newMatrix[2]
        ax.scatter(x_axes, y_axes, z_axes, c=[(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))])
    plt.title(str)
    #plt.show()
    return fig

def clusterReader(Clusters_List):
    s = ""
    for i in Clusters_List:
        if i != -1:
            s = s + str(i) + ","
        else:
            s = s[:-1] + "\n"
    return s

def point_to_string(lst):
    str_point = ""
    for t in lst:
        str_point += str(np.float64(t)) + ","
    return str_point[:-1]

def RowIdxList(K,Clusters_List):
    lst = []
    for i in range(K):
        idx = Clusters_List.index(-1)
        rowidx = np.array(Clusters_List[:idx])
        Clusters_List = Clusters_List[idx+1:]
        lst.append(rowidx)
    return lst

def summarize(N, K, jm_spectral, jm_kmeans):
    fig = plt.figure()
    plt.figtext(0.5, 0.7, "Data was generated from the values:\n"
                "n = " + str(N) + " , " + "k = " + str(K) + "\n"
                "The k that was used for both algorithms was " + str(K) + "\n"
                "The Jaccard measure for Spectral Clustring: " + str(jm_spectral)[0:6] + "\n"
                "The Jaccard measure for K-means: " + str(jm_kmeans)[0:6],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
                )
    return fig

def JaccardMeasure(y,lst,N,K):
    sum_both = 0
    sum_std=0
    sum_alg = 0

    for i in range(K):
        arr_alg = np.zeros(shape=(N))
        for j in lst[i]:
            arr_alg[j] = 1
        arr_std = np.array(y[:]==i)*1.
        arr_both = arr_alg * arr_std
        sum_alg += np.sum(arr_alg)
        sum_std += np.sum(arr_std)
        sum_both += np.sum(arr_both)
    comb_alg = nC2(sum_alg)
    comb_std = nC2(sum_std)
    comb_both = nC2(sum_both)
    return comb_both / (comb_alg + comb_std - comb_both) # subtract comb_both beacause pairs in comb_both are counted twice

def nC2(n):
    return int(n * (n-1) / 2)



