import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_pdf import PdfPages


def outputFile(dataList, y, Clusters_Spectral, Clusters_Kmeans, K, N):
    dataFile = open('data.txt', 'w')
    clustersFile = open('clusters.txt', 'w')
    for i in range(N):
        dataFile.write(point_to_string(dataList[i]) + "," + str(y[i]) + "\n")
    clustersFile.write(str(K) + "\n")
    clustersFile.write(clusterReader(Clusters_Spectral))
    clustersFile.write(clusterReader(Clusters_Kmeans))
    dataFile.close()


def graphic(dataMatrix, y, N, K, d, Clusters_Spectral, Clusters_Kmeans):
    fig, ax = plt.subplots(2, 2)
    lst_Spectral = RowIdxList(K, Clusters_Spectral)
    lst_Kmeans = RowIdxList(K, Clusters_Kmeans)
    if d == 2:
        fig1 = visualization2d(dataMatrix, K, lst_Spectral, "Normalized Spectral Clustering")
        fig2 = visualization2d(dataMatrix, K, lst_Kmeans, "K-means")
    else:
        fig1 = visualization3d(0, dataMatrix, K, lst_Spectral, "Normalized Spectral Clustering")
        fig2 = visualization3d(1, dataMatrix, K, lst_Kmeans, "K-means")

    # calculating the Jaccard-Measure
    jm_spectral = JaccardMeasure(y, lst_Spectral, N, K)
    jm_kmeans = JaccardMeasure(y, lst_Kmeans, N, K)

    # creating the PDF
    fig3 = summarize(N, K, jm_spectral, jm_kmeans)
    plt.show()
    pp = PdfPages('clusters.pdf')
    # pp.savefig(fig1)
    # pp.savefig(fig2)
    # pp.savefig(fig3)
    pp.savefig(fig)
    pp.close()


def visualization2d(dataMatrix, K, lst, alg_name):
    fig = plt.figure()
    Colors = plt.cm.viridis(np.linspace(0, 1, K))
    for i in range(K):
        if lst[i].size != 0:
            newMatrix = dataMatrix[lst[i]].T
            x_axes = newMatrix[0]
            y_axes = newMatrix[1]
            plt.scatter(x_axes, y_axes, c=[(Colors[i][0], Colors[i][1], Colors[i][2])], label="cluster " + str(i),
                        alpha=1)
        else:
            print("In " + alg_name + " ,cluster number " + str(i) + " is empty")
    plt.grid()
    plt.title(alg_name)
    plt.legend()  # TODO: remove legend and label

    # plt.show()
    return fig


def visualization3d(j, dataMatrix, K, lst, alg_name):
    fig = plt.figure()
    ax[0][j] = plt.axes(projection='3d')
    Colors = plt.cm.viridis(np.linspace(0, 1, K))  # TODO: check if works on nova
    for i in range(K):
        if lst[i].size != 0:
            newMatrix = dataMatrix[lst[i]].T
            x_axes = newMatrix[0]
            y_axes = newMatrix[1]
            z_axes = newMatrix[2]
            ax[0][j].scatter(x_axes, y_axes, z_axes, c=[(Colors[i][0], Colors[i][1], Colors[i][2])],
                       label="cluster " + str(i), alpha=1)
        else:
            print("In " + alg_name + " ,cluster number " + str(i) + " is empty")
    plt.title(alg_name)
    plt.legend()  # TODO: remove legend and label
    # plt.show()
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


def RowIdxList(K, Clusters_List):
    lst = []
    for i in range(K):
        idx = Clusters_List.index(-1)
        rowidx = np.array(Clusters_List[:idx])
        Clusters_List = Clusters_List[idx + 1:]
        lst.append(rowidx)
    return lst


def summarize(N, K, jm_spectral, jm_kmeans):
    fig = plt.figure()
    plt.figtext(0.5, 0.7, "Data was generated from the values:\n"
                          "n = " + str(N) + " , " + "k = " + str(K) + "\n"
                                                                      "The k that was used for both algorithms was " + str(
        K) + "\n"
             "The Jaccard measure for Spectral Clustring: " + str(jm_spectral)[0:6] + "\n"
                                                                                      "The Jaccard measure for K-means: " + str(
        jm_kmeans)[0:6],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
                )
    return fig


def JaccardMeasure(y, lst, N, K):
    sum_std = 0
    sum_alg = 0
    nC2 = np.frompyfunc(lambda n: int(n * (n - 1) / 2), 1, 1)

    mat_std = np.zeros(shape=(K, N))
    mat_alg = np.zeros(shape=(K, N))
    for i in range(K):
        arr_std = np.array(y[:] == i) * 1.
        mat_std[i] = arr_std
        for j in lst[i]:
            mat_alg[i][j] = 1
        # sum pairs in each cluster for standard and algorithm clustering separately
        sum_std += nC2(np.sum(arr_std))
        sum_alg += nC2(np.sum(mat_alg[i]))

    mat_both = mat_std @ (mat_alg.T)
    mat_sum = np.sum(nC2(mat_both)) # sum all pairs both in standard and algorithm clustering
    # upon return we sum pairs in the standard clustering and the algorithm clustering and subtract the shared pairs
    # of the two in order to avoid counting twice shared pairs.
    return mat_sum / (sum_std + sum_alg - mat_sum)

    # for i in range(K):
    #     arr_alg = np.zeros(shape=(N))
    #     for j in lst[i]:
    #         arr_alg[j] = 1
    #     arr_std = np.array(y[:]==i)*1.
    #     arr_both = arr_alg * arr_std
    #     if i==0:
    #         print(arr_both)
    #     sum_alg += np.sum(arr_alg)
    #     sum_std += np.sum(arr_std)
    #     sum_both += np.sum(arr_both)
    # comb_alg = nC2(sum_alg)
    # comb_std = nC2(sum_std)
    # comb_both = nC2(sum_both)
    # return comb_both / (comb_alg + comb_std - comb_both) # subtract comb_both because pairs in comb_both are counted twice


def nC2(n):
    return int(n * (n - 1) / 2)
