import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


def outputFile(dataList, y, Clusters_Spectral, Clusters_Kmeans, K, N):
    dataFile = open('data.txt', 'w')
    clustersFile = open('clusters.txt', 'w')
    for i in range(N):
        dataFile.write(point_to_string(dataList[i]) + "," + str(y[i]) + "\n")
    clustersFile.write(str(K) + "\n")
    clustersFile.write(clusterReader(Clusters_Spectral))
    clustersFile.write(clusterReader(Clusters_Kmeans))
    dataFile.close()


def graphic(dataMatrix, y, N, K, new_K, d, Clusters_Spectral, Clusters_Kmeans):
    # Set up a figure and grid partition
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 2, figure=fig)

    lst_Spectral = RowIdxList(new_K, Clusters_Spectral)
    lst_Kmeans = RowIdxList(new_K, Clusters_Kmeans)
    if d == 2:
        # add subplots
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 0])

        visualization2d(ax1, dataMatrix, new_K, lst_Spectral, "Normalized Spectral Clustering")
        visualization2d(ax2, dataMatrix, new_K, lst_Kmeans, "K-means")
    else:
        # add subplots
        ax1 = fig.add_subplot(gs[0, 1], projection='3d')
        ax2 = fig.add_subplot(gs[0, 0], projection='3d')

        visualization3d(ax1, dataMatrix, new_K, lst_Spectral, "Normalized Spectral Clustering")
        visualization3d(ax2, dataMatrix, new_K, lst_Kmeans, "K-means")

    # # calculating the Jaccard-Measure
    jm_spectral = JaccardMeasure(y, lst_Spectral, N, new_K)
    jm_kmeans = JaccardMeasure(y, lst_Kmeans, N, new_K)

    # add last subplot for text
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_axis_off()  # make plot invisible
    summarize(ax3, N, K, new_K, d, jm_spectral, jm_kmeans)
    fig.tight_layout()

    # creating the PDF
    pp = PdfPages('clusters.pdf')
    pp.savefig(fig)
    pp.close()
    # plt.show()


def visualization2d(ax, dataMatrix, K, lst, alg_name):
    Colors = plt.cm.viridis(np.linspace(0, 1, K))
    for i in range(K):
        if lst[i].size != 0:
            newMatrix = dataMatrix[lst[i]].T
            x_axes = newMatrix[0]
            y_axes = newMatrix[1]
            ax.scatter(x_axes, y_axes, c=[(Colors[i][0], Colors[i][1], Colors[i][2])], label="cluster " + str(i),
                       alpha=1)
        else:
            print("In " + alg_name + " ,cluster number " + str(i) + " is empty")
    ax.grid()
    ax.set_title(alg_name, size=16)


def visualization3d(ax, dataMatrix, K, lst, alg_name):
    Colors = plt.cm.viridis(np.linspace(0, 1, K))  # TODO: check if works on nova
    for i in range(K):
        if lst[i].size != 0:
            newMatrix = dataMatrix[lst[i]].T
            x_axes = newMatrix[0]
            y_axes = newMatrix[1]
            z_axes = newMatrix[2]
            ax.scatter(x_axes, y_axes, z_axes, c=[(Colors[i][0], Colors[i][1], Colors[i][2])],
                       label="cluster " + str(i), alpha=1)
        else:
            print("In " + alg_name + " ,cluster number " + str(i) + " is empty")
    ax.dist = 9.5
    ax.set_title("\n" + alg_name, fontsize=16, y=1.1)

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

def summarize(ax, N, K, new_K, d, jm_spectral, jm_kmeans):
    # fig = plt.figure()
    report = "Data was generated from the values:\n""n = " + str(N) + " , " + "k = " + str(
        K) + "\n""The k that was used for both algorithms was " + str(
        new_K) + "\n""The Jaccard measure for Spectral Clustring: " + str(jm_spectral)[
                                                                  0:6] + "\n""The Jaccard measure for K-means: " + str(
        jm_kmeans)[0:6]
    if d == 3:
        ax.text(0.5, 0.4, report, horizontalalignment="center", verticalalignment="center", fontsize=24)
    else:
        ax.text(0.5, 0.5, report, horizontalalignment="center", verticalalignment="center", fontsize=24)

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
    mat_sum = np.sum(nC2(mat_both))  # sum all pairs both in standard and algorithm clustering
    # upon return we sum pairs in the standard clustering and the algorithm clustering and subtract the shared pairs
    # of the two in order to avoid counting twice shared pairs.
    return mat_sum / (sum_std + sum_alg - mat_sum)
