import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from output_supp_methods import clusterReader, point_to_string, RowIdxList, JaccardMeasure
import sys

''' 
    Create data.txt & cluster.txt files and fill them with data
    from the Spectral and K-means algorithm in the right format.
'''
def outputTextFiles(dataList, y, Clusters_Spectral, Clusters_Kmeans, K, N):
    dataFile = open('data.txt', 'w')
    clustersFile = open('clusters.txt', 'w')
    for i in range(N):
        dataFile.write(point_to_string(dataList[i]) + "," + str(y[i]))
        if i < N - 1:
            dataFile.write("\n")
    clustersFile.write(str(K) + "\n")
    clustersFile.write(clusterReader(Clusters_Spectral))
    clustersFile.write(clusterReader(Clusters_Kmeans)[:-1])
    dataFile.close()
    clustersFile.close()


''' 
    Create clusters.pdf
    define matplotlib.pyplot figure and divide it to 3 subplot
    2 upper subplot filled with the 2 algorithms plots
    lower subplot contain summarize text of the generated data and calculation
'''
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


''' 
    input is subplot axes and data
    fill the 2'd axes with colored points according to cluster classification
'''
def visualization2d(ax, dataMatrix, K, lst, alg_name):
    Colors = plt.cm.viridis(np.linspace(0, 1, K))  # different colors array of size k
    for i in range(K):
        if len(lst[i]) != 0:
            newMatrix = dataMatrix[lst[i]].T
            x_axes = newMatrix[0]
            y_axes = newMatrix[1]
            ax.scatter(x_axes, y_axes, c=[(Colors[i][0], Colors[i][1], Colors[i][2])], alpha=1)
        else:
            print("In " + alg_name + " ,cluster number " + str(i) + " is empty")
            sys.exit()
    ax.grid()
    ax.set_title(alg_name, size=16)


''' 
    input is subplot axes and data
    fill the 3'd axes with colored points according to cluster classification
'''
def visualization3d(ax, dataMatrix, K, lst, alg_name):
    Colors = plt.cm.viridis(np.linspace(0, 1, K))  # different colors array of size k
    for i in range(K):
        if len(lst[i]) != 0:
            newMatrix = dataMatrix[lst[i]].T
            x_axes = newMatrix[0]
            y_axes = newMatrix[1]
            z_axes = newMatrix[2]
            ax.scatter(x_axes, y_axes, z_axes, c=[(Colors[i][0], Colors[i][1], Colors[i][2])],
                       alpha=1)
        else:
            print("In " + alg_name + " ,cluster number " + str(i) + " is empty")
            sys.exit()
    ax.dist = 9.5
    ax.set_title("\n" + alg_name, fontsize=16, y=1.1)


''' 
    input is subplot axes and data
    fill the axes with text to present the summarize report in the right format
    and with the generated data and algorithm calculation
'''
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
