import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


def visualise_mixed_cells(cancer_cells=None, T_cells=None, cyto_T_cells=None, single_image=False):
    if not single_image:
        cancer_cells_xAvg = np.mean(np.array([cancer_cells[:, 0], cancer_cells[:, 1]]), axis=0)
        cancer_cells_yAvg = np.mean(np.array([cancer_cells[:, 2], cancer_cells[:, 3]]), axis=0)

        T_cells_xAvg = np.mean(np.array([T_cells[:, 0], T_cells[:, 1]]), axis=0)
        T_cells_yAvg = np.mean(np.array([T_cells[:, 2], T_cells[:, 3]]), axis=0)

        cyto_T_cells_xAvg = np.mean(np.array([cyto_T_cells[:, 0], cyto_T_cells[:, 1]]), axis=0)
        cyto_T_cells_yAvg = np.mean(np.array([cyto_T_cells[:, 2], cyto_T_cells[:, 3]]), axis=0)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        ax1.set_title('Cancer cells', fontsize=30)
        ax1.scatter(cancer_cells_xAvg, cancer_cells_yAvg, s=0.1, c='g')

        ax2.set_title('T cells', fontsize=30)
        ax2.scatter(T_cells_xAvg, T_cells_yAvg, s=0.1, c='r')

        ax3.set_title('Cytotoxic T cells', fontsize=30)
        ax3.scatter(cyto_T_cells_xAvg, cyto_T_cells_yAvg, s=0.1, c='b')

        plt.show(fig)
    else:
        cancer_cells_xAvg = np.mean(np.array([cancer_cells[:, 0], cancer_cells[:, 1]]), axis=0)
        cancer_cells_yAvg = np.mean(np.array([cancer_cells[:, 2], cancer_cells[:, 3]]), axis=0)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

        ax1.set_title('Cancer cells', fontsize=30)
        ax1.scatter(cancer_cells_xAvg, cancer_cells_yAvg, s=0.1, c='b')

        plt.show(fig)


def visualise_clusters(cancer_clusters, size_included=True, size=(10, 10), rotated=False):
    if size_included:
        i, j, k, l = 1, 2, 3, 4
    else:
        i, j, k, l = 0, 1, 2, 3
    if rotated:
        cancer_clusters_xAvg = np.mean(np.array([cancer_clusters[i, :], cancer_clusters[j, :]]), axis=0)
        cancer_clusters_yAvg = np.mean(np.array([cancer_clusters[k, :], cancer_clusters[l, :]]), axis=0)
    else:
        cancer_clusters_xAvg = np.mean(np.array([cancer_clusters[:, i], cancer_clusters[:, j]]), axis=0)
        cancer_clusters_yAvg = np.mean(np.array([cancer_clusters[:, k], cancer_clusters[:, l]]), axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=size)

    ax1.set_title('Cancer clusters', fontsize=20)
    ax1.scatter(cancer_clusters_xAvg, cancer_clusters_yAvg, s=5, c='b')

    if rotated:
        plt.gca().invert_yaxis()

    plt.show(fig)


def visualise_lymphocytes(cd3_cells, cd8_cells):
    cd3_cells_xAvg = np.mean(np.array([cd3_cells[:, 0], cd3_cells[:, 1]]), axis=0)
    cd3_cells_yAvg = np.mean(np.array([cd3_cells[:, 2], cd3_cells[:, 3]]), axis=0)

    cd8_cells_xAvg = np.mean(np.array([cd8_cells[:, 0], cd8_cells[:, 1]]), axis=0)
    cd8_cells_yAvg = np.mean(np.array([cd8_cells[:, 2], cd8_cells[:, 3]]), axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

    ax1.set_title('CD3 cells', fontsize=30)
    ax1.scatter(cd3_cells_xAvg, cd3_cells_yAvg, s=0.1, c='r')

    ax2.set_title('CD8 cells', fontsize=30)
    ax2.scatter(cd8_cells_xAvg, cd8_cells_yAvg, s=0.1, c='g')

    plt.show(fig)


def visualise_heatmaps(heatmap, title):
    with sb.axes_style("white"):
        sb.heatmap(heatmap, cmap="hot", mask=(heatmap == -1))
        plt.title(title)
        plt.show()





















# End of file
