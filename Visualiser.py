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

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

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


def get_points_from_objects(objects, size_included=True, rotated=False):
    if size_included:
        i, j, k, l = 1, 2, 3, 4
    else:
        i, j, k, l = 0, 1, 2, 3
    if rotated:
        objects_xAvg = np.mean(np.array([objects[i, :], objects[j, :]]), axis=0)
        objects_yAvg = np.mean(np.array([objects[k, :], objects[l, :]]), axis=0)
    else:
        objects_xAvg = np.mean(np.array([objects[:, i], objects[:, j]]), axis=0)
        objects_yAvg = np.mean(np.array([objects[:, k], objects[:, l]]), axis=0)

    return objects_xAvg, objects_yAvg


def visualise_clusters(cancer_clusters, size_included=True, size=(10, 10), rotated=False, title="Cancer clusters"):
    xAvg, yAvg = get_points_from_objects(cancer_clusters, size_included, rotated)

    fig, ax1 = plt.subplots(1, 1, figsize=size)

    ax1.set_title(title, fontsize=20)
    ax1.scatter(xAvg, yAvg, s=5, c='b')

    if rotated:
        plt.gca().invert_yaxis()

    plt.show(fig)


def visualise_clusters_on_ax(cancer_clusters, ax, size_included=True, size=(10, 10), rotated=False, title="Cancer clusters"):
    xAvg, yAvg = get_points_from_objects(cancer_clusters, size_included, rotated)

    ax.set_title(title, fontsize=20)
    ax.scatter(xAvg, yAvg, s=5, c='b')

    if rotated:
        plt.gca().invert_yaxis()

    return ax


def visualise_lymphocytes_on_ax(lymphocytes, ax, title="CD3&CD8"):
    cd3_cells = lymphocytes[lymphocytes[:, 4] > 0]
    if len(cd3_cells) != 0:
        ax = visualise_one_cd_on_ax(cd3_cells, ax, "CD3", title, 0.1, 'r')

    cd8_cells = lymphocytes[lymphocytes[:, 5] > 0]
    if len(cd8_cells) != 0:
        ax = visualise_one_cd_on_ax(cd8_cells, ax, "CD8", title, 0.1, 'g')

    ax.set_title(title)
    ax.legend()

    return ax


def visualise_one_cd_on_ax(cd, ax, label, title, size, colour):
    cd_cells_xAvg, cd_cells_yAvg = get_points_from_objects(cd, size_included=False, rotated=False)

    ax.set_title(title)
    ax.scatter(cd_cells_xAvg, cd_cells_yAvg, s=size, c=colour, label=label)
    ax.legend()

    return ax


def visualise_clusters_and_lymphocytes_on_ax(mixed_data, ax, title, rotated=False):
    """
    Assumes mixed_data has layout (N, 7), where the format is as follows:

    +------+------+------+------+-----+-----+----+
    | xMin | xMax | yMin | yMax | CD3 | CD8 | TB |
    +------+------+------+------+-----+-----+----+
    | .... | .... | .... | .... | ... | ... | .. |
    +------+------+------+------+-----+-----+----+
    """
    tb_data = mixed_data[mixed_data[:, 6] > 0]
    lym_data = mixed_data[mixed_data[:, 6] == 0]  # assumes all objects are exclusively either tb or lym

    ax = visualise_lymphocytes_on_ax(lym_data, ax)
    ax = visualise_clusters_on_ax(tb_data, ax, size_included=False)
    ax.set_title(title)
    ax.legend()

    return ax


def visualise_heatmaps(heatmap, ax, title, mask):
    with sb.axes_style("white"):
        ax = sb.heatmap(heatmap, cmap="hot", mask=mask)
        ax.set_title(title)
        return ax





















# End of file
