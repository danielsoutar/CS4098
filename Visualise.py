import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

for file in glob.glob("./examples/*.p"):
    print(file)
    name = file[11:-2]
    recover = open("./examples/" + name + ".p", "rb")
    input_list = pickle.load(recover)
    print("Loaded ...")

    cancer_cells = []
    T_cells = []
    cyto_T_cells = []

    for i, row in enumerate(input_list):
        try:
            row = [int(x) for x in row]
        except ValueError:
            continue

        if row[4] > 0:
            cancer_cells.append([row[0], row[1], row[2], row[3]])
        if row[5] > 0:
            T_cells.append([row[0], row[1], row[2], row[3]])
        if row[6] > 0:
            cyto_T_cells.append([row[0], row[1], row[2], row[3]])

    cancer_cells = np.asarray(cancer_cells)
    T_cells = np.asarray(T_cells)
    cyto_T_cells = np.asarray(cyto_T_cells)

    print("Separated ...")

    cancer_cells_xAvg = np.mean(np.array([cancer_cells[:, 0], cancer_cells[:, 1]]), axis=0)
    cancer_cells_yAvg = np.mean(np.array([cancer_cells[:, 2], cancer_cells[:, 3]]), axis=0)

    T_cells_xAvg = np.mean(np.array([T_cells[:, 0], T_cells[:, 1]]), axis=0)
    T_cells_yAvg = np.mean(np.array([T_cells[:, 2], T_cells[:, 3]]), axis=0)

    cyto_T_cells_xAvg = np.mean(np.array([cyto_T_cells[:, 0], cyto_T_cells[:, 1]]), axis=0)
    cyto_T_cells_yAvg = np.mean(np.array([cyto_T_cells[:, 2], cyto_T_cells[:, 3]]), axis=0)

    fig, (ax1) = plt.subplots(1, 1, figsize=(40, 30))

    ax1.set_title("Cancer cells", fontsize=30)
    ax1.scatter(cancer_cells_xAvg, cancer_cells_yAvg, s=0.1, c="b")

    # ax2.set_title("T cells", fontsize=30)
    # ax2.scatter(T_cells_xAvg, T_cells_yAvg, s=0.1, c="r")

    # ax3.set_title("Cytotoxic T cells", fontsize=30)
    # ax3.scatter(cyto_T_cells_xAvg, cyto_T_cells_yAvg, s=0.1, c="g")

    fig.savefig("./visual/" + name + ".png", bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print("Visual saved ...")

    # t = 20
    # partitioned_cancer_cells, windows = partition.partition(cancer_cells)
    # print("Cancer cells partitioned ...")
    # result = window_cleaning_algorithm(partitioned_cancer_cells, t, windows)
    # print("Result retrieved ...")

    # tile_numbers = []
    # dups = []
    # histogram = np.zeros(21, dtype=np.uint32)

    # for i in range(t):
    #     for j in range(t):
    #         num = 0
    #         for key, value in result[i][j].items():
    #             if value not in dups:
    #                 dups.append(value)
    #             num += 1
    #         tile_numbers.append(num)

    # total_cluster_cells = 0

    # clusters_sum = 0
    # dups_length = len(dups)

    # for i in dups:
    #     value = len(i.cells)
    #     clusters_sum += value
    #     total_cluster_cells += len(i.cells)
    #     if value > 20:
    #         histogram[20] += 1
    #     else:
    #         histogram[value - 1] += 1

    # print("Histogram retrieved ...")

    # clusters_avg = clusters_sum / dups_length

    # assert(total_cluster_cells == len(cancer_cells))

    # y = np.array(histogram)
    # x = np.arange(21) + 1

    # plt.bar(x, y)
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.savefig("./inputs/" + name + ".png", bbox_inches='tight')
    # plt.show()
    # plt.close()

    # with open("./inputs/" + name + ".txt", "w", newline="") as dest:
    #     dest.write("Number of clusters: " + str(len(dups)) + "\n")
    #     dest.write("Total number of cells: " + str(total_cluster_cells) + "\n")
    #     dest.write("Cluster counts: " + "\n")
    #     for i, x in enumerate(histogram):
    #         dest.write(str(i) + ", " + str(x) + "\n")















# End of file
