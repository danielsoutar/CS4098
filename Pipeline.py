import glob
import partition
import pickle
import matplotlib.pyplot as plt
import numpy as np
# from Cluster import window_cleaning_algorithm
from Naive_Cluster import fishermans_algorithm

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

    t = 20
    partitioned_cancer_cells, windows = partition.partition(cancer_cells, tile_size=t, to_list=True)
    print("Cancer cells partitioned ...")
    result = fishermans_algorithm(partitioned_cancer_cells, t, windows)
    print("Result retrieved ...")

    dups = set()
    histogram = np.zeros(21, dtype=np.uint32)

    for cluster in result:
        if cluster not in dups:
            dups.add(cluster)

    total_cluster_cells = 0

    clusters_sum = 0
    dups_length = len(dups)

    for i in dups:
        value = len(i.cells)
        clusters_sum += value
        total_cluster_cells += len(i.cells)
        if value > 20:
            histogram[20] += 1
        else:
            histogram[value - 1] += 1

    print("Histogram retrieved ...")

    clusters_avg = clusters_sum / dups_length

    assert(total_cluster_cells == len(cancer_cells))

    y = np.array(histogram)
    x = np.arange(21) + 1

    plt.bar(x, y)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig("./inputs/" + name + ".png", bbox_inches='tight')
    plt.show()
    plt.close()

    with open("./inputs/" + name + ".txt", "w", newline="") as dest:
        dest.write("Number of clusters: " + str(len(dups)) + "\n")
        dest.write("Total number of cells: " + str(total_cluster_cells) + "\n")
        dest.write("Cluster counts: " + "\n")
        for i, x in enumerate(histogram):
            dest.write(str(i) + ", " + str(x) + "\n")









# End of file