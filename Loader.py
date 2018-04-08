import Cluster
import csv
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import partition
import pickle
import seaborn as sb
from tqdm import tqdm
import Visualiser

positive_files = ["./LargerDataset/2017_08_04__3301.czi_3035_job3033.object_results.p",
                  "./LargerDataset/2016_12_17__0733(ext).czi_3110_job2903.object_results.p",
                  "./LargerDataset/2016_12_29__0805.czi_3033_job2856.object_results.p",
                  "./LargerDataset/2016_12_17__0703(ext).czi_3082_job2873.object_results.p",
                  "./LargerDataset/2017_08_04__3305.czi_3039_job3037.object_results.p",
                  "./LargerDataset/2016_12_17__0690.czi_3064_job2861.object_results.p",
                  "./LargerDataset/2017_08_04__3303.czi_3037_job3035.object_results.p",
                  "./LargerDataset/2016_12_17__0747(ext).czi_3124_job2918.object_results.p",
                  "./LargerDataset/2016_12_17__0781(ext).czi_3149_job2942.object_results.p",
                  "./LargerDataset/2017_08_04__3302.czi_3036_job3034.object_results.p",
                  "./LargerDataset/2016_12_28__0793(ext).czi_3161_job2956.object_results.p",
                  "./LargerDataset/2017_08_04__3304.czi_3038_job3036.object_results.p",
                  "./LargerDataset/2016_12_17__0769(ext).czi_3145_job2938.object_results.p",
                  "./LargerDataset/2016_12_17__0711(ext).czi_3089_job2882.object_results.p",
                  "./LargerDataset/2016_12_17__0736(ext).czi_3113_job2906.object_results.p"
                  ]

positive_codes = ["0736", "0711", "0769", "3304", "0793",
                  "3302", "0781", "0747", "3303", "0690",
                  "3305", "0703", "0805", "0733", "3301"]


# Generic function for loading from pickle files
def load_data(src, path, pretty_print=True):
    example = open(path + src + ".p", 'rb')
    input_list = pickle.load(example)
    example.close()
    if pretty_print:
        print("Size of input list: ", len(input_list))
    return input_list


# Annoyingly have a different representation (csv) for these files, but space constraints require it.
def load_lymphocytes(src, path, pretty_print=True):
    reader = csv.reader(open(path + src + ".csv", "r"), delimiter=",")
    input_list = list(reader)
    if pretty_print:
        print("Size of input list: ", len(input_list))
    return input_list


# This function might need to filter cells within a range - let's do that at the beginning by passing in
# filters.
def split_mixed_cells(input_list, fxMin=None, fxMax=None, fyMin=None, fyMax=None, fcell_size=None, pretty_print=True):
    cancer_cells = []
    T_cells = []
    cyto_T_cells = []

    # Order is xMin, xMax, yMin, yMax, have to do cell_size separately because there are other fields.
    funcs = [fxMin, fxMax, fyMin, fyMax]

    for i, row in enumerate(input_list):
        conv_row = [int(x) for x in row[0:-1]]
        conv_row.append(float(row[-1]))

        include_cell = True

        if conv_row[4] > 0:
            for i, f in enumerate(funcs):
                if f is None:
                    continue
                else:
                    if not f(conv_row[i]):
                        include_cell = False
                        break
            if include_cell:
                if fcell_size is not None:
                    if fcell_size(conv_row[-1]):
                        cancer_cells.append([conv_row[0], conv_row[1], conv_row[2], conv_row[3]])
                else:
                    cancer_cells.append([conv_row[0], conv_row[1], conv_row[2], conv_row[3]])

        if conv_row[5] > 0:
            T_cells.append([conv_row[0], conv_row[1], conv_row[2], conv_row[3]])
        if conv_row[6] > 0:
            cyto_T_cells.append([conv_row[0], conv_row[1], conv_row[2], conv_row[3]])

    if pretty_print:
        print("Number of cancer cells:", len(cancer_cells))
    cancer_cells = np.asarray(cancer_cells)
    T_cells = np.asarray(T_cells)
    cyto_T_cells = np.asarray(cyto_T_cells)
    return cancer_cells, T_cells, cyto_T_cells


# As above with split_original, but uses num_cells instead of cell sizes
def split_clusters(input_list, fxMin=None, fxMax=None, fyMin=None, fyMax=None, fnum_cells=None, pretty_print=True):
    cancer_clusters = []
    # Order of items in each entry is num_cells, xMax, xMin, yMax, yMin, so need to reorder our filters
    funcs = [fnum_cells, fxMax, fxMin, fyMax, fyMin]

    for i, row in enumerate(input_list):
        row = [int(x) for x in row]
        include_cluster = True

        for i, f in enumerate(funcs):
            if f is None:
                continue
            else:
                if not f(row[i]):
                    include_cluster = False
                    break
        if include_cluster:
            cancer_clusters.append([row[0], row[2], row[1], row[4], row[3]])

    if pretty_print:
        print("Number of cancer clusters:", len(cancer_clusters))
    cancer_clusters = np.asarray(cancer_clusters)

    return cancer_clusters


# As above with split_new, but collects lymphocytes, not cancer clusters.
# Note that we include the binary vectors indicating the type of cell - this is so I can more easily
# combine them and split them later.
def split_lymphocytes(input_list, fxMin=None, fxMax=None, fyMin=None, fyMax=None, pretty_print=True):
    cd3_cells = []
    cd8_cells = []
    # These are the easiest - no other fields, just filter these values.
    funcs = [fxMin, fxMax, fyMin, fyMax]

    for i, row in enumerate(input_list):
        row = [int(x) for x in row]
        include_cell = True

        for i, f in enumerate(funcs):
            if f is None:
                continue
            else:
                if not f(row[i]):
                    include_cell = False
                    break
        if include_cell:
            if row[-2] > 0:
                cd3_cells.append([row[0], row[1], row[2], row[3], row[4], row[5]])
            elif row[-1] > 0:
                cd8_cells.append([row[0], row[1], row[2], row[3], row[4], row[5]])

    if pretty_print:
        print("Number of CD3 cells:", len(cd3_cells))
        print("Number of CD8 cells:", len(cd8_cells))
    cd3_cells = np.asarray(cd3_cells)
    cd8_cells = np.asarray(cd8_cells)

    return cd3_cells, cd8_cells


def load_split(src, path, fxMin=None, fxMax=None,
               fyMin=None, fyMax=None,
               fcell_size=lambda x: x >= 50.0,
               fnum_cells=lambda x: x > 0,
               input_type="mixed", pretty_print=True):
    if pretty_print:
        print(src)
    if input_type.lower() == "mixed":
        input_list = load_data(src, path, pretty_print=pretty_print)
        cancer_cells, T_cells, cyto_T_cells = split_mixed_cells(input_list, fxMin=fxMin,
                                                                fxMax=fxMax, fyMin=fyMin,
                                                                fyMax=fyMax, fcell_size=fcell_size,
                                                                pretty_print=pretty_print)
        return cancer_cells, T_cells, cyto_T_cells
    elif input_type.lower() == "cluster":
        input_list = load_data(src, path, pretty_print=pretty_print)
        cancer_clusters = split_clusters(input_list, fxMin=fxMin, fxMax=fxMax,
                                         fyMin=fyMin, fyMax=fyMax, fnum_cells=fnum_cells,
                                         pretty_print=pretty_print)
        return cancer_clusters
    elif input_type.lower() == "lymphocyte":
        input_list = load_lymphocytes(src, path, pretty_print=pretty_print)
        cd3_cells, cd8_cells = split_lymphocytes(input_list, fxMin=fxMin, fxMax=fxMax,
                                                 fyMin=fyMin, fyMax=fyMax, pretty_print=pretty_print)
        return cd3_cells, cd8_cells


# Ranges of the form 1-10, 1-20, 1-30, etc.
def load_cluster_size_histograms_extending_ranges(m=115, upper=21, step=5, scaled=True, display_plot=False, pretty_print=False):
    positives = []
    X = np.empty([m, int(math.ceil(upper / step))-1], dtype=np.float32)
    y = np.zeros([m], dtype=int)
    codes = []

    size_range = [x for x in range(step, upper, step)]

    for i, name in enumerate(sorted(glob.glob("./LargerDataset/*.p"))):
        if display_plot:
            print(name)

        code = name.split('/')[-1][0:-2]

        if "SN" in code:
            code = code[0:4]
        else:
            code = code.split("__")[1][0:4]

        codes.append(code)

        with open(name, "rb") as file:
            input_list = pickle.load(file)

            if name in positive_files:
                positives.append(i)
                y[i] = 1

            arr = np.zeros(upper, dtype=np.float32)
            for row in input_list:
                num_cells = int(row[0]) - 1
                if num_cells >= 0:
                    if num_cells < upper:
                        arr[num_cells] += 1
                    else:
                        arr[upper - 1] += 1

            new_arr = np.zeros(X.shape[1], dtype=np.float32)
            for j in range(1, X.shape[1]+1):
                new_arr[j-1] = sum(arr[0:(j * step)])
            arr = new_arr

            my_sum = sum(arr)
            if pretty_print:
                print(arr)

            if display_plot:
                ybar = np.array(arr)
                xbar = ["1-" + str(x) for x in size_range]

                # fig = plt.figure(figsize=(10, 3))
                plt.bar(xbar, ybar, align='center', width=0.8)
                plt.xlabel("Sizes of clusters")
                plt.ylabel("Frequency")
                plt.show()

            if scaled:
                arr = arr / my_sum
            X[i] = arr
        if pretty_print:
            print()

    features = ["1-" + str(x) for x in range(step, upper, step)]

    return X, y, features, codes


# Ranges of the form 1-10, 11-20, 21-30, etc.
def load_cluster_size_histograms_separate_ranges(m=115, upper=21, step=5, scaled=True, display_plot=False, pretty_print=False):
    positives = []
    X = np.empty([m, int(math.ceil(upper / step))-1], dtype=np.float32)
    y = np.zeros([m], dtype=int)
    codes = []

    size_range = [x for x in range(step, upper, step)]

    for i, name in enumerate(sorted(glob.glob("./LargerDataset/*.p"))):
        if display_plot:
            print(name)

        code = name.split('/')[-1][0:-2]

        if "SN" in code:
            code = code[0:4]
        else:
            code = code.split("__")[1][0:4]

        codes.append(code)

        with open(name, "rb") as file:
            input_list = pickle.load(file)

            if name in positive_files:
                positives.append(i)
                y[i] = 1

            arr = np.zeros(upper, dtype=np.float32)
            for row in input_list:
                num_cells = int(row[0]) - 1
                if num_cells >= 0:
                    if num_cells < upper:
                        arr[num_cells] += 1
                    else:
                        arr[upper - 1] += 1

            if size_range is not None:
                new_arr = np.zeros(X.shape[1], dtype=np.float32)
                new_arr[0] = sum(arr[0:step])
                for j in range(1, X.shape[1]):
                    new_arr[j] = sum(arr[(j * step):((j+1) * step)])
                arr = new_arr

            my_sum = sum(arr)
            if pretty_print:
                print(arr)

            if display_plot:
                ybar = np.array(arr)
                xbar = [str(x-step+1) + "-" + str(x) for x in size_range]

                # fig = plt.figure(figsize=(10, 3))
                plt.bar(xbar, ybar, align='center', width=0.8)
                plt.xlabel("Sizes of clusters")
                plt.ylabel("Frequency")
                plt.show()

            if scaled:
                arr = arr / my_sum
            X[i] = arr
        if pretty_print:
            print()

    features = [str(x-step+1) + "-" + str(x) for x in range(step, upper, step)]

    return X, y, features, codes


# Individual cluster sizes.
def load_cluster_size_histograms_singular(m=115, upper=21, scaled=True, display_plot=False, pretty_print=False):
    positives = []
    X = np.empty([m, upper], dtype=np.float32)
    y = np.zeros([m], dtype=int)
    codes = []

    for i, name in enumerate(sorted(glob.glob("./LargerDataset/*.p"))):
        if display_plot:
            print(name)

        code = name.split('/')[-1][0:-2]

        if "SN" in code:
            code = code[0:4]
        else:
            code = code.split("__")[1][0:4]

        codes.append(code)

        with open(name, "rb") as file:
            input_list = pickle.load(file)

            if name in positive_files:
                positives.append(i)
                y[i] = 1

            arr = np.zeros(X.shape[1], dtype=np.float32)
            for row in input_list:
                num_cells = int(row[0]) - 1
                if num_cells >= 0:
                    if num_cells < X.shape[1]:
                        arr[num_cells] += 1
                    else:
                        arr[X.shape[1] - 1] += 1

            my_sum = sum(arr)
            if pretty_print:
                print("Number of clusters:", my_sum)
                print(arr)

            if display_plot:
                ybar = np.array(arr)
                xbar = [str(x+1) for x in range(X.shape[1])]
                xbar[-1] = xbar[-1] + '+'

                # fig = plt.figure(figsize=(10, 3))
                plt.bar(xbar, ybar, align='center', width=0.8)
                plt.xlabel("Sizes of clusters")
                plt.ylabel("Frequency")
                plt.show()

            if scaled:
                arr = arr / my_sum
            X[i] = arr
        if pretty_print:
            print()

    features = ["size " + str(i+1) for i in range(X.shape[1])]
    features[-1] = features[-1] + '+'

    return X, y, features, codes


def get_heatmaps(m=115, d=50, t=886, cluster_size_range=[1, 2, 3, 4], display_plot=False, metric_tiling=True,
                 pretty_print=False, take_lymphocyte_ratio=True, take_cd3_ratio=False, take_cd8_ratio=False):
    if take_lymphocyte_ratio:
        ratio_type = "lymphocytes"
    elif take_cd3_ratio:
        ratio_type = "cd3"
    elif take_cd8_ratio:
        ratio_type = "cd8"
    else:
        raise Exception("Error: no ratio type specified.")

    print("ratio type set...")

    if os.path.isfile("./Ratio_Heatmaps/ratio_heatmaps_" + str(t) + "_" + str(d) + "_" + str(cluster_size_range) + "_" + ratio_type + ".p"):
        dataset = pickle.load(open("./Ratio_Heatmaps/ratio_heatmaps_" + str(t) + "_" + str(d) + "_" + str(cluster_size_range) + "_" + ratio_type + ".p", "rb"))
        return dataset["X"], dataset["y"], dataset["t"], dataset["d"], dataset["range"]

    print("file not found - extracting objects...")

    X = np.empty((m), dtype=object)
    y = np.zeros([m, 1], dtype=int)
    positives = []

    Cluster.set_d(d)

    # Doing these file-pairs one at a time - they take up an insane amount of space!
    # We have a HUGE number of these lymphocytes.
    for (i, lymph_name), (j, cluster_name) in tqdm(zip(enumerate(sorted(glob.glob("./lymphocytes/*.csv"))),
                                                   enumerate(sorted(glob.glob("./LargerDataset/*.p"))))):
        lymph_code = lymph_name.split('/')[-1][0:-4].split("__")
        cluster_code = cluster_name.split('/')[-1][0:-2].split("__")

        # No SN-- files are positive in our dataset - ignore these for potential positives
        if len(lymph_code) == len(cluster_code) and len(lymph_code) > 1:
            lymph_code = lymph_code[1][0:4]
            cluster_code = cluster_code[1][0:4]
            if lymph_code in positive_codes and lymph_code == cluster_code:
                positives.append(i)
                y[i] = 1

        print("Code:", lymph_code)

        cd3, cd8 = load_split(lymph_name.split('/')[-1][0:-4], "./lymphocytes/", input_type="lymphocyte", pretty_print=pretty_print)

        # Add the cluster class column
        cd3 = np.concatenate((cd3, np.zeros((cd3.shape[0], 1))), axis=1)
        cd8 = np.concatenate((cd8, np.zeros((cd8.shape[0], 1))), axis=1)

        # Get the clusters
        cancer_clusters = load_split(cluster_name.split('/')[-1][0:-2], "./LargerDataset/", input_type="cluster", pretty_print=pretty_print)

        # Add the cd3/cd8 class columns, filter for those with sizes in specified range (here it's 1-4, tumour buds)
        cancer_clusters = np.concatenate((cancer_clusters,
                                          np.zeros((cancer_clusters.shape[0], 2)),
                                          np.ones((cancer_clusters.shape[0], 1))),
                                         axis=1)

        cancer_clusters = cancer_clusters[cancer_clusters[:, 0] < cluster_size_range[1]+1]
        cancer_clusters = cancer_clusters[cancer_clusters[:, 0] > cluster_size_range[0]-1]
        cancer_clusters = cancer_clusters[:, 1:]

        # Combine all of them together
        items = np.concatenate((cancer_clusters, cd3, cd8))

        partitioned_items, tiles, lymph_w, lymph_h, \
            clust_w, clust_h, x_step, y_step = partition.partition(items, tile_size=t, to_list=True, input_type="mixed", by_metric=metric_tiling, scale=1)

        ratios = Cluster.get_lymphocyte_cluster_ratio_heatmap(partitioned_items, partitioned_items.shape, tiles,
                                                              lymph_w, lymph_h, clust_w, clust_h, x_step, y_step,
                                                              take_lymphocyte_ratio=take_lymphocyte_ratio,
                                                              take_cd3_ratio=take_cd3_ratio,
                                                              take_cd8_ratio=take_cd8_ratio)

        cancer_clusters = np.flip(np.rot90(cancer_clusters), 0)
        ratios = np.flip(np.rot90(ratios), 0)

        if display_plot:
            with sb.axes_style("white"):
                sb.heatmap(ratios, cmap="hot", mask=(ratios == -1))
                Visualiser.visualise_clusters(cancer_clusters, size_included=False, size=(5, 5), rotated=True)
                plt.show()

        X[i] = ratios

        if pretty_print:
            print("CD3 shape:", cd3.shape)
            print("CD8 shape:", cd8.shape)
            print("Clusters.shape:", cancer_clusters.shape)
            print("ratios.shape:", ratios.shape, "\n\n")

    dataset = {"X": X, "y": y, "t": t, "d": d, "range": cluster_size_range}

    pickle.dump(dataset, open("ratio_heatmaps_" + str(t) + "_" + str(d) + "_" + str(cluster_size_range) + "_" + ratio_type + ".p", "wb"))

    return X, y, t, d, cluster_size_range


def load_miscellaneous_data(m=115):
    clinical_data_file = "./Cleaned Medical Data (exclude all NA).csv"

    misc_data = csv.reader(open(clinical_data_file, "r"), delimiter=",")
    data_entries = list(misc_data)

    X = np.empty([m, len(data_entries[0])-1], dtype=int)
    y = np.zeros([m, 1], dtype=int)
    positives = []

    assert(len(data_entries) == m)

    for i, entry in enumerate(data_entries):
        # Format: [row_code, age, sex, pT, site, tum_type, diff]
        if entry[0] in positive_codes:
            positives.append(entry)
            y[i] = 1

        data = entry[1:]
        X[i] = np.asarray(data).astype(int)

    return X, y



















# End of file
