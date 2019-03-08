# Imports
import Cluster
import extract
import glob
import partition
import pickle
import re
import timeit
import Visualiser
import numpy as np
import math
from pysal.esda.getisord import G_Local
from pysal.weights.Distance import DistanceBand
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sb
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from shapely.geometry import Point, Polygon, LineString
from mpl_toolkits.mplot3d import Axes3D


def get_tags(hierarchy_tags):
    """
    Gets the x-y coordinates of the piecewise line defining the invasive margin.
    """
    tags = []
    x_tags, y_tags = [], []

    for tag in hierarchy_tags:
        if "X=" in tag and "Y=" in tag:
            x = tag.split("\"")[1]
            y = tag.split("\"")[3]
            x_tags.append(int(x))
            y_tags.append(int(y))

    for i, j in zip(x_tags, y_tags):
        tags.append((int(i), int(j)))

    return tags


def load_linestring(polygon_file):
    """
    Loads the x-y coordinates and creates a LineString object from them.
    This is useful for establishing the distance of cells from the
    invasive margin.
    """
    poly = None
    with open(polygon_file, "r") as f:
        hierarchy = f.read()

        p = re.compile("<*")

        hierarchy_tags = p.split(hierarchy)

        tags = get_tags(hierarchy_tags)

        poly = LineString(tags)

    if poly is None:
        raise Exception
    else:
        return poly


def load_shape(polygon_file):
    """
    Loads the x-y coordinates and creates a Polygon object from them.
    This is useful for establishing whether a given object is within
    the invasive margin or not.
    """
    poly = None

    with open(polygon_file, "r") as f:
        hierarchy = f.read()

        p = re.compile("<*")

        hierarchy_tags = p.split(hierarchy)

        tags = get_tags(hierarchy_tags)

        poly = Polygon(tags)

    if poly is None:
        raise Exception
    else:
        return poly


def plot_line_and_shape(ax, linestring, shape):
    """
    Given a matplotlib axes object, plots the Shapely LineString and Polygon
    objects onto the axes. Returns the axes.
    """
    patch = PolygonPatch(shape, fc="red", alpha=0.5, zorder=2)
    ax.add_patch(patch)

    xs = [x for (x, y) in linestring.coords]
    ys = [y for (x, y) in linestring.coords]

    ax.plot(xs, ys)

    return ax


def get_margin_cells(all_cells, linestring, my_distance):
    """
    Gets all cells that are within my_distance of the invasive margin.
    This includes TB, CD3, and CD8.
    """
    margin_cell_list = []

    assert (type(all_cells) == np.ndarray and all_cells.shape[1] >= 4), "type of all_cells is {}, all_cells.shape is {}".format(type(all_cells), all_cells.shape)

    x_avgs = (all_cells[:, 0] + all_cells[:, 1]) / 2
    y_avgs = (all_cells[:, 2] + all_cells[:, 3]) / 2
    xy_avgs = np.array((x_avgs, y_avgs)).transpose()

    for i in range(len(all_cells)):
        p = Point(xy_avgs[i])

        if p.distance(linestring) <= my_distance:
            margin_cell_list.append(all_cells[i])

    return np.asarray(margin_cell_list)


def get_core_cells(all_cells, shape, linestring, my_distance):
    """
    Gets all cells that are inside the invasive margin, but not within my_distance to it.
    This includes TB, CD3, and CD8.
    """
    core_cell_list = []

    assert (type(all_cells) == np.ndarray and all_cells.shape[1] >= 4), "type of all_cells is {}, all_cells.shape is {}".format(type(all_cells), all_cells.shape)

    x_avgs = (all_cells[:, 0] + all_cells[:, 1]) / 2
    y_avgs = (all_cells[:, 2] + all_cells[:, 3]) / 2
    xy_avgs = np.array((x_avgs, y_avgs)).transpose()

    for i in range(len(all_cells)):
        p = Point(xy_avgs[i])

        if p.distance(linestring) > my_distance and p.within(shape):
            core_cell_list.append(all_cells[i])

    return np.asarray(core_cell_list)


def get_margin_core_cells(all_cells, shape, linestring, my_distance):
    """
    Excludes all cells that are more than my_distance outside the
    invasive margin. This includes TB, CD3, and CD8.
    """
    margin_cell_list, core_cell_list = [], []

    assert (type(all_cells) == np.ndarray and all_cells.shape[1] >= 4), "type of all_cells is {}, all_cells.shape is {}".format(type(all_cells), all_cells.shape)

    x_avgs = (all_cells[:, 0] + all_cells[:, 1]) / 2
    y_avgs = (all_cells[:, 2] + all_cells[:, 3]) / 2
    xy_avgs = np.array((x_avgs, y_avgs)).transpose()

    for i in range(len(all_cells)):
        p = Point(xy_avgs[i])

        if p.distance(linestring) <= my_distance:
            margin_cell_list.append(all_cells[i])
        elif p.within(shape):
            core_cell_list.append(all_cells[i])

    return np.asarray(margin_cell_list), np.asarray(core_cell_list)


def get_lymphocytes_on_margin(cd3cd8, annotations_file):
    """
    Gets all lymphocytes (CD3/CD8) within 500um of the invasive margin.
    """
    all_cd3, all_cd8 = cd3cd8

    linestring = load_linestring(annotations_file)

    margin_cd3, margin_cd8 = [], []
    margin_cd3, margin_cd8 = np.asarray(get_margin_cells(all_cd3, linestring, 500)), np.asarray(get_margin_cells(all_cd8, linestring, 500))

    return margin_cd3, margin_cd8


def get_lymphocytes_on_margin_and_core(cd3cd8, annotations_file):
    """
    Gets all lymphocytes (CD3/CD8) except those that are more than 500um outside
    the invasive margin
    """
    all_cd3, all_cd8 = cd3cd8

    shape = load_shape(annotations_file)
    linestring = load_linestring(annotations_file)

    margin_cd3, core_cd3 = get_margin_core_cells(all_cd3, shape, linestring, 500)
    margin_cd8, core_cd8 = get_margin_core_cells(all_cd8, shape, linestring, 500)

    return np.asarray(margin_cd3), np.asarray(core_cd3), np.asarray(margin_cd8), np.asarray(core_cd8)


def partition_lymphocytes_for_heatmap(items, tile_size=886, to_list=True, input_type="clean",
                                      by_metric=True, scale=1):
    """
    Given appropriate parameters and a list of lymphocytes,
    divide them into a (T x T) grid and take the counts of each type
    of object for each tile. Return these 'heatmaps', rotated accordingly.
    """
    partitioned_items, _, _, _, _, _, _, _ = partition.partition(items, tile_size=tile_size,
                                                                 to_list=to_list, input_type=input_type,
                                                                 by_metric=by_metric, scale=scale)

    heatmap_cd3 = np.zeros(partitioned_items.shape, dtype=int)
    heatmap_cd8 = np.zeros(partitioned_items.shape, dtype=int)
    heatmap_cd3cd8 = np.zeros(partitioned_items.shape, dtype=int)

    (width, height) = partitioned_items.shape

    for col in range(width):
        for row in range(height):
            cells = partitioned_items[col][row]
            heatmap_cd3cd8[col][row] += len(cells)
            for cell in cells:
                if cell[4] > 0:
                    heatmap_cd3[col][row] += 1
                elif cell[5] > 0:
                    heatmap_cd8[col][row] += 1

    heatmap_cd3 = np.flip(np.rot90(heatmap_cd3), 0)
    heatmap_cd8 = np.flip(np.rot90(heatmap_cd8), 0)
    heatmap_cd3cd8 = np.flip(np.rot90(heatmap_cd3cd8), 0)

    return heatmap_cd3, heatmap_cd8, heatmap_cd3cd8


def partition_tb_for_heatmap(items, tile_size=886, to_list=True, input_type="clean",
                             by_metric=True, scale=1):
    """
    Given appropriate parameters and a list of tumour buds,
    divide them into a (T x T) grid and take the counts of tumour buds
    for each tile. Return this 'heatmap', rotated/flipped accordingly.
    """
    partitioned_tb, _, _, _, _, _, _, _ = partition.partition(items, tile_size=tile_size,
                                                              to_list=to_list, input_type=input_type,
                                                              by_metric=by_metric, scale=scale)

    heatmap_tb = np.zeros(partitioned_tb.shape, dtype=int)

    for i in range(partitioned_tb.shape[0]):
        for j in range(partitioned_tb.shape[1]):
            heatmap_tb[i][j] += len(partitioned_tb[i][j])

    heatmap_tb = np.flip(np.rot90(heatmap_tb), 0)

    return heatmap_tb


def partition_nearby_lymphocytes_tb_for_heatmap(items, tile_size=886, to_list=True, input_type="mixed",
                                                by_metric=True, scale=1):
    """
    Given appropriate parameters and a list of tumour buds and lymphocytes within 50um of any tumour bud,
    divide them into a (T x T) grid and take the counts of tumour buds and lymphocytes
    for each tile. Return these 'heatmaps', rotated accordingly.
    """
    partitioned_items, tiles, \
        lymph_w, lymph_h,     \
        clust_w, clust_h,     \
        x_step, y_step = partition.partition(items, tile_size=tile_size,
                                             to_list=to_list, input_type=input_type,
                                             by_metric=by_metric, scale=scale)

    counts = Cluster.get_lymphocyte_cluster_density_heatmap(partitioned_items, partitioned_items.shape, tiles,
                                                            lymph_w, lymph_h, clust_w, clust_h, x_step, y_step)

    heatmap_tb = np.zeros(counts.shape, dtype=int)
    heatmap_nearby_cd3 = np.zeros(counts.shape, dtype=int)
    heatmap_nearby_cd8 = np.zeros(counts.shape, dtype=int)
    heatmap_nearby_cd3cd8 = np.zeros(counts.shape, dtype=int)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            heatmap_nearby_cd3[i][j] += counts[i][j][0]
            heatmap_nearby_cd8[i][j] += counts[i][j][1]
            heatmap_nearby_cd3cd8[i][j] += counts[i][j][2]
            heatmap_tb[i][j] += counts[i][j][3]

    heatmap_nearby_cd3 = np.flip(np.rot90(heatmap_nearby_cd3), 0)
    heatmap_nearby_cd8 = np.flip(np.rot90(heatmap_nearby_cd8), 0)
    heatmap_nearby_cd3cd8 = np.flip(np.rot90(heatmap_nearby_cd3cd8), 0)
    heatmap_tb = np.flip(np.rot90(heatmap_tb), 0)

    return heatmap_tb, heatmap_nearby_cd3, heatmap_nearby_cd8, heatmap_nearby_cd3cd8


def get_cluster_code(cluster_name):
    """
    Path/to/file/.../ -> split on the slash and get the last token
    SN---.p -> just truncate the last 2 characters
    """
    filename = cluster_name.split('/')[-1]
    return filename[0:-2]


def get_annotations_code(annotations_name):
    """
    SN---_----_job----.annotations -> get the first 5 characters
    """
    filename = annotations_name.split('/')[-1]
    return filename[0:5]


def load_cluster_list(cluster_files):
    """
    Cluster file, load through pickle.
    """
    for cluster_filename in sorted(glob.glob(cluster_files)):
        cluster_data = pickle.load(open(cluster_filename, "rb"))
        yield cluster_filename, cluster_data


def load_lymphocyte_list(lymphocyte_files):
    """
    Lymphocyte file, load through csv.
    """
    for lymphocyte_filename in sorted(glob.glob(lymphocyte_files)):
        lymphocyte_data = extract.load(lymphocyte_filename)
        yield lymphocyte_filename, lymphocyte_data


def load_len_lymphocyte_list(lymphocyte_files):
    """
    Lymphocyte file, load through csv.
    """
    for lymphocyte_filename in sorted(glob.glob(lymphocyte_files)):
        lymphocyte_data = extract.load(lymphocyte_filename)
        yield lymphocyte_filename, len(lymphocyte_data)


def load_cluster_tb_metadata(cluster_metadata_files):
    """
    Cluster metadata file, load through .txt file.
    Get the appropriate line, get the val at the end.
    """
    for metadata_filename in tqdm(sorted(glob.glob(cluster_metadata_files))):
        f = open(metadata_filename, 'r')
        contents = f.readlines()
        cluster_count_val = -1
        tb_count_val = -1
        for line in contents:
            if "CLU" in line:
                cluster_count_val = int(line.split(": ")[-1])
            if "TB" in line:
                tb_count_val = int(line.split(": ")[-1])
        if cluster_count_val != -1 and tb_count_val != -1:
            yield metadata_filename, cluster_count_val, tb_count_val
        else:
            raise Exception("No cluster/TB value associated in file " + metadata_filename)


def load_lymphocyte_metadata(lymphocyte_metadata_files):
    """
    Lymphocyte metadata file, load through .txt file.
    Get the appropriate line, get the val at the end.
    """
    for metadata_filename in tqdm(sorted(glob.glob(lymphocyte_metadata_files))):
        f = open(metadata_filename, 'r')
        contents = f.readlines()
        cd3cd8_count_val = -1
        cd3_count_val = -1
        cd8_count_val = -1
        for line in contents:
            if "CD3&CD8" in line:
                cd3cd8_count_val = int(line.split(": ")[-1])
                continue
            if "CD3" in line:
                cd3_count_val = int(line.split(": ")[-1])
                continue
            if "CD8" in line:
                cd8_count_val = int(line.split(": ")[-1])
                continue

        if cd3cd8_count_val > -1 and cd3_count_val > -1 and cd8_count_val > -1:
            yield metadata_filename, cd3_count_val, cd8_count_val, cd3cd8_count_val
        else:
            raise Exception("No CD3/CD8/CD3CD8 value associated in file " + metadata_filename)


def load_margin(margin_files):
    """
    Margin defining shape of invasive margin, load through helpers.
    """
    for margin_filename in sorted(glob.glob(margin_files)):
        linestring, shape = load_linestring(margin_filename), load_shape(margin_filename)
        yield margin_filename, linestring, shape


def load_meta(obj_type, meta_data_files, code_extract_func):
    """
    meta_data_values = [(meta_code1, meta_val1), (meta_code2, meta_val2), ...]
    """
    meta_data_values = []

    for meta_data_filename in tqdm(sorted(glob.glob(meta_data_files))):
        f = open(meta_data_filename, 'r')
        contents = f.readlines()
        obj_count_val = -1

        for line in contents:
            if obj_type in line:
                obj_count_val = int(line.split(": ")[-1])

        meta_data_values.append((code_extract_func(meta_data_filename), obj_count_val))

    return meta_data_values


def code_extract_func(filename):
    return filename.split('/')[-1].split('.')[0]


def visualisation_block(visualise_func, data, linestring, shape, title, image_filename):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = visualise_func(data, ax, title=title)

    xs = [x for (x, y) in linestring.coords]
    ys = [y for (x, y) in linestring.coords]

    patch = PolygonPatch(shape, fc="red", alpha=0.2, zorder=2)
    ax.add_patch(patch)
    ax.plot(xs, ys)
    fig.savefig(image_filename, dpi=100, bbox_inches="tight")


def visualisation_interaction_block(data, linestring, shape, title, image_filename):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = Visualiser.visualise_clusters_and_lymphocytes_on_ax(data, ax, title=title)

    xs = [x for (x, y) in linestring.coords]
    ys = [y for (x, y) in linestring.coords]

    patch = PolygonPatch(shape, fc="red", alpha=0.2, zorder=2)
    ax.add_patch(patch)
    ax.plot(xs, ys)
    fig.savefig(image_filename, dpi=100, bbox_inches="tight")


def statistic_generation_block(partition_func, statistic_funcs, data, meta_data_value, d, heatmap_filename_func, target_directory_stub, filename):
    heatmap = partition_func(data, meta_data_value, d)

    assert heatmap.shape[2] == 4, "Output of transform should be 3D numpy array with 4 layers"  # Layers for TB, CD3, CD8, CD3CD8 respectively

    for (statistic_name, statistic_func, statistic_mask) in statistic_funcs:
        transformed_data = statistic_func(heatmap)

        for i, obj_type in enumerate(["TB_CD3", "TB_CD8", "TB_CD3CD8"]):

            heatmap_filename = heatmap_filename_func(target_directory_stub + obj_type + "_" + str(d) + "/", filename, statistic_name)

            p_file = open(heatmap_filename, "wb")
            pickle.dump(transformed_data[:, :, i+1], p_file)
            p_file.close()

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            title = "Heatmap for " + target_directory_stub.split('/')[-1] + obj_type + " with d=" + str(d)
            ax = Visualiser.visualise_heatmaps(transformed_data[:, :, i+1], ax, title, statistic_mask(transformed_data[:, :, i+1]))
            image_filename = heatmap_filename_func(target_directory_stub + obj_type + "_IMAGES/", filename, "_heatmap" + "_" + str(d) + "_" + statistic_name)[0:-2] + ".png"
            fig.savefig(image_filename, dpi=100, bbox_inches="tight")


def get_images_with_margins(obj_type, obj_files, expected_num_files, meta_data_files, load_obj_func, target_directory_stub, convert_func, visualise_func, image_filename_func, partition_func, heatmap_filename_func, margin_files, d):
    meta_data_pairs = load_meta(obj_type, meta_data_files, code_extract_func)
    meta_data_codes = [code for (code, val) in meta_data_pairs]
    meta_data_values = [val for (code, val) in meta_data_pairs]

    num_files = 0
    obj_file_type = obj_type if obj_type != "CD3&CD8" else "CD3CD8"

    for (filename, raw_data), (margin_filename, linestring, shape) in zip(load_obj_func(obj_files), load_margin(margin_files)):
        meta_data_value = 0
        try:
            meta_data_value = meta_data_values[meta_data_codes.index(code_extract_func(filename))]
        except:
            raise Exception("Meta-data does not have a value for file " + filename + "\nwith obj_type " + obj_type)

        all_data = convert_func(raw_data, meta_data_value)

        assert type(all_data) == np.ndarray

        ## ALL
        print(code_extract_func(filename) + ": " + "All {}".format(obj_type))
        visualisation_block(visualise_func, all_data, linestring, shape, "All {}".format(obj_type), image_filename_func(target_directory_stub + "ALL_{}/".format(obj_file_type), filename))
        statistic_generation_block(partition_func, [("_density", density)], all_data, meta_data_value, [heatmap_filename_func(target_directory_stub + "ALL_{}/".format(obj_file_type), filename)])

        margin_data, core_data = get_margin_core_cells(all_data, shape, linestring, 500)

        assert margin_data.shape[0] > 0 and margin_data.shape[1] >= 4, "Something wrong with {} margin data".format(obj_type)
        assert core_data.shape[0] > 0 and core_data.shape[1] >= 4, "Something wrong with {} core data".format(obj_type)

        ## MARGIN
        print(code_extract_func(filename) + ": " + "{} (IM)".format(obj_type))
        visualisation_block(visualise_func, margin_data, linestring, shape, "{} (IM)".format(obj_type), image_filename_func(target_directory_stub + "MARGIN_{}/".format(obj_file_type), filename))
        statistic_generation_block(partition_func, [("_density", density)], margin_data, meta_data_value, [heatmap_filename_func(target_directory_stub + "MARGIN_{}/".format(obj_file_type), filename)])

        ## CORE
        print(code_extract_func(filename) + ": " + "{} (CT)".format(obj_type))
        visualisation_block(visualise_func, core_data, linestring, shape, "{} (CT)".format(obj_type), image_filename_func(target_directory_stub + "CORE_{}/".format(obj_file_type), filename))
        statistic_generation_block(partition_func, [("_density", density)], core_data, meta_data_value, [heatmap_filename_func(target_directory_stub + "CORE_{}/".format(obj_file_type), filename)])

        ## IMCT
        print(code_extract_func(filename) + ": " + "{} (IMCT)".format(obj_type))
        imct_data = np.concatenate((margin_data, core_data), axis=0)

        assert imct_data.shape[0] == (margin_data.shape[0] + core_data.shape[0]), "Margin and core cells not added properly"
        del(margin_data)  # numpy creates a copy on concat, so safe to delete
        del(core_data)    # these two arrays

        visualisation_block(visualise_func, imct_data, linestring, shape, "{} (IMCT)".format(obj_type), image_filename_func(target_directory_stub + "IMCT_{}/".format(obj_file_type), filename))
        statistic_generation_block(partition_func, [("_density", density)], imct_data, meta_data_value, [heatmap_filename_func(target_directory_stub + "IMCT_{}/".format(obj_file_type), filename)])

        num_files += 1

    assert(num_files == expected_num_files)


def get_images_with_interactions_condensed(tb_files, lym_files, expected_num_files, meta_data_files,
                                           target_directory_stub, heatmap_transform, statistics, image_filename_func,
                                           heatmap_filename_func, margin_files, d):
    tb_meta_pairs = load_meta("TB", meta_data_files, code_extract_func)
    tb_meta_codes = [code for (code, val) in tb_meta_pairs]
    tb_meta_values = [val for (code, val) in tb_meta_pairs]

    cd3_meta_pairs = load_meta("CD3", meta_data_files, code_extract_func)
    cd8_meta_pairs = load_meta("CD8", meta_data_files, code_extract_func)
    cd3cd8_meta_pairs = load_meta("CD3&CD8", meta_data_files, code_extract_func)

    cd3_meta_codes = [code for (code, val) in cd3_meta_pairs]
    cd8_meta_codes = [code for (code, val) in cd8_meta_pairs]
    cd3cd8_meta_codes = [code for (code, val) in cd3cd8_meta_pairs]
    assert(cd3_meta_codes == cd8_meta_codes and cd8_meta_codes == cd3cd8_meta_codes)

    cd3_meta_values = [val for (code, val) in cd3_meta_pairs]
    cd8_meta_values = [val for (code, val) in cd8_meta_pairs]
    cd3cd8_meta_values = [val for (code, val) in cd3cd8_meta_pairs]

    num_files = 0

    for (tb_filename, cluster_raw_data), (lym_filename, lym_raw_data), (margin_filename, linestring, shape) in zip(load_cluster_list(tb_files),
                                                                                                                   load_lymphocyte_list(lym_files),
                                                                                                                   load_margin(margin_files)):
        tb_meta_value = 0
        cd3_meta_value, cd8_meta_value, cd3cd8_meta_value = 0, 0, 0
        try:
            tb_meta_value = tb_meta_values[tb_meta_codes.index(code_extract_func(tb_filename))]
        except:
            raise Exception("Meta-data does not have a value for file " + tb_filename + "\nwith obj_type TB")

        try:
            cd3_meta_value = cd3_meta_values[cd3_meta_codes.index(code_extract_func(lym_filename))]
            cd8_meta_value = cd8_meta_values[cd8_meta_codes.index(code_extract_func(lym_filename))]
            cd3cd8_meta_value = cd3cd8_meta_values[cd3cd8_meta_codes.index(code_extract_func(lym_filename))]
        except:
            raise Exception("Meta-data does not have a value for file " + lym_filename)

        all_tb_data = convert_cluster_to_tb(cluster_raw_data, tb_meta_value)
        all_lym_data = convert_lymphocyte_to_cd3cd8(lym_raw_data, cd3cd8_meta_value)

        assert type(all_tb_data) == np.ndarray and type(all_lym_data) == np.ndarray

        #####################
        ### CD3&CD8 BLOCK ###
        #####################
        combined_tb_lym_data = merge_tb_lym(all_tb_data, all_lym_data)

        ## ALL - TB AND CD3/CD8
        title = "All TB and CD3/CD8"
        print(code_extract_func(tb_filename) + ": " + title)
        visualisation_interaction_block(combined_tb_lym_data, linestring, shape, title, image_filename_func(target_directory_stub + "ALL_TB_CD3CD8_IMAGES/", tb_filename))
        statistic_generation_block(heatmap_transform, statistics, combined_tb_lym_data, tb_meta_value + cd3cd8_meta_value, 50, heatmap_filename_func, target_directory_stub + "ALL_", tb_filename)
        statistic_generation_block(heatmap_transform, statistics, combined_tb_lym_data, tb_meta_value + cd3cd8_meta_value, 100, heatmap_filename_func, target_directory_stub + "ALL_", tb_filename)

        start_time = timeit.default_timer()
        tb_lym_margin_data, tb_lym_core_data = get_margin_core_cells(combined_tb_lym_data, shape, linestring, 500)
        elapsed = timeit.default_timer() - start_time
        print("get_margin_core_cells() took:", elapsed)
        del(combined_tb_lym_data)

        assert tb_lym_margin_data.shape[0] > 0 and tb_lym_margin_data.shape[1] >= 4, "Something wrong with TB-CD3CD8 margin data"
        assert tb_lym_core_data.shape[0] > 0 and tb_lym_core_data.shape[1] >= 4, "Something wrong with TB-CD3CD8 core data"

        ## MARGIN - TB AND CD3/CD8
        title = "TB and CD3/CD8 (IM)"
        print(code_extract_func(tb_filename) + ": " + title)
        visualisation_interaction_block(tb_lym_margin_data, linestring, shape, title, image_filename_func(target_directory_stub + "MARGIN_TB_CD3CD8_IMAGES/", tb_filename))
        statistic_generation_block(heatmap_transform, statistics, tb_lym_margin_data, len(tb_lym_margin_data), 50, heatmap_filename_func, target_directory_stub + "MARGIN_", tb_filename)
        statistic_generation_block(heatmap_transform, statistics, tb_lym_margin_data, len(tb_lym_margin_data), 100, heatmap_filename_func, target_directory_stub + "MARGIN_", tb_filename)

        ## CORE - TB AND CD3/CD8
        title = "TB and CD3/CD8 (CT)"
        print(code_extract_func(tb_filename) + ": " + title)
        visualisation_interaction_block(tb_lym_core_data, linestring, shape, title, image_filename_func(target_directory_stub + "CORE_TB_CD3CD8_IMAGES/", tb_filename))
        statistic_generation_block(heatmap_transform, statistics, tb_lym_core_data, len(tb_lym_core_data), 50, heatmap_filename_func, target_directory_stub + "CORE_", tb_filename)
        statistic_generation_block(heatmap_transform, statistics, tb_lym_core_data, len(tb_lym_core_data), 100, heatmap_filename_func, target_directory_stub + "CORE_", tb_filename)

        ## IMCT - TB AND CD3/CD8
        title = "TB and CD3/CD8 (IMCT)"
        print(code_extract_func(tb_filename) + ": " + title)
        tb_lym_imct_data = np.concatenate((tb_lym_margin_data, tb_lym_core_data), axis=0)

        assert tb_lym_imct_data.shape[0] == (tb_lym_margin_data.shape[0] + tb_lym_core_data.shape[0]), "Margin and core cells not added properly"

        visualisation_interaction_block(tb_lym_imct_data, linestring, shape, title, image_filename_func(target_directory_stub + "IMCT_TB_CD3CD8_IMAGES/", tb_filename))
        statistic_generation_block(heatmap_transform, statistics, tb_lym_imct_data, len(tb_lym_imct_data), 50, heatmap_filename_func, target_directory_stub + "IMCT_", tb_filename)
        statistic_generation_block(heatmap_transform, statistics, tb_lym_imct_data, len(tb_lym_imct_data), 100, heatmap_filename_func, target_directory_stub + "IMCT_", tb_filename)

        num_files += 1

    assert(num_files == expected_num_files)

# def get_images_with_interactions(tb_files, tb_convert, lym_files, lym_convert, tb_lym_interaction_funcs, expected_num_files, meta_data_files, target_directory_stub, visualise_funcs, title_tree, output_filename_func, margin_files, d):
    """

    TB and do CD3, CD8, CD3&CD8 all at once.

    Four(!) levels of looping:
    for all files:
        meta vals...
        for tb_select, lym_select (must be the same region!):
            clean_tb, clean_lym = select_tb(), select_lym()

            for all interactions between current tb and lym selection (target_directories):
                create visualisation (pointwise-scatter with margin)
                for all statistics of interactions between tb and lym:
                    ...
        num_files++

    Titles, target directories either need stubs or need to be hierarichial. Visualisation funcs are fine.
    """
    tb_meta_pairs = load_meta("TB", meta_data_files, code_extract_func)
    tb_meta_codes = [code for (code, val) in tb_meta_pairs]
    tb_meta_values = [val for (code, val) in tb_meta_pairs]

    cd3_meta_pairs = load_meta("CD3", meta_data_files, code_extract_func)
    cd8_meta_pairs = load_meta("CD8", meta_data_files, code_extract_func)
    cd3cd8_meta_pairs = load_meta("CD3&CD8", meta_data_files, code_extract_func)

    cd3_meta_codes = [code for (code, val) in cd3_meta_pairs]
    cd8_meta_codes = [code for (code, val) in cd8_meta_pairs]
    cd3cd8_meta_codes = [code for (code, val) in cd3cd8_meta_pairs]
    assert(cd3_meta_codes == cd8_meta_codes and cd8_meta_codes == cd3cd8_meta_codes)

    cd3_meta_values = [val for (code, val) in cd3_meta_pairs]
    cd8_meta_values = [val for (code, val) in cd8_meta_pairs]
    cd3cd8_meta_values = [val for (code, val) in cd3cd8_meta_pairs]

    num_files = 0

    for (tb_file, tb_raw_data), (lym_file, lym_raw_data) in zip(load_cluster_list(tb_files), load_lymphocyte_list(lym_files)):
        tb_meta_value, lym_meta_value = 0, 0
        try:
            tb_meta_value = tb_meta_values[tb_meta_codes.index(code_extract_func(tb_file))]
            cd3_meta_value = cd3_meta_values[cd3_meta_codes.index(code_extract_func(lym_file))]
            cd8_meta_value = cd8_meta_values[cd8_meta_codes.index(code_extract_func(lym_file))]
            cd3cd8_meta_value = cd3cd8_meta_values[cd3cd8_meta_codes.index(code_extract_func(lym_file))]
        except:
            raise Exception("Meta-data does not have a value for file-pair (" + tb_file + ", " + lym_file + ")")

        tb_data = tb_convert(tb_raw_data, tb_meta_value)
        lym_data = lym_convert(tb_raw_data, lym_meta_value)

        for selection in region_selections:
            selected_tb = selection(tb_data, linestring, shape, d)
            selected_lym = selection(lym_data, linestring, shape, d)
            for interaction_func in tb_lym_interaction_funcs:
                tb_lym_data = interaction_func(selected_tb, selected_lym)
                for feature in feature_funcs:
                    feature_map = feature(tb_lym_data)

                    tb_data = feature_map["TB"]
                    cd3_data = feature_map["CD3"]
                    cd8_data = feature_map["CD8"]
                    cd3cd8_data = feature_map["CD3&CD8"]

            # for beta_transform, title_list, target_directory_list in zip(beta_transforms, title_tree, target_directory_tree):
            #     clean_beta_data = beta_transform(beta_raw_data, beta_meta_value)

            #     for alpha_beta_transform, visualise_func, title, target_directory in zip(alpha_beta_transforms, visualise_funcs, title_list, target_directory_list):
            #         clean_data = alpha_beta_transform(clean_alpha_data, clean_beta_data)

            #         fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            #         ax = visualise_func(clean_data, ax, title=title)

            #         output_filename = output_filename_func(target_directory, alpha_file)  # doesn't matter whether alpha-file or beta-file for the name
            #         fig.savefig(output_filename, dpi=100, bbox_inches="tight")

        num_files += 1

    assert(num_files == expected_num_files)


def convert_cluster_to_tb(cluster_data, expected_size):
    data = np.array([[int(row[2]), int(row[1]), int(row[4]), int(row[3])] for row in cluster_data if int(row[0]) > 0 and int(row[0]) < 5])
    assert(len(data) == expected_size or expected_size == -1)  # Allow a by-pass in the case of results we can't tell ahead of time.
    return data


def convert_lymphocyte_to_cd3(lymphocyte_data, expected_size):
    data = np.array([[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])] for row in lymphocyte_data if int(row[4]) > 0])
    assert(len(data) == expected_size or expected_size == -1)
    return data


def convert_lymphocyte_to_cd8(lymphocyte_data, expected_size):
    data = np.array([[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])] for row in lymphocyte_data if int(row[5]) > 0])
    assert(len(data) == expected_size or expected_size == -1)
    return data


def convert_lymphocyte_to_cd3cd8(lymphocyte_data, expected_size):
    data = np.array([[int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])] for row in lymphocyte_data if int(row[4]) > 0 or int(row[5]) > 0])
    assert(len(data) == expected_size or expected_size == -1)
    return data


def density(heatmap):
    """
    Assumes a non-interacting heatmap of lists.
    """
    densities = np.vectorize(len)(heatmap)

    return densities


def density_values(heatmap):
    return heatmap  # it already is the answer


def ratio_values(heatmap):
    w, h = heatmap.shape[0], heatmap.shape[1]

    ratios = np.zeros(heatmap.shape)

    for i in range(w):
        for j in range(h):
            tb_val = heatmap[i][j][0]
            cd3_val = heatmap[i][j][1]
            cd8_val = heatmap[i][j][2]
            cd3cd8_val = heatmap[i][j][3]

            ratios[i][j][0] = 0  # don't care about tb, only the other 3
            ratios[i][j][1] = cd3_val / tb_val if tb_val != 0 else -1
            ratios[i][j][2] = cd8_val / tb_val if tb_val != 0 else -1
            ratios[i][j][3] = cd3cd8_val / tb_val if tb_val != 0 else -1

    return ratios


# def get_num_hotspots(heatmap, p_value_threshold=0.05, Z_score_threshold=1.96):
#     hotspots = np.zeros((heatmap.shape[0], heatmap.shape[1], 2))  # layer 0 corresponds to coldspots, layer 1 to hotspots
#     for i in range(array_of_heatmaps.shape[0]):
#         all_count = 0
#         for j in range(array_of_heatmaps.shape[1]):
#             num_hot, num_cold = 0, 0
#             (n1, n2) = heatmap.shape

#             points = [[(x, y) for y in range(n2)] for x in range(n1)]
#             dist = math.sqrt(2)

#             with sb.axes_style("white"):
#                 msk = heatmap != 0

#                 points = [points[r][c] for r in range(n1) for c in range(n2) if msk[r][c]]

#                 w = DistanceBand(points, threshold=dist)

#                 lg_star = G_Local(heatmap[msk], w, transform='B', star=True)

#                 Z_scores = np.zeros(heatmap.shape, dtype=np.float32)
#                 p_values = np.zeros(heatmap.shape, dtype=np.float32)

#                 ind = 0
#                 for r in range(n1):
#                     for c in range(n2):
#                         if msk[r][c]:
#                             Z_scores[r, c] = lg_star.Zs[ind]
#                             p_values[r, c] = lg_star.p_sim[ind]
#                             ind += 1
#                         else:
#                             Z_scores[r][c] = -100.0  # why -100?

#                 for ind in range(lg_star.Zs.shape[0]):
#                     if lg_star.p_sim[ind] < p_value_threshold:

#                         if lg_star.Zs[ind] > Z_score_threshold:
#                             num_hot += 1
#                         elif lg_star.Zs[ind] < -(Z_score_threshold):
#                             num_cold += 1

#                 array_of_hotspots[i, all_count] = num_cold
#                 array_of_hotspots[i, all_count+1] = num_hot

#     return heatmap


def get_tb_images(cluster_files, expected_num_files, meta_data_files, target_directory_stub, margin_files, d):
    # All, IM, CT, IM&CT
    obj_type = "TB"
    load_obj_func = load_cluster_list
    convert_func = lambda data, exp_size: convert_cluster_to_tb(data, exp_size)

    visualise_func = lambda data, ax, title: Visualiser.visualise_clusters_on_ax(data, ax, size_included=False, title=title)

    partition_func = lambda entire_list, exp_size: partition.partition(entire_list, tile_size=886, to_list=True, input_type="clean", by_metric=True, scale=1)[0]

    image_filename_func = lambda target_directory, filename: target_directory + code_extract_func(filename) + ".png"
    heatmap_filename_func = lambda target_directory, filename, statistic: target_directory + code_extract_func(filename) + statistic + ".p"

    args = (obj_type, cluster_files, expected_num_files, meta_data_files, load_obj_func,
            target_directory_stub, convert_func, visualise_func, image_filename_func, partition_func,
            heatmap_filename_func, margin_files, d)

    get_images_with_margins(*args)


def get_lym_images(lym_type, lymphocyte_files, expected_num_files, meta_data_files, target_directory_stub, margin_files, d):
    # All, IM, CT, IM&CT, for CD3, CD8, CD3&CD8
    load_obj_func = load_lymphocyte_list
    if lym_type == "CD3":
        convert_func = lambda data, exp_size: convert_lymphocyte_to_cd3(data, exp_size)
    elif lym_type == "CD8":
        convert_func = lambda data, exp_size: convert_lymphocyte_to_cd8(data, exp_size)
    else:  # CD3&CD8
        convert_func = lambda data, exp_size: convert_lymphocyte_to_cd3cd8(data, exp_size)

    if lym_type == "CD3&CD8":
        visualise_func = lambda data, ax, title: Visualiser.visualise_lymphocytes_on_ax(data, ax, title=title)
    else:
        visualise_func = lambda data, ax, title: Visualiser.visualise_one_cd_on_ax(data, ax, "CD3", title=title, size=0.1, colour='r')

    partition_func = lambda entire_list, exp_size: partition.partition(entire_list, tile_size=886, to_list=True, input_type="clean", by_metric=True, scale=1)[0]

    image_filename_func = lambda target_directory, filename: target_directory + code_extract_func(filename) + ".png"
    heatmap_filename_func = lambda target_directory, filename, statistic: target_directory + code_extract_func(filename) + statistic + ".p"

    args = (lym_type, lymphocyte_files, expected_num_files, meta_data_files, load_obj_func,
            target_directory_stub, convert_func, visualise_func, image_filename_func,
            partition_func, heatmap_filename_func, margin_files, d)

    get_images_with_margins(*args)


def partition_and_transform_tb_interactions(items, meta_data_value, d):
    """
    Mixed partition, produce heatmap of interaction given some function of interaction.
    """
    init_num_objs = len(items)

    partitioned_items, tiles, \
        lymph_w, lymph_h,     \
        clust_w, clust_h,     \
        x_step, y_step = partition.partition(items, tile_size=886, to_list=True, input_type="mixed",
                                             by_metric=True, scale=1)

    num_objs = 0
    for i in range(partitioned_items.shape[0]):
        for j in range(partitioned_items.shape[1]):
            (lymphocytes, cancer_clusters) = partitioned_items[i][j]
            num_objs += (len(lymphocytes) + len(cancer_clusters))

    assert num_objs == init_num_objs and init_num_objs == meta_data_value, "num_objs: {}, init_num_objs: {}, and meta_data_value: {}".format(num_objs, init_num_objs, meta_data_value)

    heatmap = Cluster.get_interacting_object_counts(partitioned_items, d, partitioned_items.shape, tiles,
                                                    lymph_w, lymph_h, clust_w, clust_h, x_step, y_step)

    assert heatmap.shape[0] == partitioned_items.shape[0] and heatmap.shape[1] == partitioned_items.shape[1]

    heatmap = np.flip(np.rot90(heatmap), 0)

    return heatmap


def merge_tb_lym(tb_data, lym_data):
    """
    Assumes tb_data to be np array of form:

    +------+------+------+------+
    | xMin | xMax | yMin | yMax |
    +------+------+------+------+
    | .... | .... | .... | .... |
    +------+------+------+------+

    lym data to be np array of form:

    +------+------+------+------+-----+-----+
    | xMin | xMax | yMin | yMax | CD3 | CD8 |  <-- Note that CD3 and CD8 columns are (effectively) binary classifications.
    +------+------+------+------+-----+-----+
    | .... | .... | .... | .... | ... | ... |
    +------+------+------+------+-----+-----+

    With the output to be of form:

    +------+------+------+------+-----+-----+----+
    | xMin | xMax | yMin | yMax | CD3 | CD8 | TB |  <-- As with CD3/CD8 above, TB column is for binary classification.
    +------+------+------+------+-----+-----+----+
    | .... | .... | .... | .... | ... | ... | .. |
    +------+------+------+------+-----+-----+----+

    """
    tb_data = np.concatenate((tb_data, np.zeros((tb_data.shape[0], 2)), np.ones((tb_data.shape[0], 1))), axis=1)
    lym_data = np.concatenate((lym_data, np.zeros((lym_data.shape[0], 1))), axis=1)

    mixed_data = np.concatenate((lym_data, tb_data))

    assert len(mixed_data) == len(tb_data) + len(lym_data)

    return mixed_data


# Functions for selections of data


def select_all(data, linestring, shape, d):
    return data


def select_CT(data, linestring, shape, d):
    return np.asarray(get_core_cells(data, shape, linestring, d))


def select_IM(data, linestring, shape, d):
    return np.asarray(get_margin_cells(data, linestring, d))


def select_IMCT(data, linestring, shape, d):
    margin_cells, core_cells = get_margin_core_cells(data, shape, linestring, d)
    margin_cells.extend(core_cells)
    return np.asarray(margin_cells)


def get_tb_cd3_cd8_cd3cd8_images(root_dir, meta_data_dir, target_dir_root, expected_num_files):
    """
    All input dirs should have a terminating slash

    This function assumes that select(data, linestring, shape, d) => return convert(data), i.e. ALL-DATA, no interaction with margin
    """
    # TODO: confirm that d = 500um is the correct distance from margin to allow
    d = 500

    tb_target_directory_stub = target_dir_root+"IMAGES/"
    cd3_target_directory_stub = tb_target_directory_stub
    cd8_target_directory_stub = tb_target_directory_stub
    cd3cd8_target_directory_stub = tb_target_directory_stub

    tb_options = (root_dir+"TB_pickle_files/*", expected_num_files, meta_data_dir+"*", tb_target_directory_stub, root_dir+"margin_files/*", d)
    cd3_options = ("CD3", root_dir+"lymphocyte_csv_files/*", expected_num_files, meta_data_dir+"*", cd3_target_directory_stub, root_dir+"margin_files/*", d)
    cd8_options = ("CD8", root_dir+"lymphocyte_csv_files/*", expected_num_files, meta_data_dir+"*", cd8_target_directory_stub, root_dir+"margin_files/*", d)
    cd3cd8_options = ("CD3&CD8", root_dir+"lymphocyte_csv_files/*", expected_num_files, meta_data_dir+"*", cd3cd8_target_directory_stub, root_dir+"margin_files/*", d)

    get_tb_images(*tb_options)  # cluster_files, expected_num_files, meta_data_files, target_directories, margin_files, d
    get_lym_images(*cd3_options)  # (lym_type, lymphocyte_files, expected_num_files, meta_data_files, target_directories, margin_files, d)
    get_lym_images(*cd8_options)
    get_lym_images(*cd3cd8_options)


def get_tb_cd3_cd8_cd3cd8_interaction_images(root_dir, target_dir_root, expected_num_files):
    """
    All input dirs should have a terminating slash

    This function assumes that select(data, linestring, shape, d) returns:

     > return convert(data)  // ALL-DATA, but display with margin
     > return get_core(convert(data), linestring, shape, d)  // CORE TUMOUR (CT)
     > return get_margin(convert(data), linestring, shape, d)  // INVASIVE MARGIN (IM)
     > return get_margin_and_core(convert(data), linestring, shape, d)  // IM & CT
    """
    # d is the maximum distance between nearby lymphocytes and any TB
    d = 50

    target_directory_stub = target_dir_root+"IMAGES/"

    tb_files = root_dir+"TB_pickle_files/*"
    lym_files = root_dir+"lymphocyte_csv_files/*"
    margin_files = root_dir + "margin_files/*"
    meta_data_files = root_dir+"meta_data/*"
    partition_func = partition_and_transform_tb_interactions
    statistics = [("_density", density_values, lambda heatmap: heatmap == 0), ("_ratio", ratio_values, lambda heatmap: heatmap == -1)]
    image_filename_func = lambda target_directory, filename: target_directory + code_extract_func(filename) + ".png"
    heatmap_filename_func = lambda target_directory, filename, statistic: target_directory + code_extract_func(filename) + statistic + ".p"

    args = (tb_files, lym_files, expected_num_files, meta_data_files, target_directory_stub, partition_func, statistics, image_filename_func, heatmap_filename_func, margin_files, d)

    get_images_with_interactions_condensed(*args)








        # ax = fig.gca(projection='3d')

        # Add margin
        # ax = plot_line_and_shape(ax, linestring, shape)

        # X_3D = np.arange(0, heatmap_tb.shape[1], 1)
        # Y_3D = np.arange(0, heatmap_tb.shape[0], 1)
        # X_3D, Y_3D = np.meshgrid(X_3D, Y_3D)
        # Z = heatmap_tb

        # # print(X_3D.shape, Y_3D.shape, Z.shape)

        # # Plot the surface.
        # surf = ax.plot_surface(X_3D, Y_3D, Z, cmap=cm.hot, linewidth=0, antialiased=False, alpha=1)

        # # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()




# End of file
