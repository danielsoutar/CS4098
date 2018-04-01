from collections import deque
from tqdm import tqdm
import numpy as np
import math


d = 50

extended_width = 0
extended_height = 0

# Define field of interest to be just one 'ring' of windows around the tile in question (the upper limit is exclusive, hence 2).
# If d is too large for the window size, add more rings to our field until d is within field of interest.
width_field_range = [1, 2]
height_field_range = [1, 2]


def set_d(value):
    global d
    d = value


def set_extensions(w, h):
    global extended_width
    global extended_height
    extended_width = np.ceil(w / 2)
    extended_height = np.ceil(h / 2)


def increase_window_field_x(x_step, d, w1, w2):
    global width_field_range
    while x_step <= d + w1 + w2:
        width_field_range[0] += 1
        width_field_range[1] += 1
        x_step += x_step


def increase_window_field_y(y_step, d, h1, h2):
    global height_field_range
    while y_step <= d + h1 + h2:
        height_field_range[0] += 1
        height_field_range[1] += 1
        y_step += y_step


class Cluster:

    def __init__(self, cell):
        self.cells = []
        self.add_cell(cell)

    def add_cell(self, cell):
        self.cells.append(cell)

    def add_cells(self, cells):
        self.cells = self.cells + cells

    def surrender(self):
        """ This function surrenders its data - used when this cluster is subsumed by another. """
        temp_cells = self.cells[:]
        self.cells = None
        return temp_cells


def get_key(cell):
    return (cell[0], cell[1], cell[2], cell[3])


def get_cluster(image_clusters, cell):
    key = get_key(cell)
    return image_clusters[key]


def pure_get_cluster(image_clusters):
    return image_clusters[-1]


def create_cluster(cell, image_clusters):
    key = get_key(cell)
    image_clusters[key] = Cluster(cell)


def pure_create_cluster(cell, image_clusters):
    image_clusters.append(Cluster(cell))


def already_in_cluster(image_clusters, cell):
    key = get_key(cell)
    return key in image_clusters


def pure_add_to_current_cluster(cluster, cell):
    cluster.add_cell(cell)


def extend_current_cluster(cluster, neighbours):
    cluster.cells.extend(neighbours)


def add_to_current_cluster(cluster, image_clusters, cell):
    cluster.add_cell(cell)
    key = get_key(cell)
    image_clusters[key] = cluster


def update_winning_cluster(cluster, image_clusters, new_cells):
    cluster.add_cells(new_cells)
    for cell in new_cells:
        key = get_key(cell)
        image_clusters[key] = cluster


def in_same_cluster(cell1, cell2, image_clusters):
    cluster1 = get_cluster(image_clusters, cell1)
    cluster2 = get_cluster(image_clusters, cell2)
    return cluster1 == cluster2


def are_neighbours(c1, c2):
    xMin1, xMax1, yMin1, yMax1 = c1[0], c1[1], c1[2], c1[3]
    xMin2, xMax2, yMin2, yMax2 = c2[0], c2[1], c2[2], c2[3]

    width = (xMax2 - xMin2) + d
    height = (yMax2 - yMin2) + d

    xMinOk = (xMin1 - width) < xMin2
    xMaxOk = xMax2 < (xMax1 + width)
    yMinOk = (yMin1 - height) < yMin2
    yMaxOk = yMax2 < (yMax1 + height)

    return xMinOk and xMaxOk and yMinOk and yMaxOk


def are_extended_neighbours(c1, c2):
    xMin1, xMax1, yMin1, yMax1 = c1[0], c1[1], c1[2], c1[3]
    xMin2, xMax2, yMin2, yMax2 = c2[0], c2[1], c2[2], c2[3]

    width = (xMax2 - xMin2) + d + extended_width
    height = (yMax2 - yMin2) + d + extended_height

    xMinOk = (xMin1 - width) < xMin2
    xMaxOk = xMax2 < (xMax1 + width)
    yMinOk = (yMin1 - height) < yMin2
    yMaxOk = yMax2 < (yMax1 + height)

    return xMinOk and xMaxOk and yMinOk and yMaxOk


def get_all_neighbours(c1, image, cell_index):
    neighbours = []
    for i, c2 in enumerate(image[0:-cell_index-1]):
        if are_neighbours(c1, c2):
            neighbours.append(c2)

    return neighbours


def subsume(cluster1, cluster2, image_clusters):
    winner, loser = None, None

    if(len(cluster1.cells) >= len(cluster2.cells)):
        winner = cluster1
        loser = cluster2
    else:
        winner = cluster2
        loser = cluster1

    new_cells = loser.surrender()
    update_winning_cluster(winner, image_clusters, new_cells)


def simplest(image):
    image_clusters = {}

    reversed_image = image[::-1]
    for index, cell in tqdm(enumerate(reversed_image)):
        if not already_in_cluster(image_clusters, cell):
            create_cluster(cell, image_clusters)
        neighbours = get_all_neighbours(cell, image, index)
        for neighbour in neighbours:
            if not already_in_cluster(image_clusters, neighbour):
                cluster = get_cluster(image_clusters, cell)
                add_to_current_cluster(cluster, image_clusters, neighbour)
            elif not in_same_cluster(cell, neighbour, image_clusters):
                cluster = get_cluster(image_clusters, cell)
                neighbour_cluster = get_cluster(image_clusters, neighbour)
                subsume(cluster, neighbour_cluster, image_clusters)

    return image_clusters


def fishermans_algorithm(image, shape, windows, max_cell_width, max_cell_height):
    """
    Fisherman's algorithm on images to extract clusters.
    For every tile, for every available cell, get all neighbours and assign into a cluster.
    From each neighbour, recursively get all neighbours and assign into same cluster.
    Remove each cell along the way to reduce search for future cells.
    """
    (n1, n2) = shape

    clusters = []
    set_extensions(max_cell_width, max_cell_height)

    for i in tqdm(range(n1)):
        for j in tqdm(range(n2)):
            while image[i][j]:
                cell = image[i][j].pop()
                pure_create_cluster(cell, clusters)
                clusterise(cell, clusters, image, (i, j), n1, n2, windows)

    return clusters


def get_and_remove_all_neighbours(c1, image, neighbouring_indices, neighbouring_windows):
    neighbours = []
    indices_to_check = []

    for i, window in enumerate(neighbouring_windows):
        win = np.array([window[0], window[2], window[1], window[3]])
        if are_extended_neighbours(c1, win):
            indices_to_check.append(neighbouring_indices[i])

    for (win_i, win_j) in indices_to_check:
        base = len(image[win_i][win_j]) - 1
        for k, c2 in enumerate(reversed(image[win_i][win_j])):
            if are_neighbours(c1, c2):
                neighbours.append((c2, (win_i, win_j)))
                image[win_i][win_j].pop(base - k)

    return neighbours


def get_neighbouring_windows_fisherman(i, j, n1, n2, windows):
    neighbouring_indices = []
    neighbouring_windows = []

    for range_i in range(i - width_field_range[0], i + width_field_range[1]):
        for range_j in range(j - height_field_range[0], j + height_field_range[1]):
            if(range_i >= 0 and range_i < n1 and range_j >= 0 and range_j < n2):
                neighbouring_indices.append((range_i, range_j))

    for (win_i, win_j) in neighbouring_indices:
        neighbouring_windows.append(windows[win_i][win_j])

    return neighbouring_indices, neighbouring_windows


def clusterise(cell, image_clusters, image, index, n1, n2, windows):
    stack = deque([(cell, index)])
    cluster = pure_get_cluster(image_clusters)
    while stack:
        cell, index = stack.pop()
        neighbouring_indices, neighbouring_windows = get_neighbouring_windows_fisherman(index[0], index[1], n1, n2, windows)
        neighbours = get_and_remove_all_neighbours(cell, image, neighbouring_indices, neighbouring_windows)
        extend_current_cluster(cluster, neighbours)
        stack.extend(neighbours)


def get_and_remove_all_nearby_lymphocytes(cluster, image, neighbouring_indices, neighbouring_windows):
    nearby_cd3, nearby_cd8 = [], []
    indices_to_check = []

    for i, window in enumerate(neighbouring_windows):
        win = np.array([window[0], window[2], window[1], window[3]])
        if are_extended_neighbours(cluster, win):
            indices_to_check.append(neighbouring_indices[i])

    for (win_i, win_j) in indices_to_check:
        base = len(image[win_i][win_j][0]) - 1
        for k, lymphocyte in enumerate(reversed(image[win_i][win_j][0])):
            if are_nearby(cluster, lymphocyte):
                if lymphocyte[4] > 0:
                    nearby_cd3.append(lymphocyte)
                    image[win_i][win_j][0].pop(base - k)
                elif lymphocyte[5] > 0:
                    nearby_cd8.append(lymphocyte)
                    image[win_i][win_j][0].pop(base - k)

    return nearby_cd3, nearby_cd8


def are_nearby(cluster, lymphocyte):
    xAvg1, yAvg1 = (cluster[0] + cluster[1]) / 2, (cluster[2] + cluster[3]) / 2
    xAvg2, yAvg2 = (lymphocyte[0] + lymphocyte[1]) / 2, (lymphocyte[2] + lymphocyte[3]) / 2

    return math.sqrt(math.pow((xAvg2 - xAvg1), 2) + math.pow((yAvg2 - yAvg1), 2)) <= d


def find_nearby_lymphocytes(cluster, image, index, n1, n2, windows):
    neighbouring_indices, neighbouring_windows = get_neighbouring_windows_fisherman(index[0], index[1], n1, n2, windows)
    nearby_cd3, nearby_cd8 = get_and_remove_all_nearby_lymphocytes(cluster, image, neighbouring_indices, neighbouring_windows)

    return len(nearby_cd3), len(nearby_cd8)


# Get the ratio of CD3/CD8 to cancer clusters.
# Note that we find any CD3/CD8 cell within the specified margin (d) of any cancer cluster.
# We only find these cells once. Consequently we don't double-count!
def get_lymphocyte_cluster_ratio_heatmap(image, shape, windows, max_lymph_width, max_lymph_height, max_cluster_width, max_cluster_height, x_step, y_step,
                                         take_lymphocyte_ratio=True, take_cd3_ratio=False, take_cd8_ratio=False):
    if x_step <= d + max_lymph_width + max_cluster_width:
        increase_window_field_x(x_step, d, max_lymph_width, max_cluster_width)
    elif y_step <= d + max_lymph_height + max_cluster_height:
        increase_window_field_y(y_step, d, max_lymph_height, max_cluster_height)

    (n1, n2) = shape[0], shape[1]

    heatmap = np.zeros((n1, n2), dtype=np.float32)
    set_extensions(max_lymph_width, max_lymph_height)

    for i in range(n1):
        for j in range(n2):
            cd3_count, cd8_count, cluster_count = 0, 0, 0
            ratio = -1

            while image[i][j][1]:
                cluster = image[i][j][1].pop()
                cluster_count += 1
                local_cd3_count, local_cd8_count = find_nearby_lymphocytes(cluster, image, (i, j), n1, n2, windows)
                cd3_count += local_cd3_count
                cd8_count += local_cd8_count

            if cluster_count != 0:
                if take_lymphocyte_ratio:
                    ratio = (cd3_count + cd8_count) / cluster_count
                elif take_cd3_ratio:
                    ratio = cd3_count / cluster_count
                elif take_cd8_ratio:
                    ratio = cd8_count / cluster_count

            heatmap[i][j] = ratio

    return heatmap







# End of file
