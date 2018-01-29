from collections import deque
from tqdm import tqdm
import numpy as np


d = 7


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


extended_width = 0
extended_height = 0

def set_extensions(w, h):
    global extended_width
    global extended_height
    extended_width = np.ceil(w / 2)
    extended_height = np.ceil(h / 2)


def fishermans_algorithm(image, n, windows, max_cell_width, max_cell_height):
    """
    Fisherman's algorithm on images to extract clusters.
    For every tile, for every available cell, get all neighbours and assign into a cluster.
    From each neighbour, recursively get all neighbours and assign into same cluster.
    Remove each cell along the way to reduce search for future cells.
    """
    image_clusters = []
    set_extensions(max_cell_width, max_cell_height)

    for i in tqdm(range(n)):
        for j in tqdm(range(n)):
            while image[i][j]:
                cell = image[i][j].pop()
                pure_create_cluster(cell, image_clusters)
                clusterise(cell, image_clusters, image, (i, j), n, windows)

    return image_clusters


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


def get_neighbouring_windows_fisherman(i, j, n, windows):
    neighbouring_indices = []
    neighbouring_windows = []

    for range_i in range(i-1, i+2):
        for range_j in range(j-1, j+2):
            if(range_i >= 0 and range_i < n and range_j >= 0 and range_j < n):
                neighbouring_indices.append((range_i, range_j))

    for (win_i, win_j) in neighbouring_indices:
        neighbouring_windows.append(windows[win_i][win_j])

    return neighbouring_indices, neighbouring_windows


def clusterise(cell, image_clusters, image, index, n, windows):
    stack = deque([(cell, index)])
    cluster = pure_get_cluster(image_clusters)
    while stack:
        cell, index = stack.pop()
        neighbouring_indices, neighbouring_windows = get_neighbouring_windows_fisherman(index[0], index[1], n, windows)
        neighbours = get_and_remove_all_neighbours(cell, image, neighbouring_indices, neighbouring_windows)
        extend_current_cluster(cluster, neighbours)
        stack.extend(neighbours)









# End of file
