from collections import deque
from tqdm import tqdm
import numpy as np

d = 7


def set_d(value):
    global d
    d = value


class Cluster:

    def __init__(self, cell, tile):
        self.cells = []
        self.tiles = []
        self.add_cell(cell, tile)

    def add_cell(self, cell, tile):
        self.cells.append(cell)
        self.tiles.append(tile)

    def add_cells(self, cells):
        self.cells = self.cells + cells

    def add_tiles(self, tiles):
        self.tiles = self.tiles + tiles

    def surrender(self):
        """ This function surrenders its data - used when this cluster is subsumed by another. """
        temp_cells = self.cells[:]
        temp_tiles = self.tiles[:]
        self.cells = None
        self.tiles = None
        return temp_cells, temp_tiles


def get_key(cell):
    return (cell[0], cell[1], cell[2], cell[3])


def create_cluster(cell, tile, index):
    key = get_key(cell)
    tile[key] = Cluster(cell, index)


def create_and_return_cluster(cell, tile, index):
    key = get_key(cell)
    tile[key] = Cluster(cell, index)
    return tile[key]


def get_cluster(tile, cell):
    key = get_key(cell)
    return tile[key]


def add_to_current_cluster(cluster, image_clusters, cell, index):
    cluster.add_cell(cell, index)
    (i, j) = index
    tile = image_clusters[i][j]
    key = get_key(cell)
    tile[key] = cluster


def in_same_cluster(cell1, cell2, image_clusters, index1, index2):
    (i, j) = index1
    (n_i, n_j) = index2
    tile1 = image_clusters[i][j]
    tile2 = image_clusters[n_i][n_j]
    cluster1 = get_cluster(tile1, cell1)
    cluster2 = get_cluster(tile2, cell2)
    return cluster1 == cluster2


def update_winning_cluster(cluster, image_clusters, new_cells, new_tiles):
    cluster.add_cells(new_cells)
    cluster.add_tiles(new_tiles)
    for cell, (i, j) in zip(new_cells, new_tiles):
        tile = image_clusters[i][j]
        key = get_key(cell)
        tile[key] = cluster


def already_in_cluster(tile, cell):
    key = get_key(cell)
    return key in tile


def subsume(cluster1, cluster2, image_clusters):
    winner, loser = None, None

    if(len(cluster1.cells) >= len(cluster2.cells)):
        winner = cluster1
        loser = cluster2
    else:
        winner = cluster2
        loser = cluster1

    new_cells, new_tiles = loser.surrender()
    update_winning_cluster(winner, image_clusters, new_cells, new_tiles)


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


def get_all_neighbours(c1, image, cell_index, tile_index, neighbouring_windows, neighbouring_indices, num_searched_windows):
    neighbours = []
    tiles = []
    img_i, img_j = tile_index[0], tile_index[1]
    num_searched_windows += 1
    for i, c2 in enumerate(image[img_i][img_j][0:-cell_index-1]):
        if are_neighbours(c1, c2):
            neighbours.append(c2)
            tiles.append(tile_index)

    indices_to_check = []

    # NOTE FOR OPTIMISATION: cache values within range of the boundary to a buffer for minimising subsequent searches.
    for i, window in enumerate(neighbouring_windows):
        win = np.array([window[0], window[2], window[1], window[3]])
        if are_neighbours(c1, win):
            indices_to_check.append(neighbouring_indices[i])

    for (win_i, win_j) in indices_to_check:
        num_searched_windows += 1
        for c2 in image[win_i][win_j]:
            if are_neighbours(c1, c2):
                neighbours.append(c2)
                tiles.append((win_i, win_j))

    return zip(neighbours, tiles), num_searched_windows


def in_any_previous_cluster(image_clusters, previous_indices, cell):
    for (i, j) in previous_indices:
        if already_in_cluster(image_clusters[i][j], cell):
            return True
    return False


def get_neighbouring_windows(i, j, n, windows):
    neighbouring_indices = []
    previous_indices = []
    neighbouring_windows = []

    for range_i in range(i-1, i+2):
        for range_j in range(j-1, j+2):
            if not(range_i == i and range_j == j):
                if(range_i >= 0 and range_i < n and range_j >= 0 and range_j < n):
                    neighbouring_indices.append((range_i, range_j))
                    if range_i == i-1:
                        previous_indices.append((range_i, range_j))
                    elif range_i == i and range_j == j-1:
                        previous_indices.append((range_i, range_j))

    previous_indices.append((i, j))

    for (win_i, win_j) in neighbouring_indices:
        neighbouring_windows.append(windows[win_i][win_j])

    return neighbouring_windows, neighbouring_indices, previous_indices


def window_cleaning_algorithm(image, n, windows):
    """
    Window-Cleaning algorithm on images to extract clusters.
    For every tile, for every cell, get all neighbours and assign into a cluster.
    Resolve overlapping clusters (either within same tile or over multiple tiles) by having the smaller one subsumed by the larger.
    Remove current cell at the end to reduce search for future cells.
    """
    image_clusters = np.empty((n, n), dtype=object)
    num_searched_windows = 0

    for i in range(n):
        for j in range(n):
            image_clusters[i][j] = {}

    for i in range(n):
        for j in range(n):
            current_tile = image[i][j]
            reversed_tile = current_tile[::-1]
            neighbouring_windows, neighbouring_indices, previous_indices = get_neighbouring_windows(i, j, n, windows)
            for index, current_cell in enumerate(reversed_tile):
                if not already_in_cluster((i, j), current_cell):  # if not in_any_previous_cluster(image_clusters, previous_indices, current_cell):
                    create_cluster(current_cell, image_clusters[i][j], (i, j))
                neighbours, num_searched_windows = get_all_neighbours(current_cell, image, index, (i, j), neighbouring_windows, neighbouring_indices, num_searched_windows)
                for neighbour, (n_i, n_j) in neighbours:
                    if already_in_cluster((n_i, n_j), neighbour):
                        if not in_same_cluster(current_cell, neighbour, image_clusters, (i, j), (n_i, n_j)):
                            current_cell_cluster = get_cluster(image_clusters[i][j], current_cell)
                            neighbour_cluster = get_cluster(image_clusters[n_i][n_j], neighbour)
                            subsume(current_cell_cluster, neighbour_cluster, image_clusters)
                    else:
                        current_cell_cluster = get_cluster(image_clusters[i][j], current_cell)
                        add_to_current_cluster(current_cell_cluster, image_clusters, neighbour, (n_i, n_j))

    return image_clusters, num_searched_windows


def fishermans_algorithm(image, n, windows):
    """
    Fisherman's algorithm on images to extract clusters.
    For every tile, for every available cell, get all neighbours and assign into a cluster.
    From each neighbour, recursively get all neighbours and assign into same cluster.
    Remove each cell along the way to reduce search for future cells.
    """
    image_clusters = np.empty((n, n), dtype=object)
    # image = image.tolist()

    for i in range(n):
        for j in range(n):
            image_clusters[i][j] = {}

    for i in tqdm(range(n)):
        for j in range(n):
            while image[i][j]:
                cell = image[i][j].pop()
                cluster = create_and_return_cluster(cell, image_clusters[i][j], (i, j))
                clusterise(cell, cluster, (i, j), image, n, windows, image_clusters)

    return image_clusters


def get_and_remove_all_neighbours(c1, image, neighbouring_windows, neighbouring_indices):
    neighbours, tiles = [], []
    indices_to_check = []

    for i, window in enumerate(neighbouring_windows):
        # win = np.array([window[0], window[2], window[1], window[3]])
        # if are_neighbours(c1, win):
        indices_to_check.append(neighbouring_indices[i])

    for (win_i, win_j) in indices_to_check:
        base = len(image[win_i][win_j]) - 1
        for i, c2 in enumerate(reversed(image[win_i][win_j])):
            if are_neighbours(c1, c2):
                neighbours.append(c2)
                image[win_i][win_j].pop(base - i)
                tiles.append((win_i, win_j))

    return zip(neighbours, tiles)


def get_neighbouring_windows_fisherman(i, j, n, windows):
    neighbouring_indices = []
    neighbouring_windows = []

    for range_i in range(i-1, i+2):
        for range_j in range(j-1, j+2):
            if(range_i >= 0 and range_i < n and range_j >= 0 and range_j < n):
                neighbouring_indices.append((range_i, range_j))

    for (win_i, win_j) in neighbouring_indices:
        neighbouring_windows.append(windows[win_i][win_j])

    return neighbouring_windows, neighbouring_indices


def clusterise(cell, cluster, tile_index, image, n, windows, image_clusters):
    stack = deque([(cell, tile_index)])
    while stack:
        cell, tile_index = stack.pop()
        neighbouring_windows, neighbouring_indices = get_neighbouring_windows_fisherman(tile_index[0], tile_index[1], n, windows)
        neighbours = get_and_remove_all_neighbours(cell, image, neighbouring_windows, neighbouring_indices)

        if neighbours:
            for neighbour, (n_i, n_j) in neighbours:
                add_to_current_cluster(cluster, image_clusters, neighbour, (n_j, n_j))
                stack.append((neighbour, (n_i, n_j)))

    # NB: hard-remove is required, which is to actually delete the cell, instead of soft-remove, which is merely skip by index

    # Think tiles have to be lists of cells, as Vaiva did it
    # You have an advantage though because you can calculate whether a cell is close enough to a tile before checking it. Vaiva does not do this,
    # so you search a lot less already because of this. She has to check all 9 tiles - you search at most the number of tiles that you are close
    # enough to that you could be a neighbour (which is at most 4 if you include current tile)

    # for each tile in image:
    #     for each non-empty cell in tile:
    #         hard-remove cell from tile
    #         assign cell to cluster in its tile
    #         clusterise(cell, cluster)

    # def clusterise(cell, cluster):
    #     get all neighbours (current tile or otherwise) of cell, hard-remove neighbours from their tiles while doing so (use windows here)
    #     if neighbours is not empty:
    #         for each neighbour:
    #             assign neighbour to cluster
    #             clusterise(neighbour, cluster)





# End of file
