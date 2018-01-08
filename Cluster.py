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
        """
        This function surrenders its data - used when this cluster is subsumed
        by another.
        """
        temp_cells = self.cells[:]
        temp_tiles = self.tiles[:]
        self.cells = None
        self.tiles = None
        return temp_cells, temp_tiles


def get_key(cell):
    return (cell[0], cell[1], cell[2], cell[3])


def get_cluster(tile, cell):
    key = get_key(cell)
    return tile[key]


def create_cluster(cell, tile, index):
    key = get_key(cell)
    tile[key] = Cluster(cell, index)


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

    xMaxOk = (xMin1 - width) < xMin2
    xMinOk = xMax2 < (xMax1 + width)
    yMaxOk = (yMin1 - height) < yMin2
    yMinOk = yMax2 < (yMax1 + height)

    return xMaxOk and xMinOk and yMaxOk and yMinOk


def get_all_neighbours(c1, image, cell_index, image_index, neighbouring_windows, neighbouring_indices):
    neighbours = []
    tiles = []
    img_i, img_j = image_index[0], image_index[1]
    for i, c2 in enumerate(image[img_i][img_j][0:-cell_index-1]):
        if are_neighbours(c1, c2):
            neighbours.append(c2)
            tiles.append(image_index)

    indices_to_check = []

    # NOTE FOR OPTIMISATION: cache values within range of the boundary to a buffer for minimising subsequent searches.
    xMin1, xMax1, yMin1, yMax1 = c1[0], c1[1], c1[2], c1[3]
    for i, window in enumerate(neighbouring_windows):
        win_xMin, win_yMin, win_xMax, win_yMax = window[0], window[1], window[2], window[3]

        width = (xMax1 - xMin1) + d
        height = (yMax1 - yMin1) + d

        within_xMin = (xMin1 - width) < win_xMin
        within_yMin = (yMin1 - height) < win_yMin
        within_xMax = (xMax1 + width) >= win_xMax
        within_yMax = (yMax1 + height) >= win_yMax

        if within_xMin or within_yMin or within_xMax or within_yMax:
            indices_to_check.append(neighbouring_indices[i])

    for (win_i, win_j) in indices_to_check:
        for c2 in image[win_i][win_j]:
            if are_neighbours(c1, c2):
                neighbours.append(c2)
                tiles.append((win_i, win_j))

    # return tuples of (neighbour, neighbour tile)
    return zip(neighbours, tiles)


def in_any_previous_cluster(image_clusters, previous_indices, cell):
    for (i, j) in previous_indices:
        if already_in_cluster(image_clusters[i][j], cell):
            return True
    return False


def in_same_cluster(cell1, cell2, image_clusters, index1, index2):
    (i, j) = index1
    (n_i, n_j) = index2
    tile1 = image_clusters[i][j]
    tile2 = image_clusters[n_i][n_j]
    cluster1 = get_cluster(tile1, cell1)
    cluster2 = get_cluster(tile2, cell2)
    return cluster1 == cluster2


def add_to_current_cluster(tile_cluster, current_cell, neighbour, neighbour_tile_index):
    neighbour_key = get_key(neighbour)
    cluster = get_cluster(tile_cluster, current_cell)
    tile_cluster[neighbour_key] = cluster
    cluster.add_cell(neighbour, neighbour_tile_index)


def get_neighbouring_windows(i, j, n, windows):
    neighbouring_indices = []
    neighbouring_windows = []

    for range_i in range(i-1, i+1):
        for range_j in range(j-1, j+1):
            if not(range_i == i and range_j == j):
                if(range_i >= 0 and range_i < n and range_j >= 0 and range_j < n):
                    neighbouring_indices.append((range_i, range_j))

    for (win_i, win_j) in neighbouring_indices:
        neighbouring_windows.append(windows[win_i][win_j])

    return neighbouring_windows, neighbouring_indices


def window_cleaning_algorithm(image, n, windows, load_bar=False):
    """Testing clustering algorithm on toy images"""
    image_clusters = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            image_clusters[i][j] = {}  # Not ideal to have to dynamically resize, but you'd have to do it regardless, so...

    if load_bar:
        for i in tqdm(range(n)):
            for j in range(n):
                current_tile = image[i][j]
                reversed_tile = current_tile[::-1]
                neighbouring_windows, neighbouring_indices = get_neighbouring_windows(i, j, n, windows)
                previous_indices = neighbouring_indices[0:4]
                previous_indices.append((i, j))
                for index, current_cell in enumerate(reversed_tile):
                    if not in_any_previous_cluster(image_clusters, previous_indices, current_cell):
                        create_cluster(current_cell, image_clusters[i][j], (i, j))
                    neighbours = get_all_neighbours(current_cell, image, index, (i, j), neighbouring_windows, neighbouring_indices)
                    for neighbour, (n_i, n_j) in neighbours:
                        if not in_any_previous_cluster(image_clusters, previous_indices, neighbour):
                            add_to_current_cluster(image_clusters[i][j], current_cell, neighbour, (n_i, n_j))
                        elif not in_same_cluster(current_cell, neighbour, image_clusters, (i, j), (n_i, n_j)):
                            current_cell_cluster = get_cluster(image_clusters[i][j], current_cell)
                            neighbour_cluster = get_cluster(image_clusters[n_i][n_j], neighbour)
                            subsume(current_cell_cluster, neighbour_cluster, image_clusters)
    else:
        for i in range(n):
            for j in range(n):
                current_tile = image[i][j]
                reversed_tile = current_tile[::-1]
                neighbouring_windows, neighbouring_indices = get_neighbouring_windows(i, j, n, windows)
                previous_indices = neighbouring_indices[0:4]
                previous_indices.append((i, j))
                for index, current_cell in enumerate(reversed_tile):
                    if not in_any_previous_cluster(image_clusters, previous_indices, current_cell):
                        create_cluster(current_cell, image_clusters[i][j], (i, j))
                    neighbours = get_all_neighbours(current_cell, image, index, (i, j), neighbouring_windows, neighbouring_indices)
                    for neighbour, (n_i, n_j) in neighbours:
                        if not in_any_previous_cluster(image_clusters, previous_indices, neighbour):
                            add_to_current_cluster(image_clusters[i][j], current_cell, neighbour, (n_i, n_j))
                        elif not in_same_cluster(current_cell, neighbour, image_clusters, (i, j), (n_i, n_j)):
                            current_cell_cluster = get_cluster(image_clusters[i][j], current_cell)
                            neighbour_cluster = get_cluster(image_clusters[n_i][n_j], neighbour)
                            subsume(current_cell_cluster, neighbour_cluster, image_clusters)
    return image_clusters

    # for index, current_cell in reversed_tile:
    #     if not already_in_cluster(current_cell):  # This assumes in current tile or any previous neighbouring tile
    #         create new cluster with current_cell in it in current tile
    #     neighbours is all cells in *all* neighbouring tiles within distance d of current_cell, accessed from 0 to index in current tile, else all
    #     for all neighbours (within distance d of current_cell and not current_cell itself):
    #         if not_already_in_cluster(neighbour):  # This assumes in current tile or any previous neighbouring tile
    #             add to current_cell's cluster  # In current tile
    #         else if not_same_cluster(current_cell, neighbour):  # This assumes either in current_cell's cluster or neighbour's cluster
    #             if subsume(current_cell's cluster versus neighbour's cluster) == True:
    #                 Delete neighbour's cluster
    #             else:
    #                 Delete current_cell's cluster
