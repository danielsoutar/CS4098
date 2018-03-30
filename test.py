import Cluster
import partition
import unittest
import numpy as np

# Test cases for fishermans clustering algorithm - these assumed a distance of 7um.
Cluster.set_d(7)


def initialise_clusters(n):
    image_clusters = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            image_clusters[i][j] = {}

    return image_clusters


def construct_input_neighbour_tile_checking():
    t = 3

    xMin = np.array([0, 30, 60, 0,  30, 60, 0,  30, 60])
    xMax = np.array([3, 33, 63, 3,  33, 63, 3,  33, 63])
    yMin = np.array([0, 0,  0,  30, 30, 30, 60, 60, 60])
    yMax = np.array([3, 3,  3,  33, 33, 33, 63, 63, 63])

    points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
    partitioned_points, tiles, _, _ = partition.partition(points, tile_size=t, to_list=True)

    return t, partitioned_points, tiles


def construct_input_get_and_remove_all_neighbours_checking(extension=None):
    t = 3

    if extension is None:
        xMin = np.array([0, 30, 60, 0,  30, 60, 0,  30, 60])
        xMax = np.array([3, 33, 63, 3,  33, 63, 3,  33, 63])
        yMin = np.array([0, 0,  0,  30, 30, 30, 60, 60, 60])
        yMax = np.array([3, 3,  3,  33, 33, 33, 63, 63, 63])
    elif extension is "one_neighbour_same_tile":
        xMin = np.array([0, 3, 30, 33, 60, 63, 0, 3, 30, 33, 60, 63, 0, 3, 30, 33, 60, 63])
        xMax = np.array([3, 6, 33, 36, 63, 66, 3, 6, 33, 36, 63, 66, 3, 6, 33, 36, 63, 66])
        yMin = np.array([0, 3, 0, 3, 0, 3, 30, 33, 30, 33, 30, 33, 60, 63, 60, 63, 60, 63])
        yMax = np.array([3, 6, 3, 6, 3, 6, 33, 36, 33, 36, 33, 36, 63, 66, 63, 66, 63, 66])
    elif extension is "one_neighbour_different_tile":
        xMin = np.array([17, 25, 60, 0,  30, 60, 0,  30, 60])
        xMax = np.array([21, 28, 63, 3,  33, 63, 3,  33, 63])
        yMin = np.array([8,  8,  0,  30, 30, 30, 60, 60, 60])
        yMax = np.array([12, 12,  3,  33, 33, 33, 63, 63, 63])
    elif extension is "one_neighbour_different_tile_overlap":
        xMin = np.array([15, 31, 60, 0,  30, 60, 0,  30, 60])
        xMax = np.array([26, 33, 63, 3,  33, 63, 3,  33, 63])
        yMin = np.array([8,  8,  0,  30, 30, 30, 60, 60, 60])
        yMax = np.array([12, 12, 3,  33, 33, 33, 63, 63, 63])

        points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
        partitioned_points, tiles, w, h = partition.partition(points, tile_size=t, to_list=True)

        return t, partitioned_points, tiles, w, h

    elif extension is "many_neighbours_same_tile":
        xMin = np.array([0, 0, 3, 3, 30, 30, 33, 33, 60, 60, 63, 63, 0, 0, 3, 3, 30, 30, 33, 33, 60, 60,
                         63, 63, 0, 0, 3, 3, 30, 30, 33, 33, 60, 60, 63, 63])
        xMax = np.array([3, 3, 6, 6, 33, 33, 36, 36, 63, 63, 66, 66, 3, 3, 6, 6, 33, 33, 36, 36, 63, 63,
                         66, 66, 3, 3, 6, 6, 33, 33, 36, 36, 63, 63, 66, 66])
        yMin = np.array([0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 30, 33, 33, 30, 30, 33, 33, 30, 30, 33, 33,
                         30, 60, 63, 63, 60, 60, 63, 63, 60, 60, 63, 63, 60])
        yMax = np.array([3, 6, 6, 3, 3, 6, 6, 3, 3, 6, 6, 3, 33, 36, 36, 33, 33, 36, 36, 33, 33, 36, 36,
                         33, 63, 66, 66, 63, 63, 66, 66, 63, 63, 66, 66, 63])
    elif extension is "many_neighbours_many_tiles":
        xMin = np.array([40, 45, 40, 45, 0, 30, 60, 0,  60, 0,  30, 60])
        xMax = np.array([43, 48, 43, 48, 3, 33, 63, 3,  63, 3,  33, 63])
        yMin = np.array([18, 18, 23, 23, 0, 0,  0,  30, 30, 60, 60, 60])
        yMax = np.array([21, 21, 26, 26, 3, 3,  3,  33, 33, 63, 63, 63])
    elif extension is "many_clusters_many_neighbours_none_from_same_cluster":
        xMin = np.array([40, 45, 18, 23, 18, 18, 18, 23, 40, 45, 45, 40, 40, 45, 23, 0, 30, 60, 0,  23, 60, 0,  30, 60])
        xMax = np.array([43, 48, 21, 26, 21, 21, 21, 26, 43, 48, 48, 43, 43, 48, 26, 3, 33, 63, 3,  26, 63, 3,  33, 63])
        yMin = np.array([18, 18, 23, 18, 18, 40, 45, 45, 40, 45, 40, 45, 23, 23, 40, 0, 0,  0,  30, 23, 30, 60, 60, 60])
        yMax = np.array([21, 21, 26, 21, 21, 43, 48, 48, 43, 48, 43, 48, 26, 26, 43, 3, 3,  3,  33, 26, 33, 63, 63, 63])

    points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
    partitioned_points, tiles, _, _ = partition.partition(points, tile_size=t, to_list=True)

    return t, partitioned_points, tiles


def construct_input_extend_current_cluster_checking(extension=None):
    t = 3

    if extension is "no_neighbours":
        xMin = np.array([0, 30, 60, 0,  30, 60, 0,  30, 60])
        xMax = np.array([3, 33, 63, 3,  33, 63, 3,  33, 63])
        yMin = np.array([0, 0,  0,  30, 30, 30, 60, 60, 60])
        yMax = np.array([3, 3,  3,  33, 33, 33, 63, 63, 63])
    elif extension is "one_neighbour":
        xMin = np.array([0, 7,  30, 60, 0,  30, 60, 0,  30, 60])
        xMax = np.array([3, 10, 33, 63, 3,  33, 63, 3,  33, 63])
        yMin = np.array([0, 7,  0,  0,  30, 30, 30, 60, 60, 60])
        yMax = np.array([3, 10, 3,  3,  33, 33, 33, 63, 63, 63])
    elif extension is "many_neighbours":
        xMin = np.array([40, 45, 40, 45, 0, 30, 60, 0,  60, 0,  30, 60])
        xMax = np.array([43, 48, 43, 48, 3, 33, 63, 3,  63, 3,  33, 63])
        yMin = np.array([18, 18, 23, 23, 0, 0,  0,  30, 30, 60, 60, 60])
        yMax = np.array([21, 21, 26, 26, 3, 3,  3,  33, 33, 63, 63, 63])

    points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
    partitioned_points, tiles, _, _ = partition.partition(points, tile_size=t, to_list=True)

    return t, partitioned_points, tiles


def construct_input_clusterise_checking(extension=None, set_t=False, updated_t=3):
    t = 3

    if set_t:
        t = updated_t

    if extension is "one_cell_one_cluster":
        xMin = np.array([0])
        xMax = np.array([3])
        yMin = np.array([0])
        yMax = np.array([3])
    elif extension is "many_cells_one_cluster_one_tile":
        xMin = np.array([0, 4, 0, 4, 12, 30, 60, 0,  30, 60, 0,  30, 60])
        xMax = np.array([3, 7, 3, 7, 15, 33, 63, 3,  33, 63, 3,  33, 63])
        yMin = np.array([0, 4, 4, 0, 12, 0,  0,  30, 30, 30, 60, 60, 60])
        yMax = np.array([3, 7, 7, 3, 15, 3,  3,  33, 33, 33, 63, 63, 63])
    elif extension is "many_cells_one_cluster_many_tiles":
        xMin = np.arange(1000)
        xMax = np.copy(xMin) + 3
        yMin = np.arange(1000)
        yMax = np.copy(xMin) + 3
    elif extension is "many_cells_many_clusters_many_tiles":
        xMin = np.array([0, 0, 3, 3, 30, 30, 33, 33, 60, 60, 63, 63, 0, 0, 3, 3, 30, 30, 33, 33, 60, 60,
                         63, 63, 0, 0, 3, 3, 30, 30, 33, 33, 60, 60, 63, 63])
        xMax = np.array([3, 3, 6, 6, 33, 33, 36, 36, 63, 63, 66, 66, 3, 3, 6, 6, 33, 33, 36, 36, 63, 63,
                         66, 66, 3, 3, 6, 6, 33, 33, 36, 36, 63, 63, 66, 66])
        yMin = np.array([0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 30, 33, 33, 30, 30, 33, 33, 30, 30, 33, 33,
                         30, 60, 63, 63, 60, 60, 63, 63, 60, 60, 63, 63, 60])
        yMax = np.array([3, 6, 6, 3, 3, 6, 6, 3, 3, 6, 6, 3, 33, 36, 36, 33, 33, 36, 36, 33, 33, 36, 36,
                         33, 63, 66, 66, 63, 63, 66, 66, 63, 63, 66, 66, 63])

    points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
    partitioned_points, tiles, _, _ = partition.partition(points, tile_size=t, to_list=True)

    return t, partitioned_points, tiles


def construct_input_fishermans_checking(extension=None, set_t=False, updated_t=3, NUM_POINTS=0):
    t = 3

    if set_t:
        t = updated_t

    if extension is "many_cells_many_clusters_many_tiles":
        xMin = np.array([0, 0, 3, 3, 30, 30, 33, 33, 60, 60, 63, 63, 0, 0, 3, 3, 30, 30, 33, 33, 60, 60,
                         63, 63, 0, 0, 3, 3, 30, 30, 33, 33, 60, 60, 63, 63])
        xMax = np.array([3, 3, 6, 6, 33, 33, 36, 36, 63, 63, 66, 66, 3, 3, 6, 6, 33, 33, 36, 36, 63, 63,
                         66, 66, 3, 3, 6, 6, 33, 33, 36, 36, 63, 63, 66, 66])
        yMin = np.array([0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 30, 33, 33, 30, 30, 33, 33, 30, 30, 33, 33,
                         30, 60, 63, 63, 60, 60, 63, 63, 60, 60, 63, 63, 60])
        yMax = np.array([3, 6, 6, 3, 3, 6, 6, 3, 3, 6, 6, 3, 33, 36, 36, 33, 33, 36, 36, 33, 33, 36, 36,
                         33, 63, 66, 66, 63, 63, 66, 66, 63, 63, 66, 66, 63])
    elif extension is "many_cells_many_clusters_many_tiles_overlapping":
        xMin = np.array([40, 45, 18, 23, 18, 18, 18, 23, 40, 45, 45, 40, 40, 45, 23, 0, 30, 60, 0,  23, 60, 0,  30, 60])
        xMax = np.array([43, 48, 21, 26, 21, 21, 21, 26, 43, 48, 48, 43, 43, 48, 26, 3, 33, 63, 3,  26, 63, 3,  33, 63])
        yMin = np.array([18, 18, 23, 18, 18, 40, 45, 45, 40, 45, 40, 45, 23, 23, 40, 0, 0,  0,  30, 23, 30, 60, 60, 60])
        yMax = np.array([21, 21, 26, 21, 21, 43, 48, 48, 43, 48, 43, 48, 26, 26, 43, 3, 3,  3,  33, 26, 33, 63, 63, 63])
    elif extension is "random":
        xMin = np.random.choice(range(NUM_POINTS + 1), size=NUM_POINTS, replace=False)
        xMax = np.copy(xMin) + 3
        yMin = np.random.choice(range(NUM_POINTS + 1), size=NUM_POINTS, replace=False)
        yMax = np.copy(yMin) + 3

    points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
    partitioned_points, tiles, w, h = partition.partition(points, tile_size=t, to_list=True)

    if extension is "random":
        return t, partitioned_points, tiles, w, h, points
    else:
        return t, partitioned_points, tiles, w, h


class myTest(unittest.TestCase):

    # Only concerning ourselves with fishermans in this module, bar comparison with another algorithm for random inputs.

    def test_creating_cluster(self):
        cell = np.array([0, 3, 0, 3])
        image_clusters = []
        Cluster.pure_create_cluster(cell, image_clusters)
        self.assertTrue(image_clusters[0].cells[0] is cell)

    def test_creating_clusters(self):
        a = np.array([0, 3, 0, 3])
        b = np.array([5, 8, 5, 8])
        image_clusters = []
        Cluster.pure_create_cluster(a, image_clusters)
        self.assertTrue(image_clusters[-1].cells[0] is a)
        Cluster.pure_create_cluster(b, image_clusters)
        self.assertTrue(image_clusters[0].cells[0] is a)
        self.assertTrue(image_clusters[1].cells[0] is b)

    def test_get_cluster(self):
        a = np.array([0, 3, 0, 3])
        b = np.array([5, 8, 5, 8])
        image_clusters = []
        Cluster.pure_create_cluster(a, image_clusters)
        Cluster.pure_create_cluster(b, image_clusters)
        c = image_clusters[-1]
        self.assertTrue(Cluster.pure_get_cluster(image_clusters) is c)

    def test_get_neighbouring_tiles_corner(self):
        """
        Assuming a 3x3 grid, each cell to a single tile. Couple of extras to test overlapping later.
        """
        t, partitioned_points, tiles = construct_input_neighbour_tile_checking()

        # Note we are at (0, 0), so we expect to see 4 neighbours (+1 because I am including the current tile)
        neighbouring_indices, _ = Cluster.get_neighbouring_windows_fisherman(0, 0, t, t, tiles)
        self.assertTrue(len(neighbouring_indices) == 4)

        # Now let's check for the other 3 corners
        for tile in [(0, t-1), (t-1, 0), (t-1, t-1)]:
            neighbouring_indices, _ = Cluster.get_neighbouring_windows_fisherman(tile[0], tile[1], t, t, tiles)
            self.assertTrue(len(neighbouring_indices) == 4)

    def test_get_neighbouring_windows_side(self):
        """
        As above, but now for all windows along the sides. We expect 6 neighbours (+1 for current tile).
        """
        t, partitioned_points, tiles = construct_input_neighbour_tile_checking()

        # Note we are at (0, 1), so we expect to see 6 neighbours
        neighbouring_indices, _ = Cluster.get_neighbouring_windows_fisherman(0, 1, t, t, tiles)
        self.assertTrue(len(neighbouring_indices) == 6)

        # Now let's check for the other 3 sides
        for tile in [(1, 0), (2, 1), (1, 2)]:
            neighbouring_indices, _ = Cluster.get_neighbouring_windows_fisherman(tile[0], tile[1], t, t, tiles)
            self.assertTrue(len(neighbouring_indices) == 6)

    def test_get_neighbouring_windows_central(self):
        """
        Easy enough for the centre - we only have 1 tile of this kind in a 3x3!
        """
        t, partitioned_points, tiles = construct_input_neighbour_tile_checking()

        # Note we are at (1, 1), so we expect to see 9 neighbours
        neighbouring_indices, _ = Cluster.get_neighbouring_windows_fisherman(1, 1, t, t, tiles)
        self.assertTrue(len(neighbouring_indices) == 9)

    def test_correct_number_of_total_neighbouring_windows(self):
        """
        This is where I aggregate the total number of neighbouring windows for each window and compare that against the expected value.
        The mathematical formula for an (n x n) grid is:

        total = 4((n - (n - 1)) * 4) + 4((n-2) * 6) + (n-2 * n-2 * 9),

        which accounts for the 4 corners, the sides, and the 'body' of the input respectively. Some examples:

        if n = 3:  then total = 4(4) + 4(1 * 6) + 1*1*9 = 49
        if n = 9:  then total = 4(4) + 4(7 * 6) + 7*7*9 = 625
        if n = 20: then total = 4(4) + 4(18 * 6) + 18*18*9 = 3364
        if n = 25: then total = 4(4) + 4(23 * 6) + 23*23*9 = 5329
        """

        ts = [3, 9, 20, 25]  # Testing several values

        expecteds = [49, 625, 3364, 5329]

        for t, expected in zip(ts, expecteds):
            # Just creating a dummy tiles object to pass into the function.
            tiles = np.zeros((t, t, 4))
            count = 0
            for i in range(t):
                for j in range(t):
                    neighbouring_indices, _ = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    count += len(neighbouring_indices)

            self.assertTrue(count == expected)

    def test_get_and_remove_all_neighbours_not_itself(self):
        """
        We are mimicking the logic of the algorithm - first we remove the cell from the tile, so it couldn't possibly find itself in there again.
        """
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking()

        ### This is in the main body of the algorithm
        for i in range(t):
            for j in range(t):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    self.assertTrue(cell not in neighbours)

    def test_get_and_remove_all_neighbours_no_neighbours_alone(self):
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking()

        ### This is in the main body of the algorithm
        for i in range(t):
            for j in range(t):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    self.assertTrue(len(neighbours) == 0)

    def test_get_and_remove_all_neighbours_one_neighbour_same_tile(self):
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking("one_neighbour_same_tile")

        ### This is in the main body of the algorithm
        for i in range(t):
            for j in range(t):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    self.assertTrue(len(neighbours) == 1)

    def test_get_and_remove_all_neighbours_one_neighbour_different_tile_no_overlap(self):
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking("one_neighbour_different_tile")

        origin_i, origin_j = 0, 0
        target_i, target_j = 1, 0

        ### This is in the main body of the algorithm
        for i in range(t):
            for j in range(t):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    if i == origin_i and j == origin_j:
                        self.assertTrue(len(neighbours) == 1)
                        self.assertTrue(len(partitioned_points[target_i][target_j]) == 0)  # Removed the cell from that tile

    def test_get_and_remove_all_neighbours_one_neighbour_different_tile_overlap(self):
        t, partitioned_points, tiles, _, _ = construct_input_get_and_remove_all_neighbours_checking("one_neighbour_different_tile_overlap")

        origin_i, origin_j = 0, 0
        target_i, target_j = 1, 0

        ### This is in the main body of the algorithm
        for i in range(t):
            for j in range(t):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    if i == origin_i and j == origin_j:
                        self.assertTrue(len(neighbours) == 1)
                        self.assertTrue(len(partitioned_points[target_i][target_j]) == 0)  # Removed the cell from that tile

        # But what about an edge case? Say a cell c in (0, 0) was too far away from the border according to our original comparison function to check it.
        # But now assume a cell c' in (1, 0) that overlaps (0, 0) that would otherwise have been considered a neighbour of c.
        # Hence the order in which we evaluate them matters, which means we could potentially miss a cluster in other similar scenarios.

        # ... That's a problem. Unfortunately, the solution is to extend the range of search for each cell against its windows to account for
        # overlapping cells. I say unfortunately, because in addition to having to check the window, we also introduce some wasted search.

        # Let's first prove that this is a problem. We start from the end of the input and show that in the target tile we have no neighbours when
        # we should have one, the one from the original tile.

        a = np.array([15, 26, 8, 12])
        b = np.array([31, 33, 8, 12])
        self.assertTrue(Cluster.are_neighbours(a, b))  # They are neighbours.

        t, partitioned_points, tiles, _, _ = construct_input_get_and_remove_all_neighbours_checking("one_neighbour_different_tile_overlap")

        for i in reversed(range(t)):
            for j in reversed(range(t)):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    if i == target_i and j == target_j:
                        self.assertTrue(len(neighbours) == 0)
                        self.assertTrue(len(partitioned_points[origin_i][origin_j]) == 1)  # Neither noticed nor removed.

        # We fix the above by incoporating a more generous check for cells against neighbouring windows - we add on the dimensions of the largest cell we
        # found in the input to accomodate the worst possible case. We'll add these dimensions here. Note that we will set the dimensions for each input.

        t, partitioned_points, tiles, w, h = construct_input_get_and_remove_all_neighbours_checking("one_neighbour_different_tile_overlap")

        Cluster.set_extensions(w, h)

        for i in reversed(range(t)):
            for j in reversed(range(t)):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    if i == target_i and j == target_j:
                        self.assertTrue(len(neighbours) == 1)
                        self.assertTrue(len(partitioned_points[origin_i][origin_j]) == 0)  # Now successfully removed.

        Cluster.set_extensions(0, 0)  # Clear it so we don't accidentally muddle up results elsewhere in the test cases. Shouldn't affect anything though.

    def test_get_and_remove_all_neighbours_many_neighbours_same_tile(self):
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking("many_neighbours_same_tile")

        ### This is in the main body of the algorithm
        for i in range(t):
            for j in range(t):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    ### ...
                    ### This would be in the clusterise() function.
                    neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(i, j, t, t, tiles)
                    neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                    self.assertTrue(len(neighbours) == 3)
                    self.assertTrue(len(partitioned_points[i][j]) == 0)  # Removed all cells from that tile for this particular input

    def test_get_and_remove_all_neighbours_many_neighbours_multiple_tiles(self):
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking("many_neighbours_many_tiles")

        ### This is in the main body of the algorithm (for just the central tile)
        while partitioned_points[1][1]:
            cell = partitioned_points[1][1].pop()
            ### ...
            ### This would be in the clusterise() function.
            neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(1, 1, t, t, tiles)
            neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
            self.assertTrue(len(neighbours) == 3)
            self.assertTrue(len(partitioned_points[1][1]) == 0)  # Removed cell from current tile
            self.assertTrue(len(partitioned_points[0][1]) == 1 and
                            len(partitioned_points[0][2]) == 1 and
                            len(partitioned_points[1][2]) == 1)  # Removed cells from their tiles

    def test_get_and_remove_all_neighbours_none_from_same_cluster(self):
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking("many_clusters_many_neighbours_none_from_same_cluster")

        cell_to_check = np.array([23, 26, 23, 26])  # Last cell belonging to the central tile in the input above.

        ## This is in the main body of the algorithm (for just the central tile, for cell_to_check)
        while partitioned_points[1][1]:
            cell = partitioned_points[1][1].pop()
            if np.array_equal(cell, cell_to_check):
                ### ...
                ### This would be in the clusterise() function.
                neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(1, 1, t, t, tiles)
                neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
                self.assertTrue(len(neighbours) == 3)
                self.assertTrue(len(partitioned_points[1][1]) == 3)  # Still 3 other cells from the central tile
                self.assertTrue(len(partitioned_points[0][1]) == 2 and  # Other cells still need to be evaluated!
                                len(partitioned_points[1][0]) == 2 and
                                len(partitioned_points[0][0]) == 1)  # Removed neighbours from their tiles

    def test_get_and_remove_all_neighbours_none_already_searched(self):
        t, partitioned_points, tiles = construct_input_get_and_remove_all_neighbours_checking("many_clusters_many_neighbours_none_from_same_cluster")

        cell_to_check = np.array([40, 43, 40, 43])  # First cell belonging to the central tile in the input above.

        ## This is in the main body of the algorithm (for just the central tile, for cell_to_check)
        while partitioned_points[1][1]:
            cell = partitioned_points[1][1].pop()
            ### ...
            ### This would be in the clusterise() function.
            neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(1, 1, t, t, tiles)
            neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)
            if np.array_equal(cell, cell_to_check):
                self.assertTrue(len(neighbours) == 3)
                self.assertTrue(len(partitioned_points[1][1]) == 0)  # We already removed the other cells, this is the last one we evaluate in the central tile.
                self.assertTrue(len(partitioned_points[2][1]) == 1 and
                                len(partitioned_points[1][2]) == 1 and
                                len(partitioned_points[2][2]) == 1)  # Removed neighbours from their tiles

    def test_extend_current_cluster_no_neighbours(self):
        t, partitioned_points, tiles = construct_input_extend_current_cluster_checking("no_neighbours")

        cell_to_check = np.array([0, 3, 0, 3])
        image_clusters = []

        ## This is in the main body of the algorithm (for just the first tile, for cell_to_check)
        while partitioned_points[0][0]:
            cell = partitioned_points[0][0].pop()
            if np.array_equal(cell, cell_to_check):
                Cluster.pure_create_cluster(cell, image_clusters)
                ### ...
                ### This would be in the clusterise() function.
                cluster = Cluster.pure_get_cluster(image_clusters)
                initial_length = len(cluster.cells)

                neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(0, 0, t, t, tiles)
                neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)

                self.assertTrue(len(neighbours) == 0)
                self.assertTrue(len(partitioned_points[0][0]) == 0)  # It was the only cell in its tile
                Cluster.extend_current_cluster(cluster, neighbours)
                final_length = len(cluster.cells)
                self.assertTrue(initial_length == final_length and initial_length == 1)

    def test_extend_current_cluster_one_neighbour(self):
        t, partitioned_points, tiles = construct_input_extend_current_cluster_checking("one_neighbour")

        cell_to_check = np.array([7, 10, 7, 10])
        image_clusters = []

        ## This is in the main body of the algorithm (for just the first tile, for cell_to_check)
        while partitioned_points[0][0]:
            cell = partitioned_points[0][0].pop()
            if np.array_equal(cell, cell_to_check):
                Cluster.pure_create_cluster(cell, image_clusters)
                ### ...
                ### This would be in the clusterise() function.
                cluster = Cluster.pure_get_cluster(image_clusters)
                initial_length = len(cluster.cells)

                neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(0, 0, t, t, tiles)
                neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)

                self.assertTrue(len(neighbours) == 1)
                self.assertTrue(len(partitioned_points[0][0]) == 0)  # All cells in this tile of the same cluster, so all removed in one go
                Cluster.extend_current_cluster(cluster, neighbours)
                final_length = len(cluster.cells)
                self.assertTrue(initial_length != final_length and initial_length == 1 and final_length == 2)

    def test_extend_current_cluster_many_neighbours(self):
        t, partitioned_points, tiles = construct_input_extend_current_cluster_checking("many_neighbours")

        cell_to_check = np.array([40, 43, 23, 26])
        image_clusters = []

        ## This is in the main body of the algorithm (for just the central tile, for cell_to_check)
        while partitioned_points[1][1]:
            cell = partitioned_points[1][1].pop()
            if np.array_equal(cell, cell_to_check):
                Cluster.pure_create_cluster(cell, image_clusters)
                ### ...
                ### This would be in the clusterise() function.
                cluster = Cluster.pure_get_cluster(image_clusters)
                initial_length = len(cluster.cells)

                neighbouring_indices, neighbouring_tiles = Cluster.get_neighbouring_windows_fisherman(1, 1, t, t, tiles)
                neighbours = Cluster.get_and_remove_all_neighbours(cell, partitioned_points, neighbouring_indices, neighbouring_tiles)

                self.assertTrue(len(neighbours) == 3)
                self.assertTrue(len(partitioned_points[1][1]) == 0)  # Was the only cell in the tile
                Cluster.extend_current_cluster(cluster, neighbours)
                final_length = len(cluster.cells)
                self.assertTrue(initial_length != final_length and initial_length == 1 and final_length == 4)

    def test_clusterise_one_cell_one_cluster(self):
        t, partitioned_points, tiles = construct_input_clusterise_checking("one_cell_one_cluster")

        cell_to_check = np.array([0, 3, 0, 3])
        image_clusters = []

        ## This is in the main body of the algorithm (for just the central tile, for cell_to_check)
        while partitioned_points[0][0]:
            cell = partitioned_points[0][0].pop()
            if np.array_equal(cell, cell_to_check):
                Cluster.pure_create_cluster(cell, image_clusters)

                Cluster.clusterise(cell, image_clusters, partitioned_points, (0, 0), t, t, tiles)

                cluster = Cluster.pure_get_cluster(image_clusters)

                self.assertTrue(len(cluster.cells) == 1)
                self.assertTrue(len(partitioned_points[0][0]) == 0)  # Only cell and only cluster, so none left in this tile

    def test_clusterise_many_cells_one_cluster_same_tile(self):
        t, partitioned_points, tiles = construct_input_clusterise_checking("many_cells_one_cluster_one_tile")

        cell_to_check = np.array([12, 15, 12, 15])  # Last cell in the tile
        image_clusters = []

        ## This is in the main body of the algorithm (for just the central tile, for cell_to_check)
        while partitioned_points[0][0]:
            cell = partitioned_points[0][0].pop()
            if np.array_equal(cell, cell_to_check):
                Cluster.pure_create_cluster(cell, image_clusters)

                Cluster.clusterise(cell, image_clusters, partitioned_points, (0, 0), t, t, tiles)

                cluster = Cluster.pure_get_cluster(image_clusters)

                self.assertTrue(len(cluster.cells) == 5)
                self.assertTrue(len(partitioned_points[0][0]) == 0)  # All cells in this tile were in the same cluster, so none left in this tile

    def test_clusterise_many_cells_one_cluster_many_tiles(self):
        t, partitioned_points, tiles = construct_input_clusterise_checking("many_cells_one_cluster_many_tiles", set_t=True, updated_t=25)

        # This time we'll create a huge line - not the most 'cluster-y' looking cluster, but it satisfies the definition. Trying out a different
        # size of t and size of input to test that's not dependent or anything.

        cell_to_check = np.array([999, 1002, 999, 1002])  # Last cell in the tile
        image_clusters = []

        ## This is in the main body of the algorithm (for just the central tile, for cell_to_check)
        while partitioned_points[t-1][t-1]:
            cell = partitioned_points[t-1][t-1].pop()
            if np.array_equal(cell, cell_to_check):
                Cluster.pure_create_cluster(cell, image_clusters)

                Cluster.clusterise(cell, image_clusters, partitioned_points, (t-1, t-1), t, t, tiles)

                cluster = Cluster.pure_get_cluster(image_clusters)

                self.assertTrue(len(cluster.cells) == 1000)
                for i in range(t):
                    for j in range(t):
                        self.assertTrue(len(partitioned_points[i][j]) == 0)  # All removed!

    def test_clusterise_many_cells_many_clusters_many_tiles(self):
        t, partitioned_points, tiles = construct_input_clusterise_checking("many_cells_many_clusters_many_tiles")

        image_clusters = []

        # This is in the main body of the algorithm - now we can see it closely resembles the over-arching algorithm. We have successfully coded
        # up to the final algorithm!
        for i in range(t):
            for j in range(t):
                while partitioned_points[i][j]:
                    cell = partitioned_points[i][j].pop()
                    Cluster.pure_create_cluster(cell, image_clusters)
                    # cell, image_clusters, image, index, n1, n2, windows
                    Cluster.clusterise(cell, image_clusters, partitioned_points, (i, j), t, t, tiles)
                    cluster = Cluster.pure_get_cluster(image_clusters)

                    self.assertTrue(len(cluster.cells) == 4)
                    self.assertTrue(len(partitioned_points[i][j]) == 0)

    def test_fishermans_algorithm_many_cells_many_clusters_many_tiles(self):
        t, partitioned_points, tiles, w, h = construct_input_fishermans_checking("many_cells_many_clusters_many_tiles")

        clusters = Cluster.fishermans_algorithm(partitioned_points, (t, t), tiles, w, h)

        self.assertTrue(len(clusters) == 9)  # There are 9 clusters in the input

        for cluster in clusters:
            self.assertTrue(len(cluster.cells) == 4)  # ... and each has 4 cells.

    def test_fishermans_algorithm_many_cells_many_clusters_many_tiles_overlapping(self):
        t, partitioned_points, tiles, w, h = construct_input_fishermans_checking("many_cells_many_clusters_many_tiles_overlapping")

        clusters = Cluster.fishermans_algorithm(partitioned_points, (t, t), tiles, w, h)

        self.assertTrue(len(clusters) == 12)  # There are 12 clusters in the input

        num_singletons = 0
        num_groups = 0

        for cluster in clusters:
            if len(cluster.cells) == 1:  # ... and each has either 1 cell or 4
                num_singletons += 1
            elif len(cluster.cells) == 4:
                num_groups += 1

        self.assertTrue(num_singletons == 8 and num_groups == 4)  # ... and there are 8 singleton clusters and 4 with 4 cells each.

    def test_fishermans_with_simplest_many_examples(self):
        # That's pretty much the entire algorithm - but there are many situations that I won't have time to do myself.
        # Solution? Brute-force as many possible scenarios as I can and match the fisherman's algorithm with a naïve one.
        # If they always get the same results, then I can be confident I have implemented both correctly.

        for i in range(100):
            t, partitioned_points, tiles, w, h, points = construct_input_fishermans_checking("random", NUM_POINTS=1000)

            # The 'simplest' algorithm does not tile its inputs or take any other parameters
            simplest_clusters = Cluster.simplest(points)
            fished_clusters = Cluster.fishermans_algorithm(partitioned_points, (t, t), tiles, w, h)

            # Both algorithms return different objects - so we transfer them both into sets
            set1, set2 = set(), set()

            # Simplest
            for key, value in simplest_clusters.items():
                if value not in set1:
                    set1.add(value)  # We have a mapping from cells to clusters, so we have multiple duplicate values here.

            # Fisherman's
            for i in fished_clusters:
                set2.add(i)

            # Now check both algorithms return the same number of clusters - ignore the ordering!
            self.assertTrue(len(set1) == len(set2))

            # Now get the histogram of cluster sizes from both and assert that they are the same. This should suffice as a measure of correctness.
            histogram1, total_cluster_cells1 = get_histo(set1)
            histogram2, total_cluster_cells2 = get_histo(set2)

            self.assertTrue(total_cluster_cells1 == total_cluster_cells2)
            self.assertTrue(np.array_equal(histogram1, histogram2))


def get_histo(my_set):
    histogram = np.zeros(21, dtype=np.uint32)
    total_cluster_cells = 0
    for i in my_set:
        if i is None:
            raise TypeError
        value = len(i.cells)
        total_cluster_cells += len(i.cells)
        if value > 20:
            histogram[20] += 1
        else:
            histogram[value - 1] += 1
    return histogram, total_cluster_cells

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)



















 # End of file
