import Current_Cluster as Clu
import partition
import unittest
import numpy as np

# Test cases for clustering algorithm

# Tests need to cover:

## References v values (want to minimise memory duplication but maintain correctness)
## - E.g. want cells in tile dict to reference a Clu, not duplicate it by value!

## Make sure that current_tile is outwardly traversed backwards and when checking neighbours iterates forwards
## - Need to check that we get the expected savings. Don't want to traverse right but include redundant cells!

## Check the method(s) associated with 'already_in_cluster()'

## Check the method(s) associated with initialising clusters

## Check neighbour retrieval gets expected results
## - E.g. does not include current_cell
## - E.g. includes those already in same cluster
## - E.g. does not include any cell beyond distance d of current_cell
## - E.g. does not include any cell already accounted for

## Check case where a neighbour is not in a cluster
## - i.e. cluster updates correctly, tile's dict updates correctly

## Check case where a neighbour is in same cluster as current_cell
## - i.e. no change occurs

## Check case where a neighbour is in another cluster
## - E.g. in smaller cluster (should be subsumed)
## - E.g. in same-sized cluster (should be subsumed)
## - E.g. in larger cluster (should subsume current_cell's cluster)

## Check that subsumed cluster is deleted and removed altogether
## - i.e. from tile's dict altogether (check for multiple references that it is completely gone)

## Check cases involving other tiles (yet to implement this)
## - i.e. do not search other tile if too far away from it
## - i.e. search over in one tile
## - i.e. search over in more than one tile

### Want lists of examples and expected outputs/results.


def initialise_clusters(n):
    image_clusters = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            image_clusters[i][j] = {}

    return image_clusters


class myTest(unittest.TestCase):

    def test_get_key(self):
        cell = np.array([0, 3, 0, 3])
        key = Clu.get_key(cell)
        expected = (0, 3, 0, 3)
        self.assertEqual(key, expected)

    def test_create_cluster_empty_tile(self):
        cell = np.array([0, 3, 0, 3])
        tile = {}
        index = (0, 0)
        key = (0, 3, 0, 3)
        expected_cells = [cell]
        expected_indices = [index]
        Clu.create_cluster(cell, tile, index)
        for e1, e2, e3, e4 in zip(tile[key].cells, expected_cells, tile[key].tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_create_cluster_nonempty_tile(self):
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10])
        tile = {}
        index1, index2 = (0, 0), (0, 0)
        keys = [(0, 3, 0, 3), (7, 10, 7, 10)]
        expected_cells = [[cell1], [cell2]]
        expected_indices = [[index1], [index2]]
        Clu.create_cluster(cell1, tile, index1)
        Clu.create_cluster(cell2, tile, index2)
        for key, expected_cell, expected_index in zip(keys, expected_cells, expected_indices):
            for e1, e2, e3, e4 in zip(tile[key].cells, expected_cell, tile[key].tiles, expected_index):
                for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                    self.assertTrue(c1 == c2)
                    self.assertTrue(c3 == c4)

    def test_get_cluster_one_cell(self):
        cell = np.array([0, 3, 0, 3])
        tile = {}
        index = (0, 0)
        key = (0, 3, 0, 3)
        Clu.create_cluster(cell, tile, index)
        clusters = tile.items()
        c = None
        for (key, cluster) in clusters:
            c = cluster
        d = Clu.get_cluster(tile, cell)
        self.assertTrue(c is d)

    def test_get_cluster_not_valid(self):
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10])
        tile = {}
        index2 = (0, 0)
        Clu.create_cluster(cell2, tile, index2)
        with self.assertRaises(KeyError):
            Clu.get_cluster(tile, cell1)

    def test_add_to_current_cluster_cell_and_cluster_in_same_tile(self):
        image_clusters = initialise_clusters(1)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10])
        tile = {}
        image_clusters[0][0] = tile
        index1, index2 = (0, 0), (0, 0)
        expected_cells = [cell1, cell2]
        expected_indices = [index1, index2]
        Clu.create_cluster(cell1, tile, index1)
        c = Clu.get_cluster(tile, cell1)

        Clu.add_to_current_cluster(c, image_clusters, cell2, index2)

        c = Clu.get_cluster(tile, cell1)
        d = Clu.get_cluster(tile, cell2)
        self.assertTrue(c is d)
        self.assertTrue(len(c.cells) == 2)
        for e1, e2, e3, e4 in zip(c.cells, expected_cells, c.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_add_to_current_cluster_cell_and_cluster_in_different_tiles(self):
        image_clusters = initialise_clusters(2)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10])
        tile1, tile2 = {}, {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index1, index2 = (0, 0), (0, 1)
        expected_cells = [cell1, cell2]
        expected_indices = [index1, index2]
        Clu.create_cluster(cell1, tile1, index1)
        Clu.create_cluster(cell2, tile2, index2)
        c = Clu.get_cluster(tile1, cell1)
        d = Clu.get_cluster(tile2, cell2)

        Clu.add_to_current_cluster(c, image_clusters, cell2, index2)
        c = Clu.get_cluster(tile1, cell1)
        d = Clu.get_cluster(tile2, cell2)
        self.assertTrue(c is d)
        self.assertTrue(len(c.cells) == 2)
        for e1, e2, e3, e4 in zip(c.cells, expected_cells, c.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_update_winning_cluster_121(self):
        image_clusters = initialise_clusters(1)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10])
        tile = {}
        image_clusters[0][0] = tile
        index1, index2 = (0, 0), (0, 0)
        expected_cells = [cell1, cell2]
        expected_indices = [index1, index2]
        Clu.create_cluster(cell1, tile, index1)
        c = Clu.get_cluster(tile, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2], [index2])
        c = Clu.get_cluster(tile, cell1)
        d = Clu.get_cluster(tile, cell2)

        self.assertTrue(c is d)
        self.assertTrue(len(c.cells) == 2)
        for e1, e2, e3, e4 in zip(c.cells, expected_cells, c.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_update_winning_cluster_many21(self):
        image_clusters = initialise_clusters(1)
        cell1, cell2, cell3 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10]), np.array([4, 7, 1, 4])
        tile = {}
        image_clusters[0][0] = tile
        index1, index2, index3 = (0, 0), (0, 0), (0, 0)
        expected_cells = [cell1, cell2, cell3]
        expected_indices = [index1, index2, index3]
        Clu.create_cluster(cell1, tile, index1)
        c = Clu.get_cluster(tile, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2], [index2])
        Clu.update_winning_cluster(c, image_clusters, [cell3], [index3])
        c = Clu.get_cluster(tile, cell1)
        d = Clu.get_cluster(tile, cell2)
        e = Clu.get_cluster(tile, cell3)

        self.assertTrue(c is d and d is e)
        self.assertTrue(len(c.cells) == 3)
        for e1, e2, e3, e4 in zip(c.cells, expected_cells, c.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_update_winning_cluster_many2many(self):
        image_clusters = initialise_clusters(1)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10])
        cell3, cell4 = np.array([4, 7, 1, 4]), np.array([5, 8, 4, 7])
        tile = {}
        image_clusters[0][0] = tile
        index1, index2, index3, index4 = (0, 0), (0, 0), (0, 0), (0, 0)
        expected_cells = [cell1, cell2, cell3, cell4]
        expected_indices = [index1, index2, index3, index4]
        Clu.create_cluster(cell1, tile, index1)
        c = Clu.get_cluster(tile, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2], [index2])
        Clu.update_winning_cluster(c, image_clusters, [cell3, cell4], [index3, index4])
        c = Clu.get_cluster(tile, cell1)
        d = Clu.get_cluster(tile, cell2)
        e = Clu.get_cluster(tile, cell3)
        f = Clu.get_cluster(tile, cell3)

        self.assertTrue(c is d and d is e and e is f)
        self.assertTrue(len(c.cells) == 4)
        for e1, e2, e3, e4 in zip(c.cells, expected_cells, c.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_create_cluster_fail_if_invalid_args(self):
        cell = None
        tile = None
        index = None
        with self.assertRaises(TypeError):
            Clu.create_cluster(cell, tile, index)

        cell = np.array([0, 3, 0, 3])

        with self.assertRaises(TypeError):
            Clu.create_cluster(cell, tile, index)  # Still fails, as None is not indexable

    def test_create_cluster_init_not_none(self):
        cell = np.array([0, 3, 0, 3])
        key = (0, 3, 0, 3)
        tile = {}
        index = (0, 0)
        Clu.create_cluster(cell, tile, index)
        self.assertTrue(tile[key].cells is not None)
        self.assertTrue(tile[key].tiles is not None)

    def test_create_cluster_init_correct_cells(self):
        expected = [np.array([0, 3, 0, 3])]
        cell = np.array([0, 3, 0, 3])
        key = (0, 3, 0, 3)
        tile = {}
        index = (0, 0)
        Clu.create_cluster(cell, tile, index)
        for c1, c2 in zip(expected, tile[key].cells):
            self.assertTrue(c1[0] == c2[0])
            self.assertTrue(c1[1] == c2[1])
            self.assertTrue(c1[2] == c2[2])
            self.assertTrue(c1[3] == c2[3])

    def test_create_cluster_init_correct_tiles(self):
        expected = [(0, 0)]
        cell = np.array([0, 3, 0, 3])
        key = (0, 3, 0, 3)
        tile = {}
        index = (0, 0)
        Clu.create_cluster(cell, tile, index)
        for t1, t2 in zip(expected, tile[key].tiles):
            self.assertTrue(t1[0] == t2[0])
            self.assertTrue(t1[1] == t2[1])

    def test_already_in_cluster_true(self):
        image_clusters = initialise_clusters(1)
        cell = np.array([0, 3, 0, 3])
        tile = {}
        image_clusters[0][0] = tile
        index = (0, 0)
        Clu.create_cluster(cell, tile, index)
        self.assertTrue(Clu.already_in_cluster(tile, cell))

    def test_already_in_cluster_false_cell_no_cluster(self):
        cell = np.array([0, 3, 0, 3])
        tile = {}
        self.assertFalse(Clu.already_in_cluster(tile, cell))

    def test_already_in_cluster_false_cell_in_other_tile(self):
        image_clusters = initialise_clusters(3)
        cell = np.array([0, 3, 0, 3])
        tile1 = {}
        tile2 = {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index = (0, 1)
        Clu.create_cluster(cell, tile2, index)
        self.assertFalse(Clu.already_in_cluster(tile1, cell))
        self.assertTrue(Clu.already_in_cluster(tile2, cell))

    ############################ CRITICAL ##############################
    def test_subsume_cluster1_larger(self):
        image_clusters = initialise_clusters(3)
        cell1, cell2, cell3 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10]), np.array([4, 7, 1, 4])
        cell4, cell5 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile1, tile2 = {}, {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index1, index2, index3, index4, index5 = (0, 0), (0, 0), (0, 0), (0, 1), (0, 1)
        expected_cells, expected_indices = [cell1, cell2, cell3, cell4, cell5], [index1, index2, index3, index4, index5]
        Clu.create_cluster(cell1, tile1, index1)
        c = Clu.get_cluster(tile1, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2, cell3], [index2, index3])
        Clu.create_cluster(cell4, tile2, index4)
        d = Clu.get_cluster(tile2, cell4)
        Clu.update_winning_cluster(d, image_clusters, [cell5], [index5])
        Clu.subsume(c, d, image_clusters)
        self.assertTrue(len(c.cells) == 5)
        self.assertTrue(d.cells is None)
        for e1, e2, e3, e4 in zip(c.cells, expected_cells, c.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_subsume_cluster1_equal(self):
        image_clusters = initialise_clusters(3)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10])
        cell3, cell4 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile1, tile2 = {}, {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index1, index2, index3, index4 = (0, 0), (0, 0), (0, 0), (0, 1)
        expected_cells, expected_indices = [cell1, cell2, cell3, cell4], [index1, index2, index3, index4]
        Clu.create_cluster(cell1, tile1, index1)
        c = Clu.get_cluster(tile1, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2], [index2])
        Clu.create_cluster(cell3, tile2, index3)
        d = Clu.get_cluster(tile2, cell3)
        Clu.update_winning_cluster(d, image_clusters, [cell4], [index4])
        Clu.subsume(c, d, image_clusters)
        self.assertTrue(len(c.cells) == 4)
        self.assertTrue(d.cells is None)
        for e1, e2, e3, e4 in zip(c.cells, expected_cells, c.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    def test_subsume_cluster1_smaller(self):
        image_clusters = initialise_clusters(3)
        cell1 = np.array([7, 10, 7, 10])
        cell2, cell3 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile1, tile2 = {}, {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index1, index2, index3 = (0, 0), (0, 0), (0, 1)
        expected_cells, expected_indices = [cell2, cell3, cell1], [index2, index3, index1]
        Clu.create_cluster(cell1, tile1, index1)
        c = Clu.get_cluster(tile1, cell1)
        Clu.create_cluster(cell2, tile2, index2)
        d = Clu.get_cluster(tile2, cell2)
        Clu.update_winning_cluster(d, image_clusters, [cell3], [index3])
        Clu.subsume(c, d, image_clusters)
        self.assertTrue(len(d.cells) == 3)
        self.assertTrue(c.cells is None)
        for e1, e2, e3, e4 in zip(d.cells, expected_cells, d.tiles, expected_indices):
            for c1, c2, c3, c4 in zip(e1, e2, e3, e4):
                self.assertTrue(c1 == c2)
                self.assertTrue(c3 == c4)

    # Need to show that neighbouring relation is commutative (i.e. if a is a neighbour of b, then b is a neighbour of a,
    # and likewise for non-neighbouring)
    # These tests assume d = 7
    def test_are_neighbours_all_too_far(self):
        Clu.set_d(7)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([15, 18, 15, 18])
        cell3 = np.array([30, 33, 30, 33])
        cell4, cell5 = np.array([5, 8, 0, 3]), np.array([16, 19, 11, 14])
        cell6 = np.array([40, 80, 40, 80])
        self.assertFalse(Clu.are_neighbours(cell1, cell2) or Clu.are_neighbours(cell2, cell1))
        self.assertFalse(Clu.are_neighbours(cell2, cell3) or Clu.are_neighbours(cell3, cell2))
        self.assertFalse(Clu.are_neighbours(cell4, cell5) or Clu.are_neighbours(cell5, cell4))
        self.assertFalse(Clu.are_neighbours(cell1, cell6) or Clu.are_neighbours(cell6, cell1))

    def test_are_neighbours_some_too_far(self):
        Clu.set_d(7)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([5, 8, 15, 18])
        cell3 = np.array([15, 18, 5, 8])
        cell4, cell5 = np.array([6, 7, 0, 10]), np.array([8, 9, 17, 117])
        cell6 = np.array([15, 18, 15, 80])
        self.assertFalse(Clu.are_neighbours(cell1, cell2) or Clu.are_neighbours(cell2, cell1))
        self.assertFalse(Clu.are_neighbours(cell1, cell3) or Clu.are_neighbours(cell3, cell1))
        self.assertFalse(Clu.are_neighbours(cell4, cell5) or Clu.are_neighbours(cell5, cell4))
        self.assertFalse(Clu.are_neighbours(cell3, cell6) or Clu.are_neighbours(cell6, cell3))

    def test_are_neighbours_all_close(self):
        Clu.set_d(7)
        cell1, cell2 = np.array([0, 3, 0, 3]), np.array([6, 9, 5, 8])
        cell3 = np.array([15, 18, 5, 8])
        cell4, cell5 = np.array([24, 27, 14, 17]), np.array([33, 68, 23, 60])
        cell6 = np.array([15, 18, 15, 80])
        cell7, cell8 = np.array([50, 53, 50, 53]), np.array([0, 100, 41, 44])
        self.assertTrue(Clu.are_neighbours(cell1, cell2) and Clu.are_neighbours(cell2, cell1))
        self.assertTrue(Clu.are_neighbours(cell2, cell3) and Clu.are_neighbours(cell3, cell2))
        self.assertTrue(Clu.are_neighbours(cell3, cell4) and Clu.are_neighbours(cell4, cell3))
        self.assertTrue(Clu.are_neighbours(cell4, cell5) and Clu.are_neighbours(cell5, cell4))
        self.assertTrue(Clu.are_neighbours(cell6, cell6))
        self.assertTrue(Clu.are_neighbours(cell7, cell8) and Clu.are_neighbours(cell8, cell7))
    ####################################################################

    ############################ CRITICAL ##############################
    # Need to do LOADS of examples for each of these, and test ALL permutations of tiles
    def test_get_all_neighbours_no_neighbours(self):
        tile_size = 3

        xMin = np.array([0, 30, 60, 0, 30, 60, 0, 30, 60])
        xMax = np.array([3, 33, 63, 3, 33, 63, 3, 33, 63])
        yMin = np.array([0, 0, 0, 30, 30, 30, 60, 60, 60])
        yMax = np.array([3, 3, 3, 33, 33, 33, 63, 63, 63])

        points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
        partitioned_points, windows = partition.partition(points, tile_size=tile_size)
        self.assertTrue(len(windows.reshape(3*3, 4)) == 9)

        # One cell per tile, so true for all 9 tiles - let's prove it.
        for i in range(tile_size):
            for j in range(tile_size):
                tile = partitioned_points[i][j]
                for cell in tile:
                    neighbouring_windows, neighbouring_indices, previous_indices = Clu.get_neighbouring_windows(i, j, tile_size, windows)
                    neighbours, num_windows = Clu.get_all_neighbours(cell, partitioned_points, 0, (i, j), neighbouring_windows, neighbouring_indices, 0)
                    length = 0
                    for n_cell, neighbour in neighbours:
                        length += 1
                    self.assertTrue(length == 0)

    def test_get_all_neighbours_same_tile(self):
        tile_size = 3
        cell = np.array([9, 12, 8, 11])  # We're going to be starting from the last cell, so we check this one.

        xMin = np.array([0, 6, 9, 30, 60, 0, 30, 60, 0, 30, 60])
        xMax = np.array([3, 9, 12, 33, 63, 3, 33, 63, 3, 33, 63])
        yMin = np.array([0, 4, 8, 0, 0, 30, 30, 30, 60, 60, 60])
        yMax = np.array([3, 7, 11, 3, 3, 33, 33, 33, 63, 63, 63])

        points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
        partitioned_points, windows = partition.partition(points, tile_size=tile_size)
        neighbouring_windows, neighbouring_indices, previous_indices = Clu.get_neighbouring_windows(0, 0, tile_size, windows)
        neighbours, num_windows = Clu.get_all_neighbours(cell, partitioned_points, 0, (0, 0), neighbouring_windows, neighbouring_indices, 0)

        length = 0
        expected_cells = [np.array([0, 3, 0, 3]), np.array([6, 9, 4, 7])]
        expected_tiles = [(0, 0), (0, 0)]
        actual_cells = []
        actual_tiles = []

        for neighbour, tile in neighbours:
            length += 1
            actual_cells.append(neighbour)
            actual_tiles.append(tile)

        self.assertTrue(length == 2)
        self.assertTrue(num_windows == 1)

        for cell1, cell2, tile1, tile2 in zip(expected_cells, actual_cells, expected_tiles, actual_tiles):
            self.assertTrue(np.array_equal(cell1, cell2))
            self.assertTrue(tile1 == tile2)

    def test_get_all_neighbours_different_tile(self):
        tile_size = 3

        # neighbours in (0, 0) and (1, 1) between [16, 19, 15, 18] and [25, 28, 24, 27] respectively.
        # First, let's show they are neighbours.
        a = np.array([16, 19, 15, 18])
        b = np.array([25, 28, 24, 27])
        self.assertTrue(Clu.are_neighbours(a, b) and Clu.are_neighbours(b, a))

        xMin = np.array([0, 6, 9,  16, 25, 30, 60, 0, 60, 0,  30, 60])
        xMax = np.array([3, 9, 12, 19, 28, 33, 63, 3, 63, 3,  33, 63])
        yMin = np.array([0, 4, 8,  15, 24, 0,  0,  30, 30, 60, 60, 60])
        yMax = np.array([3, 7, 11, 18, 27, 3,  3,  33, 33, 63, 63, 63])

        points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
        partitioned_points, windows = partition.partition(points, tile_size=tile_size)
        self.assertTrue(len(windows.reshape(3*3, 4)) == 9)

        # Finally, b ONLY has one neighbour. Let's prove it.
        num_neighbours = 0
        for i in range(tile_size):
            for j in range(tile_size):
                tile = partitioned_points[i][j]
                for cell in tile:
                    if Clu.are_neighbours(b, cell) and not np.array_equal(b, cell):
                        num_neighbours += 1

        self.assertTrue(num_neighbours == 1)

        neighbouring_windows, neighbouring_indices, previous_indices = Clu.get_neighbouring_windows(1, 1, tile_size, windows)
        neighbours, num_windows = Clu.get_all_neighbours(b, partitioned_points, 0, (1, 1), neighbouring_windows, neighbouring_indices, 0)

        # Get sole neighbour, being a
        expected_cells, expected_tiles = [a], [(0, 0)]
        actual_cells, actual_tiles = [], []

        for neighbour, tile in neighbours:
            actual_cells.append(neighbour)
            actual_tiles.append(tile)

        self.assertTrue(len(actual_cells) == 1 and len(actual_tiles) == 1)
        # cell b is borderline on an intersection of 4 tiles.
        self.assertTrue(num_windows == 4)

        for cell1, cell2, tile1, tile2 in zip(expected_cells, actual_cells, expected_tiles, actual_tiles):
            self.assertTrue(np.array_equal(cell1, cell2))
            self.assertTrue(tile1 == tile2)

    def test_get_all_neighbours_two_tiles(self):
        pass

    def test_get_all_neighbours_three_tiles(self):
        pass

    def test_get_all_neighbours_four_tiles(self):
        pass
    ####################################################################

    def test_in_same_cluster_true_same_tile(self):
        image_clusters = initialise_clusters(3)
        cell1, cell2, cell3 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10]), np.array([4, 7, 1, 4])
        cell4, cell5 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile = {}
        image_clusters[0][0] = tile
        index1, index2, index3, index4, index5 = (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        Clu.create_cluster(cell1, tile, index1)
        c = Clu.get_cluster(tile, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2, cell3], [index2, index3])
        Clu.create_cluster(cell4, tile, index4)
        d = Clu.get_cluster(tile, cell4)
        Clu.update_winning_cluster(d, image_clusters, [cell5], [index5])

        self.assertTrue(Clu.in_same_cluster(cell1, cell2, image_clusters, index1, index2) and Clu.in_same_cluster(cell2, cell1, image_clusters, index2, index1))
        self.assertTrue(Clu.in_same_cluster(cell1, cell3, image_clusters, index1, index3) and Clu.in_same_cluster(cell3, cell1, image_clusters, index3, index1))
        self.assertTrue(Clu.in_same_cluster(cell2, cell3, image_clusters, index2, index3) and Clu.in_same_cluster(cell3, cell1, image_clusters, index3, index2))

        self.assertTrue(Clu.in_same_cluster(cell4, cell5, image_clusters, index4, index5) and Clu.in_same_cluster(cell5, cell4, image_clusters, index5, index4))

    def test_in_same_cluster_true_different_tiles(self):
        image_clusters = initialise_clusters(3)
        cell1, cell2, cell3 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10]), np.array([4, 7, 1, 4])
        cell4, cell5 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile1, tile2 = {}, {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index1, index2, index3, index4, index5 = (0, 0), (0, 0), (0, 0), (0, 1), (0, 1)
        Clu.create_cluster(cell1, tile1, index1)
        c = Clu.get_cluster(tile1, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2, cell3], [index2, index3])
        Clu.create_cluster(cell4, tile2, index4)
        d = Clu.get_cluster(tile2, cell4)
        Clu.update_winning_cluster(d, image_clusters, [cell5], [index5])

        Clu.subsume(c, d, image_clusters)

        self.assertTrue(Clu.in_same_cluster(cell1, cell2, image_clusters, index1, index2) and Clu.in_same_cluster(cell2, cell1, image_clusters, index2, index1))
        self.assertTrue(Clu.in_same_cluster(cell1, cell3, image_clusters, index1, index3) and Clu.in_same_cluster(cell3, cell1, image_clusters, index3, index1))
        self.assertTrue(Clu.in_same_cluster(cell1, cell4, image_clusters, index1, index4) and Clu.in_same_cluster(cell4, cell1, image_clusters, index4, index1))
        self.assertTrue(Clu.in_same_cluster(cell1, cell5, image_clusters, index1, index5) and Clu.in_same_cluster(cell5, cell1, image_clusters, index5, index1))
        self.assertTrue(Clu.in_same_cluster(cell2, cell3, image_clusters, index2, index3) and Clu.in_same_cluster(cell3, cell2, image_clusters, index3, index2))
        self.assertTrue(Clu.in_same_cluster(cell2, cell4, image_clusters, index2, index4) and Clu.in_same_cluster(cell4, cell2, image_clusters, index4, index2))
        self.assertTrue(Clu.in_same_cluster(cell2, cell5, image_clusters, index2, index5) and Clu.in_same_cluster(cell5, cell2, image_clusters, index5, index2))
        self.assertTrue(Clu.in_same_cluster(cell3, cell4, image_clusters, index3, index4) and Clu.in_same_cluster(cell4, cell3, image_clusters, index4, index3))
        self.assertTrue(Clu.in_same_cluster(cell3, cell5, image_clusters, index3, index5) and Clu.in_same_cluster(cell5, cell3, image_clusters, index5, index3))
        self.assertTrue(Clu.in_same_cluster(cell4, cell5, image_clusters, index4, index5) and Clu.in_same_cluster(cell5, cell4, image_clusters, index5, index4))

    def test_in_same_cluster_false_same_tile(self):
        image_clusters = initialise_clusters(3)
        cell1, cell2, cell3 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10]), np.array([4, 7, 1, 4])
        cell4, cell5 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile = {}
        image_clusters[0][0] = tile
        index1, index2, index3, index4, index5 = (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        Clu.create_cluster(cell1, tile, index1)
        c = Clu.get_cluster(tile, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2, cell3], [index2, index3])
        Clu.create_cluster(cell4, tile, index4)
        d = Clu.get_cluster(tile, cell4)
        Clu.update_winning_cluster(d, image_clusters, [cell5], [index5])

        self.assertFalse(Clu.in_same_cluster(cell1, cell4, image_clusters, index1, index4) or Clu.in_same_cluster(cell4, cell1, image_clusters, index4, index1))
        self.assertFalse(Clu.in_same_cluster(cell1, cell5, image_clusters, index1, index5) or Clu.in_same_cluster(cell5, cell1, image_clusters, index5, index1))
        self.assertFalse(Clu.in_same_cluster(cell2, cell4, image_clusters, index2, index4) or Clu.in_same_cluster(cell4, cell2, image_clusters, index4, index2))
        self.assertFalse(Clu.in_same_cluster(cell2, cell5, image_clusters, index2, index5) or Clu.in_same_cluster(cell5, cell2, image_clusters, index5, index2))
        self.assertFalse(Clu.in_same_cluster(cell3, cell4, image_clusters, index3, index4) or Clu.in_same_cluster(cell4, cell3, image_clusters, index4, index3))
        self.assertFalse(Clu.in_same_cluster(cell3, cell5, image_clusters, index3, index5) or Clu.in_same_cluster(cell5, cell3, image_clusters, index5, index3))

    def test_in_same_cluster_false_different_tiles(self):
        image_clusters = initialise_clusters(3)
        cell1, cell2, cell3 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10]), np.array([4, 7, 1, 4])
        cell4, cell5 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile1, tile2 = {}, {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index1, index2, index3, index4, index5 = (0, 0), (0, 0), (0, 0), (0, 1), (0, 1)
        Clu.create_cluster(cell1, tile1, index1)
        c = Clu.get_cluster(tile1, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2, cell3], [index2, index3])
        Clu.create_cluster(cell4, tile2, index4)
        d = Clu.get_cluster(tile2, cell4)
        Clu.update_winning_cluster(d, image_clusters, [cell5], [index5])

        self.assertTrue(Clu.in_same_cluster(cell1, cell2, image_clusters, index1, index2) and Clu.in_same_cluster(cell2, cell1, image_clusters, index2, index1))
        self.assertTrue(Clu.in_same_cluster(cell1, cell3, image_clusters, index1, index3) and Clu.in_same_cluster(cell3, cell1, image_clusters, index3, index1))
        self.assertTrue(Clu.in_same_cluster(cell2, cell3, image_clusters, index2, index3) and Clu.in_same_cluster(cell3, cell2, image_clusters, index3, index2))
        self.assertTrue(Clu.in_same_cluster(cell4, cell5, image_clusters, index4, index5) and Clu.in_same_cluster(cell5, cell4, image_clusters, index5, index4))

        self.assertFalse(Clu.in_same_cluster(cell1, cell4, image_clusters, index1, index4) or Clu.in_same_cluster(cell4, cell1, image_clusters, index4, index1))
        self.assertFalse(Clu.in_same_cluster(cell1, cell5, image_clusters, index1, index5) or Clu.in_same_cluster(cell5, cell1, image_clusters, index5, index1))
        self.assertFalse(Clu.in_same_cluster(cell2, cell4, image_clusters, index2, index4) or Clu.in_same_cluster(cell4, cell2, image_clusters, index4, index2))
        self.assertFalse(Clu.in_same_cluster(cell2, cell5, image_clusters, index2, index5) or Clu.in_same_cluster(cell5, cell2, image_clusters, index5, index2))
        self.assertFalse(Clu.in_same_cluster(cell3, cell4, image_clusters, index3, index4) or Clu.in_same_cluster(cell4, cell3, image_clusters, index4, index3))
        self.assertFalse(Clu.in_same_cluster(cell3, cell5, image_clusters, index3, index5) or Clu.in_same_cluster(cell5, cell3, image_clusters, index5, index3))

    def test_in_same_cluster_true_before_and_after_subsuming_cluster(self):
        image_clusters = initialise_clusters(3)
        cell1, cell2, cell3 = np.array([0, 3, 0, 3]), np.array([7, 10, 7, 10]), np.array([4, 7, 1, 4])
        cell4, cell5 = np.array([16, 19, 16, 19]), np.array([21, 24, 15, 18])
        tile1, tile2 = {}, {}
        image_clusters[0][0] = tile1
        image_clusters[0][1] = tile2
        index1, index2, index3, index4, index5 = (0, 0), (0, 0), (0, 0), (0, 1), (0, 1)
        Clu.create_cluster(cell1, tile1, index1)
        c = Clu.get_cluster(tile1, cell1)
        Clu.update_winning_cluster(c, image_clusters, [cell2, cell3], [index2, index3])
        Clu.create_cluster(cell4, tile2, index4)
        d = Clu.get_cluster(tile2, cell4)
        Clu.update_winning_cluster(d, image_clusters, [cell5], [index5])

        self.assertTrue(Clu.in_same_cluster(cell1, cell2, image_clusters, index1, index2) and Clu.in_same_cluster(cell2, cell1, image_clusters, index2, index1))
        self.assertTrue(Clu.in_same_cluster(cell1, cell3, image_clusters, index1, index3) and Clu.in_same_cluster(cell3, cell1, image_clusters, index3, index1))
        self.assertTrue(Clu.in_same_cluster(cell4, cell5, image_clusters, index4, index5) and Clu.in_same_cluster(cell5, cell4, image_clusters, index5, index4))

        Clu.subsume(c, d, image_clusters)

        self.assertTrue(Clu.in_same_cluster(cell1, cell2, image_clusters, index1, index2) and Clu.in_same_cluster(cell2, cell1, image_clusters, index2, index1))
        self.assertTrue(Clu.in_same_cluster(cell1, cell3, image_clusters, index1, index3) and Clu.in_same_cluster(cell3, cell1, image_clusters, index3, index1))
        self.assertTrue(Clu.in_same_cluster(cell4, cell5, image_clusters, index4, index5) and Clu.in_same_cluster(cell5, cell4, image_clusters, index5, index4))

    ############################ CRITICAL ##############################
    # Need to do LOADS of examples for each of these, since we may have a bug with this function
    def test_get_neighbouring_windows_correct_num_windows_small(self):
        tile_size = 3
        expected_windows = 4*4 + 4*6 + 1*9
        expected_previous = 20 + 9

        xMin = np.array([0, 30, 60, 0, 30, 60, 0, 30, 60])
        xMax = np.array([3, 33, 63, 3, 33, 63, 3, 33, 63])
        yMin = np.array([0, 0, 0, 30, 30, 30, 60, 60, 60])
        yMax = np.array([3, 3, 3, 33, 33, 33, 63, 63, 63])

        points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()

        partitioned_points, windows = partition.partition(points, tile_size=tile_size)

        total_windows = 0
        total_previous = 0
        num_points = 0

        for i in range(tile_size):
            for j in range(tile_size):
                current_tile = partitioned_points[i][j]
                for cell in current_tile:
                    num_points += 1
                    neighbouring_windows, neighbouring_indices, previous_indices = Clu.get_neighbouring_windows(i, j, tile_size, windows)
                    self.assertTrue(len(neighbouring_windows) == len(neighbouring_indices))
                    total_windows += len(neighbouring_windows) + 1  # Accounting for the current tile as well since we would obviously check that one
                    total_previous += len(previous_indices)

        self.assertTrue(num_points == 9)
        self.assertEqual(total_windows, expected_windows)
        self.assertEqual(total_previous, expected_previous)

    def test_get_neighbouring_windows_correct_num_windows_big(self):
        tile_size = 20
        expected_windows = 4*4 + 4*18*6 + 18*18*9
        expected_previous = 39 + 19*(4 + 3 + 18*5)

        a = [x for x in range(400) if x % 20 == 0] * 20
        b = [y for x in range(400) for y in [x]*20 if x % 20 == 0]

        xMin = np.array(a)
        xMax = np.copy(xMin) + 4
        yMin = np.array(b)
        yMax = np.copy(yMin) + 4

        points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
        partitioned_points, windows = partition.partition(points, tile_size=tile_size)
        self.assertTrue(windows.shape == (tile_size, tile_size, 4))

        total_windows = 0
        total_previous = 0
        num_points = 0
        num_cells_per_tile = 0

        for i in range(tile_size):
            for j in range(tile_size):
                current_tile = partitioned_points[i][j]
                num_cells_per_tile = 0
                for cell in current_tile:
                    num_cells_per_tile += 1
                    num_points += 1
                    neighbouring_windows, neighbouring_indices, previous_indices = Clu.get_neighbouring_windows(i, j, tile_size, windows)
                    self.assertTrue(len(neighbouring_windows) == len(neighbouring_indices))
                    total_windows += len(neighbouring_windows) + 1  # Accounting for the current tile as well since we would obviously check that one
                    total_previous += len(previous_indices)
                self.assertTrue(num_cells_per_tile == 1)

        self.assertTrue(num_points == 400)
        self.assertEqual(total_windows, expected_windows)
        self.assertEqual(total_previous, expected_previous)
    ####################################################################

    ############################ CRITICAL ##############################
    # Given the amount of testing in previous sections this is tested less, but we need
    # to check our combination of those functions above is correct, and that we pass in
    # valid arguments. We'll focus on that.
    def test_window_cleaning_algorithm_one_cell_one_tile(self):
        pass

    def test_window_cleaning_algorithm_one_cell_many_tiles(self):
        pass

    def test_window_cleaning_algorithm_one_cell_per_tile_per_tiles(self):
        pass

    def test_window_cleaning_algorithm_one_cluster_of_many_cells_one_tile(self):
        pass

    def test_window_cleaning_algorithm_one_cluster_of_many_cells_per_tiles(self):
        pass

    def test_window_cleaning_algorithm_one_overlapping_cluster_two_tiles(self):
        pass

    def test_window_cleaning_algorithm_one_overlapping_cluster_three_tiles(self):
        pass

    def test_window_cleaning_algorithm_one_overlapping_cluster_four_tiles(self):
        pass

    def test_window_cleaning_algorithm_one_cluster_subsuming_another_same_tile(self):
        pass

    def test_window_cleaning_algorithm_one_cluster_subsuming_another_different_tiles(self):
        pass


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
