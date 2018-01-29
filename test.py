import Cluster
import partition
import unittest
import numpy as np

# Test cases for clustering algorithm (fishermans)

# Tests need to cover:

## Make sure that only tiles within range of a given cell are checked - do not want unnecessary search!

## Check the method(s) associated with initialising clusters and adding cells to them (always a list apart from the first cell)

## Check neighbour retrieval gets expected results
## - E.g. does not include current_cell (trivially a cell is its own neighbour)
## - E.g. does not include those already in same cluster (should have been removed)
## - E.g. does not include any cell beyond distance d of current_cell (such cells are not neighbours)
## - E.g. does not include any cell already accounted for (should have been removed)
## - E.g. includes cells overlapping on tiles even if current cell is too far away from tile (extend the neighbouring relation for this)

## Check for when a neighbour is added to the cluster
## - i.e. cluster updates correctly, neighbour removed from input

### Want lists of examples and expected outputs/results.


def initialise_clusters(n):
    image_clusters = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            image_clusters[i][j] = {}

    return image_clusters


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

    def test_get_neighbouring_windows_corner(self):
        pass

    def test_get_neighbouring_windows_side(self):
        pass

    def test_get_neighbouring_windows_central(self):
        pass

    def test_correct_number_of_total_neighbouring_windows(self):
        pass

    def test_get_and_remove_all_neighbours_no_neighbours_alone(self):
        pass

    def test_get_and_remove_all_neighbours_no_neighbours_other_cells_too_far(self):
        pass

    def test_get_and_remove_all_neighbours_one_neighbour_same_tile(self):
        pass

    def test_get_and_remove_all_neighbours_one_neighbour_different_tile_no_overlap(self):
        pass

    def test_get_and_remove_all_neighbours_one_neighbour_different_tile_overlap(self):
        pass

"""
test_tile_size = 3

# Should have neighbours in (0, 0) and (1, 1) between [15, 24, 8, 12] and [31, 33, 8, 12] respectively.
# First, let's show they are neighbours.
a = np.array([15, 26, 8, 12])
b = np.array([31, 33, 8, 12])
assert(Cluster.are_neighbours(a, b) and Current_Cluster.are_neighbours(b, a))

# BUT
# BUT
# BUT
# b cannot reach tile (0, 0), whereas a can reach tile (1, 0)
first_window = np.array([0, 22, 0, 22])
second_window = np.array([22, 44, 0, 22])
assert(not (Current_Cluster.are_neighbours(b, first_window) or Current_Cluster.are_neighbours(first_window, b)))
assert(Current_Cluster.are_neighbours(a, second_window) and Current_Cluster.are_neighbours(second_window, a))

# xMinOk = (xMin1 - width) < xMin2
# xMaxOk = xMax2 < (xMax1 + width)
# yMinOk = (yMin1 - height) < yMin2
# yMaxOk = yMax2 < (yMax1 + height)

# Also, [31, 33, 8, 12] has a neighbour in its own tile, being [30, 33, 0, 3]
c = np.array([30, 33, 0, 3])
assert(Current_Cluster.are_neighbours(b, c) and Current_Cluster.are_neighbours(c, b))

xMin = np.array([0, 15, 31, 30, 60, 0,  30, 60, 0,  30, 60])
xMax = np.array([3, 26, 33, 33, 63, 3,  33, 63, 3,  33, 63])
yMin = np.array([0, 8,  8,  0,  0,  30, 30, 30, 60, 60, 60])
yMax = np.array([3, 12, 12, 3,  3,  33, 33, 33, 63, 63, 63])

points = np.stack((xMin, xMax, yMin, yMax), 0).transpose()
partitioned_points, windows = partition.partition(points, tile_size=test_tile_size, load_bar=False)
flattened_windows_list = np.array(windows).reshape((test_tile_size*test_tile_size, 4))
print(flattened_windows_list)

fig = plt.figure()

ax1 = fig.add_subplot(111, aspect='equal')

ax1.scatter(points[:,0], points[:,2], s=5, c='g')
ax1.scatter(points[:,1], points[:,3], s=5, c='b')

for window in flattened_windows_list:
    xMin, yMin, xMax, yMax = window[0], window[1], window[2], window[3]
    ax1.add_patch(
        patches.Rectangle(
            (xMin, yMin),   # (x,y)
            xMax - xMin,    # width
            yMax - yMin,    # height
            linestyle='--',
            fill=False
        )
    )

plt.show()

# Finally, b ONLY has two neighbours. Let's prove it.
num_neighbours = 0
for i in range(test_tile_size):
    for j in range(test_tile_size):
        tile = partitioned_points[i][j]
        for cell in tile:
            if Current_Cluster.are_neighbours(b, cell) and not np.array_equal(b, cell):
                num_neighbours += 1

assert(num_neighbours == 2)

neighbouring_windows, neighbouring_indices = Current_Cluster.get_neighbouring_windows(0, 0, test_tile_size, windows)

# The order of traversal matters, so let's just rearrange the two points in tile (1, 1), which are [25, 28, 24, 27] and [30, 33, 30, 33], for the sake of this test.
temp = partitioned_points[1][1][0]
partitioned_points[1][1][0] = partitioned_points[1][1][1]
partitioned_points[1][1][1] = temp

neighbours = Current_Cluster.get_all_neighbours(b, partitioned_points, 0, (1, 1), neighbouring_windows, neighbouring_indices)

# Gets neighbours in own cell first
expected_cells, expected_tiles = [np.array([30, 33, 30, 33]), np.array([16, 19, 15, 18])], [(1, 1), (0, 0)]
actual_cells, actual_tiles = [], []

for neighbour, tile in neighbours:
    actual_cells.append(neighbour)
    actual_tiles.append(tile)

assert(len(actual_cells) == 2 and len(actual_tiles) == 2)

for cell1, cell2, tile1, tile2 in zip(expected_cells, actual_cells, expected_tiles, actual_tiles):
    assert(np.array_equal(cell1, cell2))
    assert(tile1 == tile2)
"""

    def test_get_and_remove_all_neighbours_many_neighbours_same_tile(self):
        pass

    def test_get_and_remove_all_neighbours_many_neighbours_multiple_tiles(self):
        pass

    def test_get_and_remove_all_neighbours_not_itself(self):
        pass

    def test_get_and_remove_all_neighbours_none_from_same_cluster(self):
        pass

    def test_get_and_remove_all_neighbours_none_already_searched(self):
        pass

    def test_extend_current_cluster_no_neighbours(self):
        pass

    def test_extend_current_cluster_one_neighbour(self):
        pass

    def test_extend_current_cluster_many_neighbours(self):
        pass

    def test_clusterise_one_cell_one_cluster(self):
        pass

    def test_clusterise_many_cells_one_cluster_same_tile(self):
        pass

    def test_clusterise_many_cells_one_cluster_many_tiles(self):
        pass

    def test_fishermans_with_simplest_many_examples(self):
        pass



if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
