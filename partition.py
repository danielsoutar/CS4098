import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm

TILE_SIZE = 20

convert = lambda x: int(round(x))


def partition(image, tile_size=TILE_SIZE, to_list=False):
    """Divide an image into a (tile_size x tile_size) grid and return the partitioned input."""
    partitioned_image = np.empty((tile_size, tile_size), dtype=object)
    tiles = []

    xMin = image[:, 0]
    xMax = image[:, 1]
    xAvg = np.mean(np.array([xMin, xMax]), axis=0)
    yMin = image[:, 2]
    yMax = image[:, 3]
    yAvg = np.mean(np.array([yMin, yMax]), axis=0)

    x_base = min(xMin)
    y_base = min(yMin)

    x_step = convert((max(xMax) - x_base) / (tile_size))
    y_step = convert((max(yMax) - y_base) / (tile_size))

    while (x_step * tile_size) <= max(xMax):
        x_step = x_step + 1

    while (y_step * tile_size) <= max(yMax):
        y_step = y_step + 1

    max_cell_width = 0
    max_cell_height = 0

    for cell in image:
        current_cell_width = cell[1] - cell[0]
        current_cell_height = cell[3] - cell[2]
        if current_cell_width > max_cell_width:
            max_cell_width = current_cell_width
        if current_cell_height > max_cell_height:
            max_cell_height = current_cell_height


    for i in tqdm(range(tile_size)):
        for j in range(tile_size):
            x_left = x_base + (x_step * i)
            y_low = y_base + (y_step * j)

            x_right = x_base + x_step * (i + 1)
            y_high = y_base + y_step * (j + 1)

            result = ((yAvg >= y_low) & (yAvg < y_high) & (xAvg >= x_left) & (xAvg < x_right)).nonzero()[0]

            if to_list:
                partitioned_image[i][j] = list(image[result])
            else:
                partitioned_image[i][j] = image[result]
            tiles.append((x_left, y_low, x_right, y_high))

    tiles = np.array(tiles).reshape((tile_size, tile_size, 4))

    return partitioned_image, tiles, max_cell_width, max_cell_height


def partition_and_visualise(image, tile_size=TILE_SIZE, to_list=False):
    """Divide an image into a (num_tiles x num_tiles) grid, visualise, and return the partitioned input."""
    partitioned_image = np.empty((tile_size, tile_size), dtype=object)
    draw = []
    tiles = []

    xMin = image[:, 0]
    xMax = image[:, 1]
    xAvg = np.mean(np.array([xMin, xMax]), axis=0)
    yMin = image[:, 2]
    yMax = image[:, 3]
    yAvg = np.mean(np.array([yMin, yMax]), axis=0)

    x_base = min(xMin)
    y_base = min(yMin)

    x_step = convert((max(xMax) - x_base) / tile_size)
    y_step = convert((max(yMax) - y_base) / tile_size)

    for i in tqdm(range(tile_size)):
        for j in range(tile_size):
            x_left = x_base + (x_step * i)
            y_low = y_base + (y_step * j)

            x_right = x_base + x_step * (i + 1)
            y_high = y_base + y_step * (j + 1)

            result = ((yAvg >= y_low) & (yAvg < y_high) & (xAvg >= x_left) & (xAvg < x_right)).nonzero()[0]

            for coordinates in image[result]:
                draw.append(coordinates)

            if to_list:
                partitioned_image[i][j] = list(image[result])
            else:
                partitioned_image[i][j] = image[result]
            tiles.append((x_left, y_low, x_right, y_high))

    draw = np.asarray(draw)

    draw_xAvg = np.mean(np.array([draw[:, 0], draw[:, 1]]), axis=0)
    draw_yAvg = np.mean(np.array([draw[:, 2], draw[:, 3]]), axis=0)

    image_xAvg = np.mean(np.array([image[:, 0], image[:, 1]]), axis=0)
    image_yAvg = np.mean(np.array([image[:, 2], image[:, 3]]), axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

    ax1.set_title('Test', fontsize=30)
    ax1.scatter(draw_xAvg, draw_yAvg, s=0.1, c='b')

    ax2.set_title('Image', fontsize=30)
    ax2.scatter(image_xAvg, image_yAvg, s=0.1, c='r')

    ax3.set_title('Overlay (Image)', fontsize=30)
    ax3.scatter(image_xAvg, image_yAvg, s=0.1, c='r')
    ax3.scatter(draw_xAvg, draw_yAvg, s=0.1, c='b')

    ##############
    for window in tiles:
        xMin, yMin, xMax, yMax = window[0], window[1], window[2], window[3]
        for a in [ax1, ax2, ax3]:
            a.add_patch(
                patches.Rectangle(
                    (xMin, yMin),   # (x,y)
                    xMax - xMin,    # width
                    yMax - yMin,    # height
                    linestyle='--',
                    fill=False
                )
            )
    ##############
    plt.show()

    tiles = np.array(tiles).reshape((tile_size, tile_size, 4))

    return partitioned_image, tiles













# End of file
