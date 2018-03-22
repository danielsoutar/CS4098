import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm

TILE_SIZE = 25

convert = lambda x: int(round(x))


def partition(image, tile_size=TILE_SIZE, to_list=False, by_metric=False, scale=None, input_type="clean"):
    """
    Divide an image into a (tile_size x tile_size) grid and return the partitioned input.
    If by_metric flag is true, then we construct these tiles having an area of tile_size^2, in whatever scale the caller chooses.
    """
    tiles = []

    xMin = image[:, 0]
    xMax = image[:, 1]
    xAvg = np.mean(np.array([xMin, xMax]), axis=0)
    yMin = image[:, 2]
    yMax = image[:, 3]
    yAvg = np.mean(np.array([yMin, yMax]), axis=0)

    x_base = min(xMin)
    y_base = min(yMin)

    max_cell_width = 0
    max_cell_height = 0

    for cell in image:
        if cell[-1] > 0:
            continue
        current_cell_width = cell[1] - cell[0]
        current_cell_height = cell[3] - cell[2]
        if current_cell_width > max_cell_width:
            max_cell_width = current_cell_width
        if current_cell_height > max_cell_height:
            max_cell_height = current_cell_height

    if by_metric:
        x_step = tile_size * scale
        y_step = tile_size * scale

        num_x_steps = 0
        num_y_steps = 0

        while (x_base + (num_x_steps * x_step)) <= max(xMax):
            num_x_steps += 1

        while (y_base + (num_y_steps * x_step)) <= max(yMax):
            num_y_steps += 1

        partitioned_image = np.empty((num_x_steps, num_y_steps), dtype=object)

        for i in tqdm(range(num_x_steps)):
            for j in range(num_y_steps):
                x_left = x_base + (x_step * i)
                y_low = y_base + (y_step * j)

                x_right = x_base + x_step * (i + 1)
                y_high = y_base + y_step * (j + 1)

                result = ((yAvg >= y_low) & (yAvg < y_high) & (xAvg >= x_left) & (xAvg < x_right)).nonzero()[0]

                if to_list:
                    if input_type == "mixed":
                        items = image[result]
                        lymph_indices = ((items[:, 4] > 0) | (items[:, 5] > 0)).nonzero()[0]
                        lymphocytes = list(items[lymph_indices])
                        cancer_clusters_indices = ((items[:, 6] > 0)).nonzero()[0]
                        cancer_clusters = list(items[cancer_clusters_indices])
                        partitioned_image[i][j] = (lymphocytes, cancer_clusters)
                    else:
                        partitioned_image[i][j] = list(image[result])
                else:
                    partitioned_image[i][j] = image[result]
                tiles.append((x_left, y_low, x_right, y_high))

        tiles = np.array(tiles)

        return partitioned_image, tiles, max_cell_width, max_cell_height

    partitioned_image = np.empty((tile_size, tile_size), dtype=object)

    x_step = convert((max(xMax) - x_base) / (tile_size))
    y_step = convert((max(yMax) - y_base) / (tile_size))

    while (x_step * tile_size) <= max(xMax):
        x_step = x_step + 1

    while (y_step * tile_size) <= max(yMax):
        y_step = y_step + 1

    for i in tqdm(range(tile_size)):
        for j in range(tile_size):
            x_left = x_base + (x_step * i)
            y_low = y_base + (y_step * j)

            x_right = x_base + x_step * (i + 1)
            y_high = y_base + y_step * (j + 1)

            result = ((yAvg >= y_low) & (yAvg < y_high) & (xAvg >= x_left) & (xAvg < x_right)).nonzero()[0]

            if to_list:
                if input_type == "mixed":
                    items = image[result]
                    lymph_indices = ((items[:, 4] > 0) | (items[:, 5] > 0)).nonzero()[0]
                    lymphocytes = list(items[lymph_indices])
                    cancer_clusters_indices = ((items[:, 6] > 0)).nonzero()[0]
                    cancer_clusters = list(items[cancer_clusters_indices])
                    partitioned_image[i][j] = (lymphocytes, cancer_clusters)
                else:
                    partitioned_image[i][j] = list(image[result])
            else:
                partitioned_image[i][j] = image[result]
            tiles.append((x_left, y_low, x_right, y_high))

    tiles = np.array(tiles).reshape((tile_size, tile_size, 4))

    return partitioned_image, tiles, max_cell_width, max_cell_height


def partition_and_visualise(image, tile_size=TILE_SIZE, to_list=False, by_metric=False, scale=None, type="Cluster"):
    """Divide an image into a (num_tiles x num_tiles) grid, visualise, and return the partitioned input."""
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

    if by_metric:
        x_step = tile_size * scale
        y_step = tile_size * scale

        num_x_steps = 0
        num_y_steps = 0

        while (x_base + (num_x_steps * x_step)) <= max(xMax):
            num_x_steps += 1

        while (y_base + (num_y_steps * x_step)) <= max(yMax):
            num_y_steps += 1

        partitioned_image = np.empty((num_x_steps, num_y_steps), dtype=object)

        for i in tqdm(range(num_x_steps)):
            for j in range(num_y_steps):
                x_left = x_base + (x_step * i)
                y_low = y_base + (y_step * j)

                x_right = x_base + x_step * (i + 1)
                y_high = y_base + y_step * (j + 1)

                result = ((yAvg >= y_low) & (yAvg < y_high) & (xAvg >= x_left) & (xAvg < x_right)).nonzero()[0]

                for coordinates in image[result]:
                    draw.append(coordinates[0:4])

                if to_list:
                    partitioned_image[i][j] = list(image[result])
                else:
                    partitioned_image[i][j] = image[result]
                tiles.append((x_left, y_low, x_right, y_high))

        tiles = np.array(tiles)

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

        return partitioned_image, tiles

    partitioned_image = np.empty((tile_size, tile_size), dtype=object)

    x_step = convert((max(xMax) - x_base) / tile_size)
    y_step = convert((max(yMax) - y_base) / tile_size)

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

    return partitioned_image, tiles, max_cell_width, max_cell_height













# End of file
