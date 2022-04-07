# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import deque

import numpy as np
from typing import Tuple

def get1_hot_vector(label, vec_size):
    one_hot_vec = np.zeros(vec_size)
    one_hot_vec[label] = 1
    return one_hot_vec


# This function has been taken from
# https://colab.research.google.com/drive/1uw7SmxUnLwn3Y6HcqqvEhHnyRJ0Jv8kb?usp=sharing#scrollTo=10-p1sSD25QH
def custom_get_shortest_path_to_object_type(controller, object_type):

    # Get all the reachable positions in the scene:
    positions = controller.step("GetReachablePositions").metadata["actionReturn"]

    # Preprocess each position from an xyz dictionary to a tuple with (x, y, z) entries. This allows it to be hashable
    # and the key for a dictionary / included in sets.
    positions_tuple = [(p["x"], p["y"], p["z"]) for p in positions]

    # Get all the neighbor positions from a given position. Using 1.5 * grid_size provides a bit of buffer room,
    # in case there are floating point errors.
    grid_size = 0.25
    neighbors = dict()
    for position in positions_tuple:
        position_neighbors = set()
        for p in positions_tuple:
            if position != p and (
                    (
                            abs(position[0] - p[0]) < 1.5 * grid_size
                            and abs(position[2] - p[2]) < 0.5 * grid_size
                    )
                    or (
                            abs(position[0] - p[0]) < 0.5 * grid_size
                            and abs(position[2] - p[2]) < 1.5 * grid_size
                    )
            ):
                position_neighbors.add(p)
        neighbors[position] = position_neighbors


    all_shortest_paths = []

    for obj in [o for o in controller.last_event.metadata['objects'] if o['objectType'].lower() == object_type.lower()]:
        start_pos = controller.last_event.metadata['agent']['position']
        end_pos = obj['position']
        start_pos = (start_pos['x'], start_pos['y'], start_pos['z'])
        end_pos = (end_pos['x'], end_pos['y'], end_pos['z'])
        all_shortest_paths.append(shortest_path(neighbors, positions_tuple, start_pos, end_pos))

    all_shortest_distances = []
    for path in all_shortest_paths:
        shortest_path_distance = 0
        for i in range(len(path) - 1):
            shortest_path_distance += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
        all_shortest_distances.append(shortest_path_distance)


# Rounds a given world coordinate to its closest grid position:
def closest_grid_point(positions_tuple, world_point):
    """Return the grid point that is closest to a world coordinate.
    Expects world_point=(x_pos, y_pos, z_pos). Note y_pos is ignored in the calculation.
    """
    def distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
        # ignore the y_pos
        return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

    min_dist = float("inf")
    closest_point = None
    assert len(positions_tuple) > 0
    for p in positions_tuple:
        dist = distance(p, world_point)
        if dist < min_dist:
            min_dist = dist
            closest_point = p
    return closest_point


# Calculates the shortest path (in x/z space) between 2 points:
def shortest_path(neighbors, positions_tuple, start, end):
    """Expects the start=(x_pos, y_pos, z_pos) and end=(x_pos, y_pos, z_pos).

    Note y_pos is ignored in the calculation.
    """
    start = closest_grid_point(positions_tuple, start)
    end = closest_grid_point(positions_tuple, end)
    print(start, end)

    if start == end:
        return [start]

    q = deque()
    q.append([start])

    visited = set()

    while q:
        path = q.popleft()
        pos = path[-1]

        if pos in visited:
            continue

        visited.add(pos)
        for neighbor in neighbors[pos]:
            if neighbor == end:
                return path + [neighbor]
            q.append(path + [neighbor])

    raise Exception("Invalid state. Must be a bug!")