import matplotlib.pyplot as plt
import numpy as np
import typing
from numba import njit

from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)


@njit
def is_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    N = polygon.shape[0]
    is_inside = False
    x = point[0]
    y = point[1]
    for i in range(N-1):
        x1 = polygon[i][0]
        y1 = polygon[i][1]
        x2 = polygon[i+1][0]
        y2 = polygon[i+1][1]
        if ((y < y1) != (y < y2)) and ((x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)):
            is_inside = ~is_inside
    return is_inside


if __name__ == "__main__":
    N = 100_000

    x = np.random.random(N)
    y = np.random.random(N)

    points = np.vstack((x, y)).T

    polygon = random_polygon(num_points=20)
    polygon.append(polygon[0])
    polygon = np.asarray(polygon)

    inner_points: list = []
    outer_points: list = []
    for point in points:
        is_inside = is_inside_polygon(point, polygon)
        if is_inside:
            inner_points.append(point)
        else:
            outer_points.append(point)

    inner_points = np.asarray(inner_points)
    outer_points = np.asarray(outer_points)

    plt.plot(polygon[:, 0], polygon[:, 1], "b-", linewidth=4.0)

    plt.plot(inner_points[:, 0], inner_points[:, 1], marker='.',
             markersize='3', linestyle='None', color='red')

    plt.plot(outer_points[:, 0], outer_points[:, 1], marker='.',
             markersize='3', linestyle='None', color='grey')

    plt.show()
