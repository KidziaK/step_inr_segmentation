import numpy as np


def scale_mesh_to_box(vertices: np.ndarray) -> np.ndarray:
    bbox_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    V = vertices - bbox_center
    denom = np.max(np.abs(V))
    return 0.5 * V / denom

