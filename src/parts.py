import numpy as np
import meshio
def extract_part_ids_from_mesh(mesh: meshio.Mesh) -> np.ndarray:
    return np.asarray(mesh.cell_data["part"][0]).reshape(-1).astype(int, copy=False)

