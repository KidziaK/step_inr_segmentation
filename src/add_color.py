import numpy as np
import meshio
import matplotlib.pyplot as plt


def add_color(mesh: meshio.Mesh) -> meshio.Mesh:
    faces = np.asarray([c for c in mesh.cells if c.type == "triangle"][0].data)
    part_ids = np.asarray(mesh.cell_data["part"][0]).reshape(-1).astype(int, copy=False)
    palette_value = str(mesh.field_data["palette"][0])

    uniq, inverse = np.unique(part_ids, return_inverse=True)
    cmap = plt.get_cmap(palette_value)
    colors = cmap(np.linspace(0, 1, len(uniq)))

    rgba_uint8 = (colors[inverse] * 255).astype(np.uint8)
    rgba_uint8[:, 3] = 200

    V = np.asarray(mesh.points)
    V_flat = V[faces.flatten()]
    C_flat = np.repeat(rgba_uint8, 3, axis=0)
    F_flat = np.arange(len(V_flat)).reshape(-1, 3)

    return meshio.Mesh(
        V_flat,
        [("triangle", F_flat)],
        point_data={
            "red": C_flat[:, 0],
            "green": C_flat[:, 1],
            "blue": C_flat[:, 2],
            "alpha": C_flat[:, 3],
        },
    )