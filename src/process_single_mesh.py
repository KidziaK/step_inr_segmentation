from pathlib import Path

import meshio
import numpy as np
import trimesh

from .add_color import add_color
from .color_pallette import ColorPalette
from .geometry import scale_mesh_to_box
from .io import write_mesh


def process_single_mesh(input_path: str, output_path: str) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)
    palette = ColorPalette.TAB20.value

    if in_path.suffix == ".npz":
        data = np.load(in_path, allow_pickle=True)
        V = np.asarray(data["vertices"])
        F = np.asarray(data["faces"])
        part = np.asarray(data["step_part_id_face"]).reshape(-1).astype(int, copy=False)

        mesh_in = meshio.Mesh(
            scale_mesh_to_box(V),
            [("triangle", F)],
            cell_data={"part": [part]},
            field_data={"palette": [palette]},
        )
        output_mesh = add_color(mesh_in)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_mesh(output_mesh, str(out_path))
        return

    if in_path.suffix == ".ply":
        tm = trimesh.load(str(in_path))
        vertices = np.asarray(tm.vertices)
        faces = np.asarray(tm.faces)
        ply_data = tm.metadata["_ply_raw"]
        face_data = ply_data["face"]["data"]
        face_colors = np.column_stack(
            [
                face_data["red"],
                face_data["green"],
                face_data["blue"],
                face_data["alpha"],
            ]
        )
        _, color_indices = np.unique(face_colors, axis=0, return_inverse=True)

        mesh_in = meshio.Mesh(
            scale_mesh_to_box(vertices),
            [("triangle", faces)],
            cell_data={"part": [color_indices.astype(int, copy=False)]},
            field_data={"palette": [palette]},
        )
        output_mesh = add_color(mesh_in)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_mesh(output_mesh, str(out_path))
        return

    raise ValueError(f"Unsupported file format: {in_path.suffix}")
