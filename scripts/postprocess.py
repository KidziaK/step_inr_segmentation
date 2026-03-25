#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import meshio
import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.add_color import add_color
from src.color_pallette import ColorPalette
from src.geometry import scale_mesh_to_box
from src.io import write_mesh

def process_npz_file(npz_path: Path, out_path: Path, palette: ColorPalette) -> None:
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["vertices"])
    F = np.asarray(data["faces"])
    part = np.asarray(data["step_part_id_face"]).reshape(-1).astype(int, copy=False)

    V_scaled = scale_mesh_to_box(V)

    mesh_in = meshio.Mesh(
        V_scaled,
        [("triangle", F)],
        cell_data={"part": [part]},
        field_data={"palette": [palette.value]},
    )
    output_mesh = add_color(mesh_in)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mesh(output_mesh, str(out_path))


def process_ply_file(ply_path: Path, out_path: Path, palette: ColorPalette) -> None:
    tm = trimesh.load(str(ply_path))
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

    V_scaled = scale_mesh_to_box(vertices)

    mesh_in = meshio.Mesh(
        V_scaled,
        [("triangle", faces)],
        cell_data={"part": [color_indices.astype(int, copy=False)]},
        field_data={"palette": [palette.value]},
    )
    output_mesh = add_color(mesh_in)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mesh(output_mesh, str(out_path))


def process_file(input_path: Path, output_path: Path, palette: ColorPalette) -> None:
    if input_path.suffix == ".npz":
        process_npz_file(input_path, output_path, palette)
    elif input_path.suffix == ".ply":
        process_ply_file(input_path, output_path, palette)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess mesh files: scale to [-0.5, 0.5]^3 and assign colors"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Input directory containing 'ours' and 'partfield' folders"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=ColorPalette.TAB20.value,
        choices=[p.value for p in ColorPalette],
        help="Color palette to use for part coloring"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".npz", ".ply"],
        help="File extensions to process"
    )
    
    args = parser.parse_args()
    
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    palette = ColorPalette(args.palette)
    
    for folder_name in ["ours", "partfield"]:
        input_folder = input_root / folder_name
        output_folder = output_root / folder_name
        
        if not input_folder.exists():
            print(f"Warning: {input_folder} does not exist, skipping...")
            continue
        
        print(f"Processing {folder_name}...")
        
        resolution_folders = ["coarser_meshes240", "default_meshes240", "finer_meshes240"]
        if folder_name == "partfield":
            resolution_folders = ["coarser_partfield240", "default_partfield240", "finer_partfield240"]
        
        file_groups = {}
        
        for ext in args.extensions:
            for input_file in sorted(input_folder.rglob(f"*{ext}")):
                filename = input_file.name
                base_name = filename.replace("_labels.ply", "").replace(".ply", "").replace(".npz", "")
                
                if base_name not in file_groups:
                    file_groups[base_name] = {}
                
                for res_folder in resolution_folders:
                    if res_folder in str(input_file):
                        file_groups[base_name][res_folder] = input_file
                        break
        
        for base_name, group_files in file_groups.items():
            for _, mesh_path in group_files.items():
                try:
                    rel_path = mesh_path.relative_to(input_folder)
                    output_file = output_folder / rel_path.with_suffix(".ply")
                    process_file(mesh_path, output_file, palette)
                    print(f"  Processed: {rel_path}")
                except Exception as e:
                    print(f"  Error processing {mesh_path}: {e}")
    
    print("Postprocessing complete!")


if __name__ == "__main__":
    main()
