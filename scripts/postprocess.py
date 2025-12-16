#!/usr/bin/env python3

import numpy as np
import meshio
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
import argparse

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class ColorPalette(str, Enum):
    TAB20 = "tab20"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"
    RAINBOW = "gist_rainbow"


def scale_mesh_to_box(vertices: np.ndarray) -> np.ndarray:
    V = vertices - vertices.mean(axis=0)
    
    vals, vecs = np.linalg.eigh(np.cov(V.T))
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] *= -1
    
    V = V @ vecs
    V = 0.5 * V / np.max(np.abs(V))
    
    return V


def extract_part_ids_from_mesh(mesh: meshio.Mesh) -> np.ndarray:
    triangle_cells = [c for c in mesh.cells if c.type == "triangle"]
    if not triangle_cells:
        return np.array([0])
    
    num_faces = len(triangle_cells[0].data)
    
    if mesh.cell_data:
        for key in mesh.cell_data.keys():
            if 'part' in key.lower() or 'label' in key.lower() or 'color' in key.lower():
                part_data = mesh.cell_data[key]
                if len(part_data) > 0:
                    try:
                        if isinstance(part_data[0], np.ndarray):
                            if part_data[0].ndim > 0:
                                result = np.concatenate([arr.flatten() for arr in part_data])
                            else:
                                result = np.array(part_data).flatten()
                        else:
                            result = np.array(part_data).flatten()
                        if len(result) > 0:
                            if len(result) == num_faces:
                                return result
                            elif len(result) == 1:
                                return np.repeat(result, num_faces)
                    except (ValueError, IndexError):
                        continue
    
    if mesh.point_data:
        for key in mesh.point_data.keys():
            if 'part' in key.lower() or 'label' in key.lower():
                point_parts = mesh.point_data[key]
                if len(point_parts) > 0 and len(triangle_cells) > 0:
                    try:
                        faces = triangle_cells[0].data
                        face_parts = []
                        for face in faces:
                            if len(face) > 0 and np.max(face) < len(point_parts):
                                face_vertex_parts = point_parts[face]
                                unique, counts = np.unique(face_vertex_parts, return_counts=True)
                                face_parts.append(unique[np.argmax(counts)])
                            else:
                                face_parts.append(0)
                        if len(face_parts) == num_faces:
                            return np.array(face_parts)
                    except (ValueError, IndexError):
                        continue
    
    return np.zeros(num_faces, dtype=int)


def process_npz_file(npz_path: Path, out_path: Path, palette: ColorPalette) -> None:
    data = np.load(npz_path, allow_pickle=True)
    V = data["vertices"]
    F = data["faces"]
    part = data.get("step_part_id_face", np.zeros(len(F), dtype=int))
    
    V = scale_mesh_to_box(V)
    
    uniq, inverse = np.unique(part, return_inverse=True)
    cmap = plt.get_cmap(palette.value)
    colors = cmap(np.linspace(0, 1, len(uniq)))
    rgba = colors[inverse]
    
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    rgba_uint8[:, 3] = 200
    
    V_flat = V[F.flatten()]
    C_flat = np.repeat(rgba_uint8, 3, axis=0)
    F_flat = np.arange(len(V_flat)).reshape(-1, 3)
    
    mesh = meshio.Mesh(
        V_flat,
        [("triangle", F_flat)],
        point_data={
            "red": C_flat[:, 0],
            "green": C_flat[:, 1],
            "blue": C_flat[:, 2],
            "alpha": C_flat[:, 3]
        }
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(out_path, binary=True)


def process_ply_file(ply_path: Path, out_path: Path, palette: ColorPalette) -> None:
    if HAS_TRIMESH:
        try:
            tm = trimesh.load(str(ply_path))
            if not isinstance(tm, trimesh.Trimesh):
                raise ValueError(f"File {ply_path} does not contain a triangle mesh")
            vertices = tm.vertices
            faces = tm.faces
            
            cell_data = {}
            face_colors = None
            
            if hasattr(tm, 'metadata') and tm.metadata and '_ply_raw' in tm.metadata:
                ply_data = tm.metadata['_ply_raw']
                if 'face' in ply_data and 'data' in ply_data['face']:
                    face_data = ply_data['face']['data']
                    if face_data.dtype.names and 'red' in face_data.dtype.names and 'green' in face_data.dtype.names and 'blue' in face_data.dtype.names:
                        alpha = face_data['alpha'] if 'alpha' in face_data.dtype.names else np.full(len(face_data), 255)
                        face_colors = np.column_stack([
                            face_data['red'],
                            face_data['green'],
                            face_data['blue'],
                            alpha
                        ])
            
            if hasattr(tm, 'face_attributes') and tm.face_attributes:
                for key, value in tm.face_attributes.items():
                    cell_data[key] = [value]
            
            if face_colors is not None:
                unique_colors, color_indices = np.unique(face_colors, axis=0, return_inverse=True)
                cell_data['face_color_part_id'] = [color_indices]
            
            point_data = {}
            if hasattr(tm, 'vertex_attributes') and tm.vertex_attributes:
                for key, value in tm.vertex_attributes.items():
                    point_data[key] = value
            
            mesh = meshio.Mesh(
                vertices,
                [("triangle", faces)],
                point_data=point_data,
                cell_data=cell_data
            )
        except Exception as e:
            error_msg = str(e)
            if "buffer size must be a multiple of element size" in error_msg:
                raise ValueError(f"PLY file format not supported: {ply_path}")
            try:
                mesh = meshio.read(ply_path)
            except (ValueError, OSError) as e2:
                error_msg2 = str(e2)
                if "buffer size must be a multiple of element size" in error_msg2:
                    raise ValueError(f"PLY file format not supported: {ply_path}")
                raise ValueError(f"Failed to read PLY file {ply_path}: {e} (meshio also failed: {e2})")
    else:
        try:
            mesh = meshio.read(ply_path)
        except (ValueError, OSError) as e:
            error_msg = str(e)
            if "buffer size must be a multiple of element size" in error_msg:
                raise ValueError(f"PLY file format not supported by meshio (binary format issue). Install trimesh for better support: {ply_path}")
            raise ValueError(f"Failed to read PLY file {ply_path}: {e}")
    
    if len(mesh.cells) == 0:
        raise ValueError(f"No cells found in {ply_path}")
    
    triangle_cells = [c for c in mesh.cells if c.type == "triangle"]
    if not triangle_cells:
        raise ValueError(f"No triangle cells found in {ply_path}")
    
    faces = triangle_cells[0].data
    if len(faces) == 0:
        raise ValueError(f"Empty face array in {ply_path}")
    
    vertices = mesh.points
    if len(vertices) == 0:
        raise ValueError(f"Empty vertex array in {ply_path}")
    
    V_scaled = scale_mesh_to_box(vertices)
    
    part = extract_part_ids_from_mesh(mesh)
    
    if len(part) != len(faces):
        if len(part) == 1:
            part = np.repeat(part, len(faces))
        elif len(part) > 0:
            part = np.tile(part, (len(faces) // len(part) + 1))[:len(faces)]
        else:
            part = np.zeros(len(faces), dtype=int)
    
    uniq, inverse = np.unique(part, return_inverse=True)
    cmap = plt.get_cmap(palette.value)
    colors = cmap(np.linspace(0, 1, len(uniq)))
    rgba = colors[inverse]
    
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    rgba_uint8[:, 3] = 200
    
    V_flat = V_scaled[faces.flatten()]
    C_flat = np.repeat(rgba_uint8, 3, axis=0)
    F_flat = np.arange(len(V_flat)).reshape(-1, 3)
    
    output_mesh = meshio.Mesh(
        V_flat,
        [("triangle", F_flat)],
        point_data={
            "red": C_flat[:, 0],
            "green": C_flat[:, 1],
            "blue": C_flat[:, 2],
            "alpha": C_flat[:, 3]
        }
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_mesh.write(out_path, binary=True)


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
        
        for ext in args.extensions:
            for input_file in sorted(input_folder.rglob(f"*{ext}")):
                rel_path = input_file.relative_to(input_folder)
                output_file = output_folder / rel_path.with_suffix(".ply")
                
                try:
                    process_file(input_file, output_file, palette)
                    print(f"  Processed: {rel_path}")
                except Exception as e:
                    print(f"  Error processing {rel_path}: {e}")
                    continue
    
    print("Postprocessing complete!")


if __name__ == "__main__":
    main()
