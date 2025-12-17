#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse
import csv

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False


def extract_part_ids_from_mesh(mesh) -> np.ndarray:
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


def load_mesh_classes(ply_path: Path) -> tuple[int, str]:
    if HAS_TRIMESH:
        try:
            tm = trimesh.load(str(ply_path))
            if not isinstance(tm, trimesh.Trimesh):
                return None, f"Not a triangle mesh"
            
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
            
            mesh = meshio.Mesh(
                tm.vertices,
                [("triangle", tm.faces)],
                cell_data=cell_data
            )
        except Exception as e:
            if HAS_MESHIO:
                try:
                    mesh = meshio.read(ply_path)
                except Exception:
                    return None, f"Failed to read: {e}"
            else:
                return None, f"Failed to read: {e}"
    else:
        if not HAS_MESHIO:
            return None, "Neither trimesh nor meshio available"
        mesh = meshio.read(ply_path)
    
    part_ids = extract_part_ids_from_mesh(mesh)
    num_classes = len(np.unique(part_ids))
    
    return num_classes, None


def check_mismatches(input_dir: Path, folder_name: str) -> list[dict]:
    input_folder = input_dir / folder_name
    
    if not input_folder.exists():
        return []
    
    resolution_folders = ["coarser_meshes240", "default_meshes240", "finer_meshes240"]
    if folder_name == "partfield":
        resolution_folders = ["coarser_partfield240", "default_partfield240", "finer_partfield240"]
    
    file_groups = {}
    
    for res_folder in resolution_folders:
        res_path = input_folder / res_folder
        if not res_path.exists():
            continue
        
        for ply_file in sorted(res_path.glob("*.ply")):
            filename = ply_file.name
            base_name = filename.replace("_labels.ply", "").replace(".ply", "")
            
            if base_name not in file_groups:
                file_groups[base_name] = {}
            
            file_groups[base_name][res_folder] = ply_file
    
    mismatches = []
    
    for base_name, group_files in file_groups.items():
        if len(group_files) != 3:
            continue
        
        class_counts = {}
        errors = {}
        
        for res_folder in resolution_folders:
            if res_folder in group_files:
                ply_path = group_files[res_folder]
                num_classes, error = load_mesh_classes(ply_path)
                if error:
                    errors[res_folder] = error
                else:
                    class_counts[res_folder] = num_classes
        
        if len(errors) > 0:
            mismatches.append({
                "part": base_name,
                "type": "error",
                "errors": errors,
                "class_counts": class_counts
            })
        elif len(class_counts) == 3:
            counts = list(class_counts.values())
            if len(set(counts)) > 1:
                mismatches.append({
                    "part": base_name,
                    "type": "mismatch",
                    "class_counts": class_counts
                })
    
    return mismatches


def export_to_csv(mismatches: list[dict], folder_name: str, output_dir: Path) -> None:
    csv_path = output_dir / f"class_mismatches_{folder_name}.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['part', 'type', 'coarser_classes', 'default_classes', 'finer_classes', 'errors']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for m in mismatches:
            row = {
                'part': m['part'],
                'type': m['type']
            }
            
            if m['type'] == 'mismatch':
                counts = m['class_counts']
                row['coarser_classes'] = counts.get('coarser_meshes240') or counts.get('coarser_partfield240', '')
                row['default_classes'] = counts.get('default_meshes240') or counts.get('default_partfield240', '')
                row['finer_classes'] = counts.get('finer_meshes240') or counts.get('finer_partfield240', '')
                row['errors'] = ''
            else:
                counts = m.get('class_counts', {})
                row['coarser_classes'] = counts.get('coarser_meshes240') or counts.get('coarser_partfield240', '')
                row['default_classes'] = counts.get('default_meshes240') or counts.get('default_partfield240', '')
                row['finer_classes'] = counts.get('finer_meshes240') or counts.get('finer_partfield240', '')
                errors = m.get('errors', {})
                error_str = '; '.join([f"{k}: {v}" for k, v in errors.items()])
                row['errors'] = error_str
            
            writer.writerow(row)
    
    print(f"  Exported CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check for semantic class count mismatches across tesselation levels"
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
        default=Path("."),
        help="Output directory for CSV files"
    )
    
    args = parser.parse_args()
    
    input_root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for folder_name in ["ours", "partfield"]:
        print(f"\n{'='*60}")
        print(f"Checking {folder_name}")
        print(f"{'='*60}")
        
        mismatches = check_mismatches(input_root, folder_name)
        
        if len(mismatches) == 0:
            print(f"No mismatches found in {folder_name}")
            export_to_csv([], folder_name, output_dir)
            continue
        
        mismatch_parts = [m for m in mismatches if m["type"] == "mismatch"]
        error_parts = [m for m in mismatches if m["type"] == "error"]
        
        if mismatch_parts:
            print(f"\nFound {len(mismatch_parts)} parts with class count mismatches:")
            for m in mismatch_parts:
                counts = m["class_counts"]
                print(f"  {m['part']}:")
                for res, count in sorted(counts.items()):
                    print(f"    {res}: {count} classes")
        
        if error_parts:
            print(f"\nFound {len(error_parts)} parts with errors:")
            for m in error_parts:
                print(f"  {m['part']}:")
                for res, error in m["errors"].items():
                    print(f"    {res}: {error}")
                if m["class_counts"]:
                    print(f"    (partial counts: {m['class_counts']})")
        
        export_to_csv(mismatches, folder_name, output_dir)
    
    print("\n" + "="*60)
    print("Check complete!")


if __name__ == "__main__":
    main()

