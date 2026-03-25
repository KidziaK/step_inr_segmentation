from src.process_single_mesh import process_single_mesh
from pathlib import Path
import fire

def process_folder(input_dir: str, output_dir: str):
    for mesh_path in Path(input_dir).glob("*.ply"):
        output_path = Path(output_dir).joinpath(mesh_path.name)
        process_single_mesh(str(mesh_path), str(output_path))


if __name__ == "__main__":
    fire.Fire(process_folder)
