import meshio


def read_mesh(path: str) -> meshio.Mesh:
    return meshio.read(path)


def write_mesh(mesh: meshio.Mesh, path: str) -> None:
    mesh.write(str(path), binary=True)