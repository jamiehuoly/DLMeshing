import gmsh
import numpy as np

def gmsh_tag_transform():
    # nodeTags: node ID list
    # nodeCoords: [x1, y1, z1, x2, y2, z2, ...] flattened list
    node_tag, coordinate, _ = gmsh.model.mesh.getNodes()

    # reshape to 3-dimension
    coordinate = np.array(coordinate, dtype=np.float32).reshape(len(node_tag), 3)

    # Mirror relation： Gmsh tag -> Python index; Gmsh tag may not continuous
    node_index = {tag: i for i, tag in enumerate(node_tag)}

    elementTypes, elementTags, nodeTagsPerEl = gmsh.model.mesh.getElements(dim=3)
    # print(gmsh.model.mesh.getElements(dim=3))
    # print(type(elementTypes))
    # print(elementTags)
    # print(nodeTagsPerEl)

    # 4 means tetrahedral (4 nodes)
    if 4 not in elementTypes:
        raise ValueError("No tetrahedron found! Please check if 3D mesh was generated.")

    tetra_index = elementTypes.tolist().index(4)
    tetra_nodes_tags = np.array(nodeTagsPerEl[tetra_index], dtype=np.int64).reshape(-1, 4)
    print(tetra_nodes_tags)

    # mirroring gmsh coordinate into pytorch coordinate
    elems_numpy = np.zeros_like(tetra_nodes_tags)

    for i in range(tetra_nodes_tags.shape[0]):
        for j in range(tetra_nodes_tags.shape[1]):
            gmsh_tag = tetra_nodes_tags[i, j]
            python_index = node_index[gmsh_tag]
            elems_numpy[i, j] = python_index

    print(f"Finished transferring: {coordinate.shape[0]} Nodes, {elems_numpy.shape[0]} Tetras")
    return coordinate, elems_numpy