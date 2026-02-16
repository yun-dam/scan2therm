import numpy as np
import json
from pathlib import Path

def surface_area(vertices: np.ndarray, faces: np.ndarray, object_ids: np.ndarray | None = None):
    """Sum of triangle areas. With object_ids -> dict[oid, area]."""
    v = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    if object_ids is None:
        return float(areas.sum())
    face_oid = np.array([np.bincount(row).argmax() for row in object_ids[faces]])
    return {int(oid): float(areas[face_oid == oid].sum()) for oid in np.unique(face_oid)}


def volume(vertices: np.ndarray, faces: np.ndarray, object_ids: np.ndarray | None = None):
    """Signed tetrahedron volume with centroid as ref. With object_ids -> dict[oid, volume]."""
    v = vertices[faces]
    if object_ids is None:
        p = v - vertices.mean(axis=0)
        return float((1.0 / 6.0) * np.einsum("fi,fi->f", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum())
    face_oid = np.array([np.bincount(row).argmax() for row in object_ids[faces]])
    out = {}
    for oid in np.unique(face_oid):
        mask = face_oid == oid
        ref = vertices[object_ids == oid].mean(axis=0) if (object_ids == oid).any() else v[mask].mean(axis=(0, 1))
        p = v[mask] - ref
        out[int(oid)] = float((1.0 / 6.0) * np.einsum("fi,fi->f", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum())
    return out

# Small demo on one pointclud below
DATA = Path(__file__).resolve().parent / "3RScan_Data"
ply_path = next((d / "labels.instances.annotated.v2.ply" for d in DATA.iterdir()
                    if d.is_dir() and (d / "labels.instances.annotated.v2.ply").exists()), None)
if not ply_path:
    raise FileNotFoundError("No 3RScan_Data/*/labels.instances.annotated.v2.ply")
scan = ply_path.parent.name

n_vertices = n_faces = 0
with open(ply_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
        elif line == "end_header":
            break
    verts = np.array([[float(p[0]), float(p[1]), float(p[2]), int(p[6])] for p in (next(f).split() for _ in range(n_vertices))])
    faces = np.array([[int(p[1]), int(p[2]), int(p[3])] for p in (next(f).split() for _ in range(n_faces))])
vertices = verts[:, :3].astype(np.float64)
object_ids = verts[:, 3].astype(np.int64)

labels = {int(g.get("objectId", g.get("id", -1))): g.get("label", "?") for g in (json.load(open(DATA / scan / "semseg.v2.json")).get("segGroups") or [])} if (DATA / scan / "semseg.v2.json").exists() else {}
sa, vol = surface_area(vertices, faces, object_ids), volume(vertices, faces, object_ids)
skip = {"wall", "floor", "ceiling"}
oid = max((i for i in vol if labels.get(i, "").lower() not in skip), key=vol.get, default=max(vol, key=vol.get))
print(f"{labels.get(oid, oid)}: surface_area = {sa[oid]:.6f} m², volume = {vol[oid]:.6f} m³")
