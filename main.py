import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from geometry_estimator import surface_area, volume


# -----------------------------
# Path defaults (relative)
# -----------------------------
SCAN2THERM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCAN2THERM_DIR.parent

DEFAULT_RSCAN_ROOT = PROJECT_ROOT / "3RScan"
DEFAULT_DSSG_ROOT = PROJECT_ROOT / "3DSSG"
DEFAULT_BEDROOM_JSON = SCAN2THERM_DIR / "office.json"

# ✅ Default scan you asked for
DEFAULT_SCAN_ID = "569d8f0d-72aa-2f24-8ac6-c6ee8d927c4b"


# -----------------------------
# 3RScan: PLY parsing (ASCII)
# -----------------------------
def read_3rscan_instances_ply_ascii(ply_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads 3RScan labels.instances.annotated.v2.ply (ASCII).
    Extracts:
      - vertices (N,3) float64
      - faces (F,3) int64
      - object_ids per vertex (N,) int64 from vertex property 'objectId'
    """
    n_vertices = None
    n_faces = None
    vertex_props: list[str] = []
    in_vertex = False

    with ply_path.open("r", encoding="utf-8", errors="ignore") as f:
        # --- header
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            line = line.strip()

            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
                in_vertex = True
                vertex_props = []
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                # property <type> <name>
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append(parts[-1])
            elif line == "end_header":
                break

        if n_vertices is None or n_faces is None:
            raise ValueError(f"Missing element vertex/face in header: {ply_path}")

        prop_index = {name: i for i, name in enumerate(vertex_props)}
        required = ["x", "y", "z", "objectId"]
        missing = [r for r in required if r not in prop_index]
        if missing:
            raise ValueError(
                f"Missing required vertex properties {missing} in {ply_path}\n"
                f"Found vertex properties: {vertex_props}"
            )

        # --- vertices
        verts = np.zeros((n_vertices, 4), dtype=np.float64)
        for i in range(n_vertices):
            parts = f.readline().strip().split()
            verts[i, 0] = float(parts[prop_index["x"]])
            verts[i, 1] = float(parts[prop_index["y"]])
            verts[i, 2] = float(parts[prop_index["z"]])
            verts[i, 3] = int(parts[prop_index["objectId"]])

        # --- faces (triangles)
        faces = np.zeros((n_faces, 3), dtype=np.int64)
        for i in range(n_faces):
            parts = f.readline().strip().split()
            # format: "3 v0 v1 v2"
            if len(parts) < 4:
                raise ValueError(f"Bad face row #{i} in {ply_path}: {parts}")
            if parts[0] != "3":
                raise ValueError(f"Non-triangle face row #{i} in {ply_path}: {parts[:6]}")
            faces[i, 0] = int(parts[1])
            faces[i, 1] = int(parts[2])
            faces[i, 2] = int(parts[3])

    vertices = verts[:, :3].astype(np.float64)
    object_ids = verts[:, 3].astype(np.int64)
    return vertices, faces, object_ids


def read_3rscan_semseg_labels(semseg_path: Path) -> dict[int, str]:
    """
    semseg.v2.json contains segGroups, each with objectId/id and label.
    """
    if not semseg_path.exists():
        return {}
    data = json.loads(semseg_path.read_text(encoding="utf-8"))
    seg_groups = data.get("segGroups") or []
    out: dict[int, str] = {}
    for g in seg_groups:
        oid = g.get("objectId", g.get("id", -1))
        if oid is None:
            continue
        out[int(oid)] = str(g.get("label", "?"))
    return out


def find_one_scan_dir_3rscan(rscan_root: Path, scan_id: str | None = None) -> Path:
    """
    Handles both:
      rscan_root/3RScan_Data/<scan_id>/labels.instances.annotated.v2.ply
    and:
      rscan_root/<scan_id>/labels.instances.annotated.v2.ply
    """
    data_root = rscan_root / "3RScan_Data" if (rscan_root / "3RScan_Data").exists() else rscan_root

    if scan_id:
        d = data_root / scan_id
        ply = d / "labels.instances.annotated.v2.ply"
        if not ply.exists():
            raise FileNotFoundError(f"Missing {ply}")
        return d

    # auto-pick first scan that contains the required ply
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and (d / "labels.instances.annotated.v2.ply").exists():
            return d

    raise FileNotFoundError(f"No scan directory found under {data_root} containing labels.instances.annotated.v2.ply")


# -----------------------------
# 3DSSG: ground truth materials
# -----------------------------
def find_objects_json_3dssg(dssg_root: Path) -> Path:
    candidates = [
        dssg_root / "objects.json",
        dssg_root / "3DSSG" / "objects.json",
        dssg_root / "3DSSG" / "3DSSG" / "objects.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    for p in dssg_root.rglob("objects.json"):
        return p

    raise FileNotFoundError(f"Could not find objects.json anywhere under: {dssg_root}")


def load_3dssg_objects(objects_json_path: Path) -> dict[str, dict[int, dict]]:
    data = json.loads(objects_json_path.read_text(encoding="utf-8"))
    scans = data.get("scans") or []
    out: dict[str, dict[int, dict]] = {}

    for s in scans:
        sid = str(s.get("scan"))
        inst_map: dict[int, dict] = {}
        for o in (s.get("objects") or []):
            oid = o.get("id")
            if oid is None:
                continue
            try:
                inst_map[int(oid)] = o
            except Exception:
                continue
        out[sid] = inst_map

    return out


def normalize_material_category(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    r = raw.strip().lower()
    if any(k in r for k in ["wood", "wooden", "timber", "bamboo"]):
        return "Wood"
    if any(k in r for k in ["concrete", "cement"]):
        return "Concrete"
    if "glass" in r:
        return "Glass"
    if any(k in r for k in ["metal", "steel", "iron", "aluminum", "aluminium", "brass"]):
        return "Metal"
    if any(k in r for k in ["plastic", "acrylic", "vinyl"]):
        return "Plastic"
    if any(k in r for k in ["fabric", "textile", "cloth", "leather"]):
        return "Textile"
    if any(k in r for k in ["ceramic", "tile", "porcelain"]):
        return "Ceramic/Tile"
    if any(k in r for k in ["stone", "marble", "granite"]):
        return "Stone"
    if any(k in r for k in ["paper", "cardboard"]):
        return "Paper"
    return "Other"


def oid_to_raw_material(scan_obj_map: dict[int, dict]) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    for oid, rec in scan_obj_map.items():
        attrs = rec.get("attributes") or {}
        mats = attrs.get("material")
        raw = None
        if isinstance(mats, list) and mats:
            raw = str(mats[0])
        elif isinstance(mats, str):
            raw = mats
        out[int(oid)] = raw
    return out


# -----------------------------
# Bedroom JSON update
# -----------------------------
def ensure_zone0(doc: dict) -> None:
    if "Zones" not in doc or not isinstance(doc["Zones"], list) or len(doc["Zones"]) == 0:
        doc["Zones"] = [
            {
                "Zone_ID": "Z001",
                "Zone_Name": "Zone_1",
                "Zone_Properties": {},
                "Objects": [],
            }
        ]
    doc["Zones"][0].setdefault("Objects", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rscan_root", type=str, default=str(DEFAULT_RSCAN_ROOT))
    ap.add_argument("--dssg_root", type=str, default=str(DEFAULT_DSSG_ROOT))
    ap.add_argument("-office_json", type=str, default=str(DEFAULT_BEDROOM_JSON))

    # ✅ easy scan switching:
    # - default is your scan id
    # - you can override via --scan <other_scan_id>
    ap.add_argument(
        "--scan",
        type=str,
        default=DEFAULT_SCAN_ID,
        help=f"Scan folder name. Default: {DEFAULT_SCAN_ID}. Override: --scan <scan_id>",
    )

    ap.add_argument("--out", type=str, default=None, help="Output path. Default: office_updated.json next to o.json")
    ap.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=["wall", "floor", "ceiling"],
        help="Lowercase labels to skip",
    )
    args = ap.parse_args()

    rscan_root = Path(args.rscan_root)
    dssg_root = Path(args.dssg_root)
    office_json_path = Path(args.office_json)

    if not rscan_root.exists():
        raise FileNotFoundError(f"3RScan root not found: {rscan_root}")
    if not dssg_root.exists():
        raise FileNotFoundError(f"3DSSG root not found: {dssg_root}")
    if not office_json_path.exists():
        raise FileNotFoundError(f"office.json not found: {office_json_path}")

    # --- pick scan (defaults to your scan id)
    scan_dir = find_one_scan_dir_3rscan(rscan_root, args.scan)
    scan_id = scan_dir.name

    ply_path = scan_dir / "labels.instances.annotated.v2.ply"
    semseg_path = scan_dir / "semseg.v2.json"

    # --- read geometry
    vertices, faces, object_ids = read_3rscan_instances_ply_ascii(ply_path)

    # --- compute using your imported geometry_estimator.py
    sa_by_oid = surface_area(vertices, faces, object_ids)
    vol_by_oid = volume(vertices, faces, object_ids)

    # --- labels
    labels_3rscan = read_3rscan_semseg_labels(semseg_path)

    # --- 3DSSG materials
    objects_json_path = find_objects_json_3dssg(dssg_root)
    dssg = load_3dssg_objects(objects_json_path)
    scan_obj_map = dssg.get(scan_id, {})
    mats_by_oid = oid_to_raw_material(scan_obj_map)

    # if semseg missing label but 3DSSG has it, fill it
    for oid, rec in scan_obj_map.items():
        if oid not in labels_3rscan:
            labels_3rscan[oid] = str(rec.get("label", "?"))

    # --- build object list for JSON
    skip = {s.lower() for s in (args.skip or [])}
    objects_out = []
    for oid in sorted(sa_by_oid.keys()):
        label = (labels_3rscan.get(oid) or "?").strip()
        if label.lower() in skip:
            continue

        raw_mat = mats_by_oid.get(oid)
        objects_out.append(
            {
                "ID": f"scanobj_{scan_id}_{oid:04d}",
                "Type": label,
                "Scan_Object_ID": int(oid),
                "Geometry": {
                    "Surface_Area_m2": float(sa_by_oid.get(oid, 0.0)),
                    "Volume_m3": float(vol_by_oid.get(oid, 0.0)),
                },
                "Material_Info": {
                    "GroundTruth": raw_mat if raw_mat is not None else None,
                    "Category": normalize_material_category(raw_mat),
                    "Source": "3DSSG.attributes.material",
                },
            }
        )

    # --- update office.json
    doc = json.loads(office_json_path.read_text(encoding="utf-8"))

    doc.setdefault("Metadata", {})
    doc["Metadata"]["Date"] = datetime.now().strftime("%Y-%m-%d")
    doc["Metadata"]["Description"] = f"Auto-filled from 3RScan scan {scan_id} with 3DSSG GT materials"
    doc["Metadata"]["Source_Scan_ID"] = scan_id
    doc["Metadata"]["Sources"] = {
        "3RScan_root": str(rscan_root),
        "3DSSG_root": str(dssg_root),
        "3DSSG_objects_json": str(objects_json_path),
        "3RScan_scan_dir": str(scan_dir),
        "3RScan_instances_ply": str(ply_path),
        "3RScan_semseg_json": str(semseg_path) if semseg_path.exists() else None,
    }

    ensure_zone0(doc)
    doc["Zones"][0]["Objects"] = objects_out

    out_path = Path(args.out) if args.out else office_json_path.with_name("office_updated.json")
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"[OK] scan_id={scan_id}")
    print(f"[OK] objects_written={len(objects_out)}")
    print(f"[OK] wrote={out_path}")

    if scan_id not in dssg:
        print("[WARN] scan_id not found in your 3DSSG objects.json extract; materials likely Unknown.")


if __name__ == "__main__":
    main()