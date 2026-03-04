#!/usr/bin/env python3
"""
Augment per-object geometry with CAD-based surface area and volume estimates.

For each scanned object, maps its label to a ShapeNet category, loads a
representative CAD model, scales it to match the scanned object's bounding box
dimensions, and computes surface area and volume from the scaled CAD mesh.

Objects without a ShapeNet match (structural elements, rare categories) keep
their scanned convex-hull geometry.

Usage:
    python scan2therm/cad_geometry.py \
        --scene_list scan2therm/office_scenes_105.txt \
        --data_dir scan2therm/object_images \
        --shapenet_dir Scan2CAD/Assets/shapenet-sample \
        --out_dir scan2therm/object_images
"""

import os
import os.path as osp
import argparse
import json
import numpy as np
import trimesh

from scan2therm.extract_objects import extract_objects

# ---------------------------------------------------------------------------
# Label → ShapeNet synset mapping
# ---------------------------------------------------------------------------

# Maps lowercase 3RScan / 3DSSG labels to ShapeNet synset IDs.
LABEL_TO_SYNSET = {
    # chairs
    'chair': '03001627',
    'desk chair': '03001627',
    'office chair': '03001627',
    'armchair': '03001627',
    'swivel chair': '03001627',
    'rolling chair': '03001627',
    'gaming chair': '03001627',
    # tables
    'table': '04379243',
    'coffee table': '04379243',
    'couch table': '04379243',
    'side table': '04379243',
    'end table': '04379243',
    'dining table': '04379243',
    'desk': '04379243',
    'computer desk': '04379243',
    'standing desk': '04379243',
    # cabinets
    'cabinet': '02933112',
    'file cabinet': '02933112',
    'kitchen cabinet': '02933112',
    'storage cabinet': '02933112',
    'wardrobe': '02933112',
    'closet': '02933112',
    'dresser': '02933112',
    'nightstand': '02933112',
    'chest of drawers': '02933112',
    # sofas
    'sofa': '04256520',
    'couch': '04256520',
    'sofa chair': '04256520',
    'loveseat': '04256520',
    # beds
    'bed': '02818832',
    # displays
    'monitor': '03211117',
    'tv': '03211117',
    'television': '03211117',
    'screen': '03211117',
    'display': '03211117',
    # lamps
    'lamp': '03636649',
    'table lamp': '03636649',
    'floor lamp': '03636649',
    'desk lamp': '03636649',
    # laptops
    'laptop': '03642806',
    # bookshelves
    'bookshelf': '02871439',
    'shelf': '02871439',
    'shelves': '02871439',
    'bookcase': '02871439',
    # trash bins
    'trash can': '02747177',
    'trash bin': '02747177',
    'bin': '02747177',
    'garbage bin': '02747177',
    'waste bin': '02747177',
    'recycling bin': '02747177',
    # printers
    'printer': '04004475',
    # keyboards
    'keyboard': '03085013',
    # bottles
    'bottle': '02876657',
    # pillows
    'pillow': '03938244',
    'cushion': '03938244',
}

# Structural / background labels — always use scanned geometry
STRUCTURAL_LABELS = {
    'wall', 'floor', 'ceiling', 'door', 'window', 'curtain', 'blinds',
    'pipe', 'beam', 'column', 'railing', 'staircase',
}

# ---------------------------------------------------------------------------
# ShapeNet model cache
# ---------------------------------------------------------------------------


def find_shapenet_model(shapenet_dir, synset_id):
    """Find the first available model for a synset. Returns OBJ path or None."""
    synset_dir = osp.join(shapenet_dir, synset_id)
    if not osp.isdir(synset_dir):
        return None
    for model_id in sorted(os.listdir(synset_dir)):
        obj_path = osp.join(synset_dir, model_id, 'models', 'model_normalized.obj')
        if osp.isfile(obj_path):
            return obj_path
    return None


_cad_cache = {}  # synset_id -> (mesh, area, hull_volume, bbox_extents)


def load_cad_model(shapenet_dir, synset_id):
    """Load and cache a representative CAD model for a synset.

    Returns (trimesh.Trimesh, area, convex_hull_volume, bbox_extents) or None.
    """
    if synset_id in _cad_cache:
        return _cad_cache[synset_id]

    obj_path = find_shapenet_model(shapenet_dir, synset_id)
    if obj_path is None:
        _cad_cache[synset_id] = None
        return None

    try:
        loaded = trimesh.load(obj_path)
        # OBJ files may load as Scene with multiple sub-meshes
        if isinstance(loaded, trimesh.Scene):
            meshes = [g for g in loaded.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                _cad_cache[synset_id] = None
                return None
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded

        area = float(mesh.area)
        # Use convex hull volume since ShapeNet meshes are often non-watertight
        hull_volume = float(mesh.convex_hull.volume)
        bbox_extents = mesh.bounding_box.extents  # (3,)

        result = (mesh, area, hull_volume, bbox_extents)
        _cad_cache[synset_id] = result
        return result
    except Exception as e:
        print(f"  [WARN] Failed to load CAD model {obj_path}: {e}")
        _cad_cache[synset_id] = None
        return None


# ---------------------------------------------------------------------------
# Core: scale CAD model to match scanned OBB, compute geometry
# ---------------------------------------------------------------------------


def compute_cad_geometry(scan_obb, cad_mesh, cad_bbox_extents):
    """Scale CAD model to match scanned OBB and compute area + volume.

    Args:
        scan_obb: [dx, dy, dz] from extract_objects (axis-aligned BB extents)
        cad_mesh: trimesh.Trimesh of the CAD model
        cad_bbox_extents: (3,) bounding box extents of the CAD model

    Returns:
        (surface_area, volume, scale_factors)
    """
    # Match largest-to-largest dimension (anisotropic scaling)
    scan_sorted = np.sort(scan_obb)[::-1]
    cad_sorted = np.sort(cad_bbox_extents)[::-1]

    # Avoid division by zero
    cad_sorted = np.maximum(cad_sorted, 1e-8)
    scale = scan_sorted / cad_sorted

    # Apply anisotropic scale to mesh copy
    # Map scale factors back to the original axis ordering:
    # We sorted both, so scale[i] maps cad's i-th largest to scan's i-th largest.
    # To apply: we need per-axis scale in the CAD model's coordinate frame.
    cad_axis_order = np.argsort(cad_bbox_extents)[::-1]  # indices of largest-first
    per_axis_scale = np.zeros(3)
    for i, axis_idx in enumerate(cad_axis_order):
        per_axis_scale[axis_idx] = scale[i]

    transform = np.diag([*per_axis_scale, 1.0])
    scaled_mesh = cad_mesh.copy()
    scaled_mesh.apply_transform(transform)

    surface_area = float(scaled_mesh.area)
    # Use convex hull volume for non-watertight meshes
    if scaled_mesh.is_watertight:
        volume = float(abs(scaled_mesh.volume))
    else:
        volume = float(scaled_mesh.convex_hull.volume)

    return surface_area, volume, per_axis_scale


# ---------------------------------------------------------------------------
# Process a single scan
# ---------------------------------------------------------------------------


def process_scan(scan_id, data_dir, shapenet_dir, out_dir,
                 csv_path=None, objects_json_path=None):
    """Extract objects, augment with CAD geometry, write objects.json."""

    scan_data_dir = data_dir  # extract_objects expects parent dir containing scan_id/
    # Check if PLY file exists in the scan's output directory
    ply_in_out = osp.join(out_dir, scan_id, 'labels.instances.annotated.v2.ply')
    ply_in_data = osp.join(data_dir, scan_id, 'labels.instances.annotated.v2.ply')

    # Determine which directory has the data
    if osp.isfile(ply_in_out):
        scan_data_dir = out_dir
        label_file = 'labels.instances.annotated.v2.ply'
    elif osp.isfile(osp.join(out_dir, scan_id, 'labels.instances.align.annotated.v2.ply')):
        scan_data_dir = out_dir
        label_file = 'labels.instances.align.annotated.v2.ply'
    elif osp.isfile(ply_in_data):
        scan_data_dir = data_dir
        label_file = 'labels.instances.annotated.v2.ply'
    else:
        print(f"  [SKIP] No PLY found for {scan_id}")
        return None

    try:
        objects = extract_objects(
            data_dir=scan_data_dir,
            scan_id=scan_id,
            label_file=label_file,
            csv_path=csv_path,
            objects_json_path=objects_json_path,
            min_points=50,
            skip_structural=False,  # We want structural too, for completeness
        )
    except Exception as e:
        print(f"  [ERROR] extract_objects failed for {scan_id}: {e}")
        return None

    cad_hit = 0
    cad_miss = 0

    for obj in objects:
        label = obj['label'].lower().strip()

        # Skip structural elements
        if label in STRUCTURAL_LABELS:
            obj['cad_source'] = 'scanned_geometry'
            continue

        synset_id = LABEL_TO_SYNSET.get(label)
        if synset_id is None:
            obj['cad_source'] = 'scanned_geometry'
            cad_miss += 1
            continue

        cad_info = load_cad_model(shapenet_dir, synset_id)
        if cad_info is None:
            obj['cad_source'] = 'scanned_geometry'
            cad_miss += 1
            continue

        cad_mesh, cad_area, cad_hull_vol, cad_extents = cad_info
        scan_obb = np.array(obj['obb_dimensions'])

        # Skip degenerate objects
        if np.any(scan_obb < 1e-4):
            obj['cad_source'] = 'scanned_geometry'
            cad_miss += 1
            continue

        try:
            area, volume, scale = compute_cad_geometry(scan_obb, cad_mesh, cad_extents)
            obj['surface_area_m2'] = round(area, 4)
            obj['volume_m3'] = round(volume, 4)

            # Record which model was used
            model_path = find_shapenet_model(shapenet_dir, synset_id)
            model_id = osp.basename(osp.dirname(osp.dirname(model_path)))
            obj['cad_source'] = f"shapenet:{synset_id}/{model_id}"
            cad_hit += 1
        except Exception as e:
            print(f"  [WARN] CAD geometry failed for obj {obj['object_id']} ({label}): {e}")
            obj['cad_source'] = 'scanned_geometry'
            cad_miss += 1

    # Write output
    out_path = osp.join(out_dir, scan_id, 'objects_cad.json')
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    result = {
        'scan_id': scan_id,
        'num_objects': len(objects),
        'num_cad_augmented': cad_hit,
        'objects': objects,
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  {scan_id}: {len(objects)} objects, {cad_hit} CAD-augmented, {cad_miss} scan-only → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Augment objects with CAD-based surface area and volume')
    parser.add_argument('--scene_list', type=str, required=True,
                        help='Text file with one scan_id per line')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing scan subdirs with PLY + OBJ')
    parser.add_argument('--shapenet_dir', type=str, required=True,
                        help='ShapeNet models directory (synset/model_id/models/)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory (objects.json written per scan)')
    parser.add_argument('--csv_path', type=str,
                        default='3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv',
                        help='3RScan semantic classes CSV')
    parser.add_argument('--objects_json', type=str,
                        default='3DSSG/files/objects.json',
                        help='3DSSG objects.json with material attributes')
    args = parser.parse_args()

    # Load scene list
    with open(args.scene_list) as f:
        scan_ids = [line.strip() for line in f if line.strip()]
    print(f"Processing {len(scan_ids)} scans")

    # Check available ShapeNet categories
    available = set()
    if osp.isdir(args.shapenet_dir):
        for d in os.listdir(args.shapenet_dir):
            if osp.isdir(osp.join(args.shapenet_dir, d)):
                available.add(d)
    needed = set(LABEL_TO_SYNSET.values())
    missing = needed - available
    if missing:
        print(f"WARNING: {len(missing)} ShapeNet categories not found: {missing}")
        print(f"  Available: {available}")
        print(f"  Objects in missing categories will use scanned geometry.")
    print(f"ShapeNet categories available: {available & needed}")

    # Resolve relative paths
    csv_path = args.csv_path if osp.isabs(args.csv_path) else osp.join(os.getcwd(), args.csv_path)
    objects_json = args.objects_json if osp.isabs(args.objects_json) else osp.join(os.getcwd(), args.objects_json)

    total_objects = 0
    total_cad = 0

    for i, scan_id in enumerate(scan_ids):
        print(f"[{i+1}/{len(scan_ids)}] {scan_id}")
        result = process_scan(
            scan_id=scan_id,
            data_dir=args.data_dir,
            shapenet_dir=args.shapenet_dir,
            out_dir=args.out_dir,
            csv_path=csv_path if osp.isfile(csv_path) else None,
            objects_json_path=objects_json if osp.isfile(objects_json) else None,
        )
        if result:
            total_objects += result['num_objects']
            total_cad += result['num_cad_augmented']

    print(f"\nDone. {total_objects} objects total, {total_cad} CAD-augmented.")


if __name__ == '__main__':
    main()
