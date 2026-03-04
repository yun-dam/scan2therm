#!/usr/bin/env python3
"""
Extract objects from a 3RScan scene with material labels and surface areas.
Uses 3DSSG annotations (material, color, shape, affordances) when available.

Usage:
    python scan2therm/extract_objects.py \
        --data_dir /path/to/3RScan/scans \
        --scan_id 0a4b8ef6-...
"""

import os.path as osp
import argparse
import json
import csv
import numpy as np
import trimesh

from scan2therm.scan3r_utils import load_ply_data
from scan2therm.point_cloud_utils import get_object_loc_box

# ---- NYU40 ID → fallback material (used when 3DSSG has no annotation) ----

NYU40_MATERIAL = {
    1: ('wall', 'concrete'),
    2: ('floor', 'concrete'),
    3: ('cabinet', 'wooden'),
    4: ('bed', 'padded'),
    5: ('chair', 'wooden'),
    6: ('sofa', 'padded'),
    7: ('table', 'wooden'),
    8: ('door', 'wooden'),
    9: ('window', 'glass'),
    10: ('bookshelf', 'wooden'),
    11: ('picture', 'glass'),
    12: ('counter', 'stone'),
    13: ('blinds', 'plastic'),
    14: ('desk', 'wooden'),
    15: ('shelves', 'wooden'),
    16: ('curtain', 'carpet'),
    17: ('dresser', 'wooden'),
    18: ('pillow', 'padded'),
    19: ('mirror', 'glass'),
    20: ('floor_mat', 'carpet'),
    21: ('clothes', 'carpet'),
    22: ('ceiling', 'concrete'),
    23: ('books', 'cardboard'),
    24: ('refrigerator', 'metal'),
    25: ('television', 'plastic'),
    26: ('paper', 'cardboard'),
    27: ('towel', 'carpet'),
    28: ('shower_curtain', 'plastic'),
    29: ('box', 'cardboard'),
    30: ('whiteboard', 'plastic'),
    31: ('person', None),
    32: ('nightstand', 'wooden'),
    33: ('toilet', 'ceramic'),
    34: ('sink', 'ceramic'),
    35: ('lamp', 'plastic'),
    36: ('bathtub', 'ceramic'),
    37: ('bag', 'carpet'),
    38: ('otherstructure', 'concrete'),
    39: ('otherfurniture', 'wooden'),
    40: ('otherprop', 'plastic'),
}

STRUCTURAL_NYU40 = {1, 2, 22}  # wall, floor, ceiling


def load_global_id_to_label(csv_path):
    """Load globalId → label name from 3RScan mapping CSV."""
    mapping = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 3 or len(row) < 2:
                continue
            try:
                mapping[int(row[0].strip())] = row[1].strip()
            except (ValueError, IndexError):
                continue
    return mapping


def load_3dssg_objects(objects_json_path, scan_id):
    """Load 3DSSG object attributes for a scan.

    Returns:
        {object_id: {label, material, color, shape, affordances, ...}} or empty dict.
    """
    if not objects_json_path or not osp.isfile(objects_json_path):
        return {}

    with open(objects_json_path) as f:
        data = json.load(f)

    for scan in data['scans']:
        if scan['scan'] == scan_id:
            obj_map = {}
            for obj in scan['objects']:
                obj_id = int(obj['id'])
                attrs = obj.get('attributes', {})
                obj_map[obj_id] = {
                    'label': obj.get('label', ''),
                    'nyu40': obj.get('nyu40', ''),
                    'global_id': obj.get('global_id', ''),
                    'material': attrs.get('material', []),
                    'color': attrs.get('color', []),
                    'shape': attrs.get('shape', []),
                    'texture': attrs.get('texture', []),
                    'state': attrs.get('state', []),
                    'affordances': obj.get('affordances', []),
                }
            return obj_map
    return {}


def compute_surface_area(points):
    """Convex hull surface area, OBB fallback."""
    try:
        hull = trimesh.PointCloud(points).convex_hull
        return float(hull.area)
    except Exception:
        dims = points.max(axis=0) - points.min(axis=0)
        dx, dy, dz = dims
        return float(2.0 * (dx*dy + dy*dz + dx*dz))


def extract_objects(data_dir, scan_id, label_file='labels.instances.align.annotated.v2.ply',
                    csv_path=None, objects_json_path=None, min_points=50, skip_structural=True):
    """Extract objects with material and surface area from a scan."""

    vertices = load_ply_data(data_dir, scan_id, label_file)

    # Optional: richer label names from CSV
    gid_to_label = {}
    if csv_path and osp.isfile(csv_path):
        gid_to_label = load_global_id_to_label(csv_path)

    # 3DSSG attributes (material, color, shape, etc.)
    dssg_objects = load_3dssg_objects(objects_json_path, scan_id)

    unique_ids = np.unique(vertices['objectId'])
    objects = []

    for obj_id in unique_ids:
        if obj_id == 0:
            continue

        mask = vertices['objectId'] == obj_id
        count = int(mask.sum())
        if count < min_points:
            continue

        idx = np.where(mask)[0][0]
        nyu40_id = int(vertices['NYU40'][idx])
        global_id = int(vertices['globalId'][idx])

        if skip_structural and nyu40_id in STRUCTURAL_NYU40:
            continue

        nyu40_name, fallback_material = NYU40_MATERIAL.get(nyu40_id, ('unknown', 'unknown'))
        if fallback_material is None:
            continue  # e.g. person

        # Use CSV label if available, else NYU40 class name
        label = gid_to_label.get(global_id, nyu40_name)

        # 3DSSG attributes for this object
        dssg = dssg_objects.get(int(obj_id), {})
        if dssg.get('label'):
            label = dssg['label']

        # Material: prefer 3DSSG annotation, fallback to NYU40-based guess
        dssg_material = dssg.get('material', [])
        material = dssg_material[0] if dssg_material else fallback_material
        material_source = '3dssg' if dssg_material else 'nyu40_fallback'

        points = np.column_stack([
            vertices['x'][mask],
            vertices['y'][mask],
            vertices['z'][mask],
        ])

        obj_loc, obj_box = get_object_loc_box(points)
        surface_area = compute_surface_area(points)

        entry = {
            'object_id': int(obj_id),
            'label': label,
            'nyu40_class': nyu40_name,
            'material': material,
            'material_source': material_source,
            'surface_area_m2': round(surface_area, 4),
            'obb_dimensions': [round(d, 4) for d in obj_box[3:6].tolist()],
            'centroid': [round(c, 4) for c in obj_loc[:3].tolist()],
            'num_points': count,
        }

        # Add 3DSSG attributes if available
        if dssg:
            extra = {}
            if dssg.get('color'):
                extra['color'] = dssg['color']
            if dssg.get('shape'):
                extra['shape'] = dssg['shape']
            if dssg.get('texture'):
                extra['texture'] = dssg['texture']
            if dssg.get('state'):
                extra['state'] = dssg['state']
            if dssg.get('affordances'):
                extra['affordances'] = dssg['affordances']
            if extra:
                entry['attributes'] = extra

        objects.append(entry)

    return objects


def main():
    parser = argparse.ArgumentParser(description='Extract objects with materials and surface areas from a 3RScan scene')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='3RScan scans directory (contains scan_id subdirs)')
    parser.add_argument('--scan_id', type=str, required=True,
                        help='Scan ID (e.g., 0a4b8ef6-...)')
    parser.add_argument('--csv_path', type=str,
                        default='/home/yunda/CrossOver/3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv',
                        help='3RScan semantic classes CSV for label names')
    parser.add_argument('--objects_json', type=str,
                        default='/home/yunda/CrossOver/3DSSG/files/objects.json',
                        help='3DSSG objects.json with material/color/shape attributes')
    parser.add_argument('--min_points', type=int, default=50,
                        help='Minimum points to include an object')
    parser.add_argument('--label_file', type=str,
                        default='labels.instances.align.annotated.v2.ply',
                        help='PLY annotation filename (use labels.instances.annotated.v2.ply if unaligned)')
    parser.add_argument('--include_structural', action='store_true',
                        help='Include walls, floors, ceilings')
    parser.add_argument('--output', type=str, default='',
                        help='Output JSON file path (default: print to stdout)')
    args = parser.parse_args()

    objects = extract_objects(
        data_dir=args.data_dir,
        scan_id=args.scan_id,
        label_file=args.label_file,
        csv_path=args.csv_path,
        objects_json_path=args.objects_json,
        min_points=args.min_points,
        skip_structural=not args.include_structural,
    )

    # Summary stats
    total_with_3dssg = sum(1 for o in objects if o['material_source'] == '3dssg')

    result = {
        'scan_id': args.scan_id,
        'num_objects': len(objects),
        'num_with_3dssg_material': total_with_3dssg,
        'objects': objects,
    }

    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Wrote {len(objects)} objects ({total_with_3dssg} with 3DSSG material) to {args.output}")
    else:
        print(output_json)


if __name__ == '__main__':
    main()
