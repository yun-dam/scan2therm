#!/usr/bin/env python3
"""
Visualize scanned point clouds vs. retrieved CAD models side-by-side.

Generates one HTML file per sample with 3D scatter/mesh views,
plus the 2D object image and surface area comparison.

Usage:
    # One HTML per sample for top 10:
    python scan2therm/visualize_scan_vs_cad.py \
        --base_dir scan2therm/object_images \
        --gates_dir scan2therm/pipeline_v3_gates/object_images \
        --shapenet_dir ShapeNetCore \
        --out_dir scan2therm/scan_vs_cad_html

    # Range: samples 20-28 (0-indexed from sorted results):
    python scan2therm/visualize_scan_vs_cad.py \
        --range 20-28 --out_dir scan2therm/scan_vs_cad_html

    # Single sample by index:
    python scan2therm/visualize_scan_vs_cad.py \
        --range 5 --out_dir scan2therm/scan_vs_cad_html
"""

import os
import os.path as osp
import json
import argparse
import glob
import numpy as np
import trimesh
import plotly.graph_objects as go
import base64
from pathlib import Path

import sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from scan2therm.scan3r_utils import load_ply_data
from scan2therm.cad_geometry import STRUCTURAL_LABELS



def load_scanned_points(base_dir, scan_id, object_id):
    """Load point cloud for a specific object from the PLY file."""
    for label_file in [
        'labels.instances.annotated.v2.ply',
        'labels.instances.align.annotated.v2.ply',
    ]:
        ply_path = osp.join(base_dir, scan_id, label_file)
        if osp.isfile(ply_path):
            vertices = load_ply_data(base_dir, scan_id, label_file)
            mask = vertices['objectId'] == object_id
            if mask.sum() < 10:
                return None
            points = np.column_stack([
                vertices['x'][mask],
                vertices['y'][mask],
                vertices['z'][mask],
            ])
            # Try to get colors
            colors = None
            if 'red' in vertices.dtype.names:
                colors = np.column_stack([
                    vertices['red'][mask],
                    vertices['green'][mask],
                    vertices['blue'][mask],
                ])
            return {'points': points, 'colors': colors}
    return None


def load_cad_mesh(shapenet_dir, cad_source, obb_dimensions):
    """Load and scale CAD mesh to match scanned OBB."""
    # Parse cad_source: "shapenet:SYNSET/MODEL_ID"
    parts = cad_source.replace('shapenet:', '').split('/')
    if len(parts) < 2:
        return None
    synset_id, model_id = parts[0], parts[1]

    obj_path = osp.join(shapenet_dir, synset_id, model_id,
                        'models', 'model_normalized.obj')
    if not osp.isfile(obj_path):
        return None

    loaded = trimesh.load(obj_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            return None
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    # Scale to match scanned OBB (largest-to-largest)
    scan_sorted = np.sort(obb_dimensions)[::-1]
    cad_sorted = np.sort(mesh.bounding_box.extents)[::-1]

    safe_cad = np.where(cad_sorted < 1e-6, 1e-6, cad_sorted)
    scale = scan_sorted / safe_cad

    # Apply anisotropic scaling (match dimension order)
    cad_extent_order = np.argsort(mesh.bounding_box.extents)[::-1]
    per_axis_scale = np.zeros(3)
    for i, axis in enumerate(cad_extent_order):
        per_axis_scale[axis] = scale[i]

    transform = np.diag([*per_axis_scale, 1.0])
    scaled = mesh.copy()
    scaled.apply_transform(transform)

    # Center at origin
    scaled.apply_translation(-scaled.centroid)

    return scaled


def get_object_image_base64(base_dir, scan_id, label):
    """Find and encode object image as base64 for embedding in HTML."""
    label_lower = label.lower().replace(' ', '_')
    patterns = [
        osp.join(base_dir, scan_id, f'{label_lower}*.jpg'),
        osp.join(base_dir, scan_id, f'{label_lower}*.png'),
    ]
    for pat in patterns:
        imgs = sorted(glob.glob(pat))
        if imgs:
            with open(imgs[0], 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            ext = osp.splitext(imgs[0])[1].lower()
            mime = 'image/jpeg' if ext == '.jpg' else 'image/png'
            return f'data:{mime};base64,{data}', osp.basename(imgs[0])
    return None, None


def make_pointcloud_trace(points, colors=None, name='Scanned', color='steelblue'):
    """Create a plotly Scatter3d trace for a point cloud."""
    if colors is not None:
        marker_color = [f'rgb({r},{g},{b})' for r, g, b in colors]
    else:
        marker_color = color

    # Subsample if too many points
    if len(points) > 5000:
        idx = np.random.choice(len(points), 5000, replace=False)
        points = points[idx]
        if isinstance(marker_color, list):
            marker_color = [marker_color[i] for i in idx]

    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=marker_color, opacity=0.8),
        name=name,
    )


def make_mesh_trace(mesh, name='CAD', color='orange', opacity=0.6):
    """Create a plotly Mesh3d trace from a trimesh mesh."""
    vertices = mesh.vertices
    faces = mesh.faces

    # Subsample faces if too many
    if len(faces) > 50000:
        idx = np.random.choice(len(faces), 50000, replace=False)
        faces = faces[idx]

    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity,
        name=name,
        flatshading=True,
    )


def find_comparison_samples(base_dir, gates_dir):
    """Find objects where CAD area > scanned area, ranked by ratio."""
    results = []

    for gates_json in sorted(glob.glob(
            osp.join(gates_dir, '*', 'objects_cad.json'))):
        scan_id = osp.basename(osp.dirname(gates_json))
        base_json = osp.join(base_dir, scan_id, 'objects.json')
        if not osp.isfile(base_json):
            continue

        with open(base_json) as f:
            base = json.load(f)
        with open(gates_json) as f:
            gates = json.load(f)

        base_objs = {o['object_id']: o for o in base['objects']}
        gates_objs = {o['object_id']: o for o in gates['objects']}

        for oid, g_obj in gates_objs.items():
            b_obj = base_objs.get(oid)
            if not b_obj:
                continue

            label = g_obj['label'].lower().strip()
            if label in STRUCTURAL_LABELS:
                continue

            cad_src = g_obj.get('cad_source', '')
            if 'shapenet' not in cad_src and 'crossover' not in cad_src:
                continue

            b_area = b_obj.get('surface_area_m2', 0)
            g_area = g_obj.get('surface_area_m2', 0)

            if b_area > 0.05 and g_area > b_area:
                # Check image exists
                img_data, img_name = get_object_image_base64(
                    base_dir, scan_id, g_obj['label'])
                if not img_data:
                    continue

                results.append({
                    'scan_id': scan_id,
                    'object_id': oid,
                    'label': g_obj['label'],
                    'scanned_area': b_area,
                    'cad_area': g_area,
                    'ratio': g_area / b_area,
                    'obb_dimensions': g_obj['obb_dimensions'],
                    'cad_source': cad_src,
                    'num_points': b_obj.get('num_points', 0),
                    'img_data': img_data,
                    'img_name': img_name,
                })

    results.sort(key=lambda x: x['ratio'], reverse=True)

    # Pick one per label (best ratio for each), then fill remaining
    seen = {}
    for r in results:
        key = r['label'].lower()
        if key not in seen:
            seen[key] = r

    diverse = sorted(seen.values(), key=lambda x: x['ratio'], reverse=True)

    # If not enough unique labels, add more of the same
    for r in results:
        if r not in diverse:
            diverse.append(r)

    return diverse


def find_specific_objects(base_dir, gates_dir, scan_id, object_ids=None):
    """Find specific objects from a scan for visualization."""
    gates_json = osp.join(gates_dir, scan_id, 'objects_cad.json')
    base_json = osp.join(base_dir, scan_id, 'objects.json')

    if not osp.isfile(gates_json):
        print(f"No objects_cad.json for scan {scan_id}")
        return []
    if not osp.isfile(base_json):
        print(f"No objects.json for scan {scan_id}")
        return []

    with open(base_json) as f:
        base = json.load(f)
    with open(gates_json) as f:
        gates = json.load(f)

    base_objs = {o['object_id']: o for o in base['objects']}
    gates_objs = {o['object_id']: o for o in gates['objects']}

    results = []
    for oid, g_obj in gates_objs.items():
        if object_ids and oid not in object_ids:
            continue

        b_obj = base_objs.get(oid)
        if not b_obj:
            continue

        label = g_obj['label'].lower().strip()
        if label in STRUCTURAL_LABELS:
            continue

        cad_src = g_obj.get('cad_source', '')
        if 'shapenet' not in cad_src and 'crossover' not in cad_src:
            continue

        b_area = b_obj.get('surface_area_m2', 0)
        g_area = g_obj.get('surface_area_m2', 0)

        img_data, img_name = get_object_image_base64(
            base_dir, scan_id, g_obj['label'])

        results.append({
            'scan_id': scan_id,
            'object_id': oid,
            'label': g_obj['label'],
            'scanned_area': b_area,
            'cad_area': g_area,
            'ratio': g_area / b_area if b_area > 0 else 0,
            'obb_dimensions': g_obj['obb_dimensions'],
            'cad_source': cad_src,
            'num_points': b_obj.get('num_points', 0),
            'img_data': img_data,
            'img_name': img_name,
        })

    results.sort(key=lambda x: x['object_id'])
    return results


def build_single_html(sample, rank, base_dir, shapenet_dir, output_path):
    """Build one HTML file for a single sample."""
    print(f"  [#{rank}] {sample['label']} "
          f"({sample['scan_id'][:8]}...) ratio={sample['ratio']:.2f}x")

    # Load scanned point cloud
    scan_data = load_scanned_points(
        base_dir, sample['scan_id'], sample['object_id'])

    # Load CAD mesh
    cad_mesh = load_cad_mesh(
        shapenet_dir, sample['cad_source'],
        np.array(sample['obb_dimensions']))

    # --- Scanned point cloud ---
    fig_scan = go.Figure()
    if scan_data:
        pts = scan_data['points']
        pts_centered = pts - pts.mean(axis=0)
        fig_scan.add_trace(make_pointcloud_trace(
            pts_centered, scan_data['colors'],
            name='Scanned (visible only)', color='steelblue'))
    fig_scan.update_layout(
        title=f"Scanned: {sample['scanned_area']:.2f} m²",
        scene=dict(aspectmode='data',
                   xaxis=dict(showbackground=False),
                   yaxis=dict(showbackground=False),
                   zaxis=dict(showbackground=False)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450, width=500,
    )

    # --- CAD mesh ---
    fig_cad = go.Figure()
    if cad_mesh:
        fig_cad.add_trace(make_mesh_trace(
            cad_mesh, name='CAD (complete)', color='#FF8C00', opacity=0.7))
        edges = cad_mesh.vertices[cad_mesh.edges_unique]
        if len(edges) > 5000:
            idx = np.random.choice(len(edges), 5000, replace=False)
            edges = edges[idx]
        xe, ye, ze = [], [], []
        for e in edges:
            xe.extend([e[0, 0], e[1, 0], None])
            ye.extend([e[0, 1], e[1, 1], None])
            ze.extend([e[0, 2], e[1, 2], None])
        fig_cad.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze, mode='lines',
            line=dict(color='rgba(0,0,0,0.15)', width=1),
            name='Edges', showlegend=False))
    fig_cad.update_layout(
        title=f"CAD: {sample['cad_area']:.2f} m²",
        scene=dict(aspectmode='data',
                   xaxis=dict(showbackground=False),
                   yaxis=dict(showbackground=False),
                   zaxis=dict(showbackground=False)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450, width=500,
    )

    # --- Overlay ---
    fig_overlay = go.Figure()
    if scan_data:
        pts_centered = scan_data['points'] - scan_data['points'].mean(axis=0)
        fig_overlay.add_trace(make_pointcloud_trace(
            pts_centered, scan_data['colors'],
            name='Scanned', color='steelblue'))
    if cad_mesh:
        fig_overlay.add_trace(make_mesh_trace(
            cad_mesh, name='CAD', color='#FF8C00', opacity=0.3))
    fig_overlay.update_layout(
        title="Overlay",
        scene=dict(aspectmode='data',
                   xaxis=dict(showbackground=False),
                   yaxis=dict(showbackground=False),
                   zaxis=dict(showbackground=False)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450, width=500,
    )

    scan_html = fig_scan.to_html(full_html=False, include_plotlyjs=True)
    cad_html = fig_cad.to_html(full_html=False, include_plotlyjs=False)
    overlay_html = fig_overlay.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>#{rank}: {sample['label']} - Scan vs CAD</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px;
         background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .sample {{ background: white; border-radius: 12px; padding: 20px;
            margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .sample-header {{ display: flex; align-items: center; gap: 20px;
                   margin-bottom: 15px; }}
  .sample-header img {{ max-width: 200px; max-height: 200px;
                       border-radius: 8px; border: 1px solid #ddd; }}
  .stats {{ flex: 1; }}
  .stats h2 {{ margin: 0 0 8px 0; color: #1a73e8; }}
  .stats table {{ border-collapse: collapse; }}
  .stats td {{ padding: 3px 12px 3px 0; }}
  .stats .val {{ font-weight: bold; font-family: monospace; font-size: 1.1em; }}
  .ratio {{ color: #d93025; font-size: 1.3em; font-weight: bold; }}
  .views {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .view-container {{ flex: 1; min-width: 450px; }}
</style>
</head><body>
<h1>#{rank}: {sample['label']} — Scanned vs CAD</h1>
<div class="sample">
  <div class="sample-header">
    <img src="{sample['img_data']}" alt="{sample['label']}">
    <div class="stats">
      <table>
        <tr><td>Scan ID:</td><td class="val">{sample['scan_id'][:16]}...</td></tr>
        <tr><td>Object ID:</td><td class="val">{sample['object_id']}</td></tr>
        <tr><td>Scanned Area:</td><td class="val">{sample['scanned_area']:.4f} m²</td></tr>
        <tr><td>CAD Area:</td><td class="val">{sample['cad_area']:.4f} m²</td></tr>
        <tr><td>Ratio:</td><td class="val ratio">{sample['ratio']:.2f}x</td></tr>
        <tr><td>Points:</td><td class="val">{sample['num_points']}</td></tr>
        <tr><td>CAD Source:</td><td class="val">{sample['cad_source']}</td></tr>
        <tr><td>OBB (m):</td><td class="val">{sample['obb_dimensions']}</td></tr>
      </table>
    </div>
  </div>
  <div class="views">
    <div class="view-container">{scan_html}</div>
    <div class="view-container">{cad_html}</div>
    <div class="view-container">{overlay_html}</div>
  </div>
</div>
</body></html>"""

    os.makedirs(osp.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)


def parse_range(range_str):
    """Parse range string like '20-28' or '5' into (start, end) inclusive."""
    if '-' in range_str:
        parts = range_str.split('-', 1)
        return int(parts[0]), int(parts[1])
    else:
        idx = int(range_str)
        return idx, idx


def main():
    parser = argparse.ArgumentParser(
        description='Visualize scanned objects vs retrieved CAD models')
    parser.add_argument('--base_dir', type=str,
                        default='scan2therm/object_images',
                        help='Dir with objects.json + PLY (scanned geometry)')
    parser.add_argument('--gates_dir', type=str,
                        default='scan2therm/pipeline_v3_gates/object_images',
                        help='Dir with objects_cad.json (CAD-augmented)')
    parser.add_argument('--shapenet_dir', type=str,
                        default='ShapeNetCore',
                        help='ShapeNet models root')
    parser.add_argument('--out_dir', type=str,
                        default='scan2therm/scan_vs_cad_html_v2',
                        help='Output directory for per-sample HTML files')
    parser.add_argument('--range', type=str, default=None,
                        help='Range of samples to generate, e.g. "20-28" or "5" (0-indexed)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Only consider top K samples (default: all)')
    parser.add_argument('--scan_id', type=str, default=None,
                        help='Visualize objects from a specific scan ID')
    parser.add_argument('--object_id', type=int, nargs='+', default=None,
                        help='Visualize specific object IDs (requires --scan_id)')
    args = parser.parse_args()

    # --- Mode 1: specific scan/objects ---
    if args.scan_id:
        samples = find_specific_objects(
            args.base_dir, args.gates_dir, args.scan_id, args.object_id)
        if not samples:
            print("No matching objects found!")
            return
        print(f"Found {len(samples)} objects for scan {args.scan_id}\n")
        os.makedirs(args.out_dir, exist_ok=True)
        for i, sample in enumerate(samples):
            filename = (f"{sample['scan_id'][:8]}_obj{sample['object_id']:03d}_"
                        f"{sample['label'].lower().replace(' ', '_')}.html")
            output_path = osp.join(args.out_dir, filename)
            build_single_html(sample, i, args.base_dir, args.shapenet_dir, output_path)
            print(f"    -> {output_path}")
        print(f"\nDone! {len(samples)} files in {args.out_dir}/")
        return

    # --- Mode 2: ranked comparison samples ---
    print("Finding comparison samples...")
    samples = find_comparison_samples(args.base_dir, args.gates_dir)
    print(f"Found {len(samples)} total samples\n")

    if not samples:
        print("No samples found!")
        return

    if args.top_k:
        samples = samples[:args.top_k]

    # Determine which samples to render
    if args.range:
        start, end = parse_range(args.range)
        end = min(end, len(samples) - 1)
        start = max(start, 0)
        selected = list(range(start, end + 1))
    else:
        selected = list(range(len(samples)))

    print(f"Generating {len(selected)} HTML files (indices {selected[0]}-{selected[-1]})...\n")

    os.makedirs(args.out_dir, exist_ok=True)

    for idx in selected:
        sample = samples[idx]
        filename = f"{idx:03d}_{sample['label'].lower().replace(' ', '_')}_{sample['scan_id'][:8]}.html"
        output_path = osp.join(args.out_dir, filename)
        build_single_html(sample, idx, args.base_dir, args.shapenet_dir, output_path)
        print(f"    -> {output_path}")

    print(f"\nDone! {len(selected)} files in {args.out_dir}/")


if __name__ == '__main__':
    main()
