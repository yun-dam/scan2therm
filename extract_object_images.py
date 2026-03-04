#!/usr/bin/env python3
"""Extract cropped 2D images of each object from RGB sequence frames.

For each scan:
  1. Loads the instance-annotated PLY (objectId per vertex) and the mesh OBJ
  2. Builds an Open3D raycasting scene from the mesh
  3. For every Nth frame, projects the mesh onto the 2D image plane to get a
     per-pixel object-ID map
  4. For each object, picks the frame where it occupies the most pixels
  5. Crops the object from the RGB image using a padded bounding box
  6. Saves to: <out_dir>/<scan_id>/<object_id>.jpg

Usage:
    python scan2therm/extract_object_images.py \
        --scene_list scan2therm/office_scenes_105.txt \
        --data_dir 3DSSG/data/3RScan/data/3RScan \
        --out_dir scan2therm/object_images \
        --frame_skip 10
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import open3d as o3d
from glob import glob
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utility functions (self-contained, adapted from util/scan3r.py & util/render.py)
# ---------------------------------------------------------------------------

def load_intrinsics(scan_dir):
    info_path = osp.join(scan_dir, 'sequence', '_info.txt')
    with open(info_path) as f:
        lines = f.readlines()
    for line in lines:
        if 'm_colorHeight' in line:
            height = float(line.split('= ')[1])
        elif 'm_colorWidth' in line:
            width = float(line.split('= ')[1])
        elif 'm_calibrationColorIntrinsic' in line:
            vals = line.split('= ')[1].split()
            intrinsic_mat = np.array([
                [float(vals[0]), 0, float(vals[2])],
                [0, float(vals[5]), float(vals[6])],
                [0, 0, 1]
            ], dtype=np.float32)
    return intrinsic_mat, int(width), int(height)


def load_frame_idxs(scan_dir, skip=1):
    paths = glob(osp.join(scan_dir, 'sequence', '*.jpg'))
    idxs = sorted(osp.basename(p).split('.')[0].split('-')[-1] for p in paths)
    return idxs[::skip]


def load_pose(scan_dir, frame_id):
    path = osp.join(scan_dir, 'sequence', f'frame-{frame_id}.pose.txt')
    return np.genfromtxt(path)


def load_instance_ids(ply_path):
    with open(ply_path, 'rb') as f:
        ply = PlyData.read(f)
    return ply['vertex']['objectId'].astype(np.int16)


def project_to_2d(scene, intrinsics, extrinsics, w, h, triangles, n_tri, obj_ids):
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsics.astype(np.float64),
        extrinsic_matrix=extrinsics.astype(np.float64),
        width_px=w, height_px=h,
    )
    ans = scene.cast_rays(rays)
    hit_ids = ans['primitive_ids'].numpy()
    valid = hit_ids < n_tri
    hit_verts = triangles[hit_ids[valid]][:, 0]

    obj_map = np.zeros((h, w), dtype=np.int16)
    obj_map[valid] = obj_ids[hit_verts]
    return obj_map


def mask_to_box(mask, pad_ratio=0.1):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    # pad
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    return x1 - pw, y1 - ph, x2 + pw, y2 + ph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_scan(scan_id, data_dir, out_dir, frame_skip, min_pixels):
    scan_dir = osp.join(data_dir, scan_id)
    ply_path = osp.join(scan_dir, 'labels.instances.annotated.v2.ply')
    mesh_path = osp.join(scan_dir, 'mesh.refined.v2.obj')

    if not osp.isfile(ply_path):
        print(f'  SKIP {scan_id}: no PLY file')
        return
    if not osp.isfile(mesh_path):
        print(f'  SKIP {scan_id}: no mesh OBJ')
        return

    scan_out = osp.join(out_dir, scan_id)
    os.makedirs(scan_out, exist_ok=True)

    # Load mesh + instance IDs
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    triangles = np.asarray(mesh.triangles)
    n_tri = triangles.shape[0]
    obj_ids = load_instance_ids(ply_path)

    # Truncate to min length (PLY vs mesh vertex count may differ)
    min_v = min(len(obj_ids), np.asarray(mesh.vertices).shape[0])
    obj_ids = obj_ids[:min_v]

    # Build raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    # Camera intrinsics
    intrinsics, img_w, img_h = load_intrinsics(scan_dir)

    # Load frames
    frame_idxs = load_frame_idxs(scan_dir, skip=frame_skip)

    # Project each frame and accumulate per-object pixel counts
    # obj_best[obj_id] = (best_pixel_count, best_frame_idx, best_obj_map)
    obj_best = {}

    for frame_idx in frame_idxs:
        pose = load_pose(scan_dir, frame_idx)
        if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
            continue
        extrinsics = np.linalg.inv(pose)

        obj_map = project_to_2d(scene, intrinsics, extrinsics, img_w, img_h,
                                triangles, n_tri, obj_ids)

        ids, counts = np.unique(obj_map, return_counts=True)
        for oid, cnt in zip(ids, counts):
            if oid == 0:  # background
                continue
            if oid not in obj_best or cnt > obj_best[oid][0]:
                obj_best[oid] = (cnt, frame_idx, obj_map)

    # Crop and save each object from its best frame
    saved = 0
    for oid, (px_count, frame_idx, obj_map) in obj_best.items():
        if px_count < min_pixels:
            continue

        # The obj_map is (H, W) in landscape orientation matching the raw image
        mask = obj_map == oid
        # Transpose + flip to match the rotated image (ROTATE_270)
        mask_rotated = np.flip(mask.T, axis=1)

        img_path = osp.join(scan_dir, 'sequence', f'frame-{frame_idx}.color.jpg')
        img = Image.open(img_path).transpose(Image.ROTATE_270)
        img_np = np.array(img)

        box = mask_to_box(mask_rotated, pad_ratio=0.15)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_np.shape[1], x2)
        y2 = min(img_np.shape[0], y2)

        crop = img.crop((x1, y1, x2, y2))
        crop.save(osp.join(scan_out, f'{oid}.jpg'))
        saved += 1

    print(f'  {scan_id}: saved {saved} object images')


def main():
    parser = argparse.ArgumentParser(description='Extract cropped 2D object images from 3RScan sequences.')
    parser.add_argument('--scene_list', required=True, help='Text file with scan IDs (one per line)')
    parser.add_argument('--data_dir', required=True, help='3RScan data root (contains scan_id/ dirs)')
    parser.add_argument('--out_dir', default='scan2therm/object_images', help='Output directory')
    parser.add_argument('--frame_skip', type=int, default=10, help='Process every Nth frame (default: 10)')
    parser.add_argument('--min_pixels', type=int, default=500, help='Min pixels for an object to be saved (default: 500)')
    args = parser.parse_args()

    scan_ids = [l.strip() for l in open(args.scene_list) if l.strip()]
    print(f'Processing {len(scan_ids)} scans ...')

    for scan_id in tqdm(scan_ids):
        process_scan(scan_id, args.data_dir, args.out_dir, args.frame_skip, args.min_pixels)

    print('Done.')


if __name__ == '__main__':
    main()
