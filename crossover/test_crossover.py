#!/usr/bin/env python3
"""
Test CrossOver 3D-text matching against ground-truth labels.

For each object in a scene:
  1. Encode its 3D point cloud with CrossOver's 3D encoder
  2. Encode NYU40 class names with CrossOver's text encoder
  3. Predict class via highest cosine similarity
  4. Compare to ground-truth NYU40 label

Usage:
    python scan2therm/test_crossover.py \
        --data_dir /home/yunda/CrossOver/3DSSG/data/3RScan/data/3RScan \
        --scan_id 1d2f8514-d757-207c-8e40-377957df6f67 \
        --ckpt checkpoints/scene_crossover_scannet+scan3r.pth
"""

import sys
import os
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
os.environ["PYTHONPATH"] = osp.join(osp.dirname(__file__), '..', 'third_party', 'BLIP') \
    + ":" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'third_party', 'BLIP'))
import argparse
import json
import csv
import numpy as np
import torch
import MinkowskiEngine as ME
import trimesh
from datetime import timedelta
from plyfile import PlyData

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from model.scene_crossover import SceneCrossOverModel
from util import torch_util

# NYU40 class names for text encoding
NYU40_CLASSES = {
    1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair',
    6: 'sofa', 7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf',
    11: 'picture', 12: 'counter', 13: 'blinds', 14: 'desk', 15: 'shelves',
    16: 'curtain', 17: 'dresser', 18: 'pillow', 19: 'mirror', 20: 'floor mat',
    21: 'clothes', 22: 'ceiling', 23: 'books', 24: 'refrigerator', 25: 'television',
    26: 'paper', 27: 'towel', 28: 'shower curtain', 29: 'box', 30: 'whiteboard',
    31: 'person', 32: 'nightstand', 33: 'toilet', 34: 'sink', 35: 'lamp',
    36: 'bathtub', 37: 'bag', 38: 'other structure', 39: 'other furniture', 40: 'other prop',
}

STRUCTURAL_NYU40 = {1, 2, 22}


def load_model(ckpt_path, device='cuda'):
    """Load CrossOver scene model."""
    init_kw = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    acc = Accelerator(kwargs_handlers=[init_kw])
    args = argparse.Namespace(
        input_dim_3d=512, input_dim_2d=1536, input_dim_1d=768, out_dim=768
    )
    model = SceneCrossOverModel(args, acc.device)
    model = acc.prepare(model)
    model.eval()
    model.to(acc.device)
    torch_util.load_weights(model, ckpt_path, acc.device)
    return model, acc.device


def load_scan(data_dir, scan_id):
    """Load PLY annotations and mesh."""
    ply_path = osp.join(data_dir, scan_id, 'labels.instances.annotated.v2.ply')
    ply_data = PlyData.read(ply_path)
    v = ply_data['vertex']

    mesh = trimesh.load(osp.join(data_dir, scan_id, 'mesh.refined.v2.obj'))
    mesh_points = np.asarray(mesh.vertices)
    mesh_colors = mesh.visual.to_color().vertex_colors[:, :3]

    n = min(len(v['objectId']), mesh_points.shape[0])
    return {
        'points': mesh_points[:n],
        'colors': mesh_colors[:n],
        'object_ids': np.array(v['objectId'][:n]),
        'nyu40_ids': np.array(v['NYU40'][:n]),
        'global_ids': np.array(v['globalId'][:n]),
    }


def encode_object_3d(model, points, colors, device, voxel_size=0.02):
    """Sparse-quantize an object point cloud and encode with CrossOver 3D encoder."""
    # Normalize colors to [-1, 1]
    feats = colors.astype(np.float32) / 255.0
    feats = (feats - 0.5) / 0.5

    _, sel = ME.utils.sparse_quantize(points / voxel_size, return_index=True)
    if len(sel) < 10:
        return None

    coords = np.floor(points[sel] / voxel_size)
    coords -= coords.min(0)
    feats_sel = feats[sel]

    coords, feats_sel = ME.utils.sparse_collate([coords], [feats_sel])

    with torch.no_grad():
        emb = model.encode_3d(coords, feats_sel)
    return emb


def encode_text_labels(model, class_names, device):
    """Encode class names as text with CrossOver text encoder."""
    embeddings = {}
    for nyu_id, name in class_names.items():
        prompt = f"a {name} in an indoor room"
        with torch.no_grad():
            emb = model.encode_1d([prompt])
        embeddings[nyu_id] = emb
    return embeddings


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a = a.flatten()
    b = b.flatten()
    return (a @ b) / (a.norm() * b.norm() + 1e-8)


def load_global_id_to_label(csv_path):
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


def main():
    parser = argparse.ArgumentParser(description='Test CrossOver 3D-text matching vs ground truth')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--scan_id', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='checkpoints/scene_crossover_scannet+scan3r.pth')
    parser.add_argument('--csv_path', type=str,
                        default='/home/yunda/CrossOver/3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv')
    parser.add_argument('--min_points', type=int, default=50)
    parser.add_argument('--skip_structural', action='store_true', default=True)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    # Load model
    print("Loading CrossOver model...")
    model, device = load_model(args.ckpt)

    # Encode all NYU40 class names as text
    print("Encoding NYU40 class names as text...")
    text_embeddings = encode_text_labels(model, NYU40_CLASSES, device)

    # Load scan
    print(f"Loading scan {args.scan_id}...")
    scan = load_scan(args.data_dir, args.scan_id)

    # Optional CSV labels
    gid_to_label = {}
    if osp.isfile(args.csv_path):
        gid_to_label = load_global_id_to_label(args.csv_path)

    # Process each object
    unique_ids = np.unique(scan['object_ids'])
    results = []
    correct = 0
    total = 0

    print(f"\nProcessing {len(unique_ids)} objects...\n")
    print(f"{'ObjID':>6} {'GT Label':<20} {'Predicted':<20} {'Match':>5} {'Top-3 Predictions'}")
    print("-" * 90)

    for obj_id in unique_ids:
        if obj_id == 0:
            continue

        mask = scan['object_ids'] == obj_id
        count = mask.sum()
        if count < args.min_points:
            continue

        nyu40_gt = int(scan['nyu40_ids'][np.where(mask)[0][0]])
        global_id = int(scan['global_ids'][np.where(mask)[0][0]])

        if args.skip_structural and nyu40_gt in STRUCTURAL_NYU40:
            continue

        gt_name = NYU40_CLASSES.get(nyu40_gt, 'unknown')
        gt_label = gid_to_label.get(global_id, gt_name)

        # Encode object 3D
        obj_points = scan['points'][mask]
        obj_colors = scan['colors'][mask]
        obj_emb = encode_object_3d(model, obj_points, obj_colors, device)
        if obj_emb is None:
            continue

        # Compare to all text embeddings
        sims = {}
        for nyu_id, text_emb in text_embeddings.items():
            sims[nyu_id] = cosine_sim(obj_emb, text_emb).item()

        # Rank
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        pred_id = ranked[0][0]
        pred_name = NYU40_CLASSES[pred_id]
        is_correct = (pred_id == nyu40_gt)

        top3 = [(NYU40_CLASSES[r[0]], f"{r[1]:.3f}") for r in ranked[:3]]
        top3_str = ", ".join([f"{n} ({s})" for n, s in top3])

        total += 1
        if is_correct:
            correct += 1

        match_str = "YES" if is_correct else "no"
        print(f"{obj_id:>6} {gt_label:<20} {pred_name:<20} {match_str:>5} {top3_str}")

        results.append({
            'object_id': int(obj_id),
            'ground_truth': {'nyu40_id': nyu40_gt, 'nyu40_class': gt_name, 'label': gt_label},
            'predicted': {'nyu40_id': int(pred_id), 'nyu40_class': pred_name},
            'correct': is_correct,
            'top3': [{'nyu40_id': int(r[0]), 'class': NYU40_CLASSES[r[0]], 'similarity': round(r[1], 4)} for r in ranked[:3]],
            'num_points': int(count),
        })

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'=' * 60}")

    output_data = {
        'scan_id': args.scan_id,
        'accuracy': round(accuracy, 2),
        'correct': correct,
        'total': total,
        'objects': results,
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        # Write to default location
        out_path = f'crossover_vs_gt_{args.scan_id}.json'
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
