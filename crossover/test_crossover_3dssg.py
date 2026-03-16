#!/usr/bin/env python3
"""
Test CrossOver on 3RScan scenes using 3DSSG scene graph ground truth.

For each scene:
  1. Load 3DSSG ground truth: object labels + relationships from semseg.v2.json & relationships.json
  2. Encode full scene 3D point cloud with CrossOver
  3. Encode text descriptions derived from the scene graph
  4. Measure retrieval: does the 3D embedding match its own scene description?

Also tests per-object: encode each object's point cloud, match to 3DSSG class labels.

Usage:
    python scan2therm/test_crossover_3dssg.py \
        --data_dir /home/yunda/CrossOver/3DSSG/data/3RScan/data/3RScan \
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
import numpy as np
import torch
import MinkowskiEngine as ME
import trimesh
from datetime import timedelta
from collections import defaultdict

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from model.scene_crossover import SceneCrossOverModel
from util import torch_util


def load_model(ckpt_path):
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


def load_semseg(scan_dir):
    """Load semseg.v2.json -> {objectId: label}."""
    path = osp.join(scan_dir, 'semseg.v2.json')
    with open(path) as f:
        data = json.load(f)
    return {seg['objectId']: seg['label'] for seg in data['segGroups']}


def load_relationships(rel_json_path, scan_id):
    """Load relationships for a scan from relationships.json."""
    with open(rel_json_path) as f:
        data = json.load(f)
    for scan in data['scans']:
        if scan['scan'] == scan_id:
            return scan['relationships']
    return []


def build_scene_description(obj_labels, relationships, rel_names):
    """Build a natural text description from scene graph."""
    # Count objects
    label_counts = defaultdict(int)
    for label in obj_labels.values():
        label_counts[label] += 1

    # Build object summary
    obj_parts = []
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        if label in ('wall', 'floor', 'ceiling'):
            continue
        if count > 1:
            obj_parts.append(f"{count} {label}s")
        else:
            obj_parts.append(f"a {label}")

    desc = "a room with " + ", ".join(obj_parts[:8])

    # Add a few relationships
    rel_parts = []
    for rel in relationships[:5]:
        subj_id, obj_id, rel_idx = rel[0], rel[1], rel[2]
        rel_name = rel[3] if len(rel) > 3 else rel_names[rel_idx] if rel_idx < len(rel_names) else "near"
        subj_label = obj_labels.get(subj_id, "object")
        obj_label = obj_labels.get(obj_id, "object")
        rel_parts.append(f"the {subj_label} is {rel_name} the {obj_label}")

    if rel_parts:
        desc += ". " + ". ".join(rel_parts[:3])

    return desc


def encode_scene_3d(model, scan_dir, device, voxel_size=0.02):
    """Load mesh, sparse quantize, encode with CrossOver 3D encoder."""
    mesh = trimesh.load(osp.join(scan_dir, 'mesh.refined.v2.obj'))
    points = np.asarray(mesh.vertices)
    colors = mesh.visual.to_color().vertex_colors[:, :3]

    feats = colors.astype(np.float32) / 255.0
    feats = (feats - 0.5) / 0.5

    _, sel = ME.utils.sparse_quantize(points / voxel_size, return_index=True)
    coords = np.floor(points[sel] / voxel_size)
    coords -= coords.min(0)
    feats_sel = feats[sel]
    coords, feats_sel = ME.utils.sparse_collate([coords], [feats_sel])

    with torch.no_grad():
        emb = model.encode_3d(coords, feats_sel)
    return emb


def encode_text(model, text):
    """Encode a single text with CrossOver text encoder."""
    with torch.no_grad():
        emb = model.encode_1d([text])
    return emb


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b) / (a.norm() * b.norm() + 1e-8)


def main():
    parser = argparse.ArgumentParser(description='Test CrossOver with 3DSSG scene graphs')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='checkpoints/scene_crossover_scannet+scan3r.pth')
    parser.add_argument('--relationships', type=str,
                        default='/home/yunda/CrossOver/3DSSG/files/relationships.json')
    parser.add_argument('--output', type=str, default='crossover_3dssg_results.json')
    args = parser.parse_args()

    # Load relationship names
    rel_names_path = '/home/yunda/CrossOver/3DSSG/files/relationships.txt'
    with open(rel_names_path) as f:
        rel_names = [line.strip() for line in f]

    # Discover scans
    scan_ids = sorted([
        d for d in os.listdir(args.data_dir)
        if osp.isdir(osp.join(args.data_dir, d))
        and osp.isfile(osp.join(args.data_dir, d, 'mesh.refined.v2.obj'))
        and osp.isfile(osp.join(args.data_dir, d, 'semseg.v2.json'))
    ])
    print(f"Found {len(scan_ids)} scans with semseg data\n")

    # Load model
    print("Loading CrossOver model...")
    model, device = load_model(args.ckpt)

    # ---- Test 1: Scene retrieval ----
    # Encode all scenes as 3D, build text description from scene graph, test if 3D matches its own text
    print("\n=== TEST 1: Scene-Level Retrieval (3D -> Text) ===\n")

    scene_3d_embs = []
    scene_text_embs = []
    scene_descs = []
    valid_scan_ids = []

    for i, scan_id in enumerate(scan_ids):
        scan_dir = osp.join(args.data_dir, scan_id)
        print(f"[{i+1}/{len(scan_ids)}] Encoding {scan_id}...", end=" ", flush=True)

        try:
            obj_labels = load_semseg(scan_dir)
            rels = load_relationships(args.relationships, scan_id)
            desc = build_scene_description(obj_labels, rels, rel_names)

            emb_3d = encode_scene_3d(model, scan_dir, device)
            emb_text = encode_text(model, desc)

            scene_3d_embs.append(emb_3d)
            scene_text_embs.append(emb_text)
            scene_descs.append(desc)
            valid_scan_ids.append(scan_id)
            print(f"OK ({len(obj_labels)} objects, {len(rels)} rels)")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Retrieval: for each 3D embedding, rank all text embeddings
    n = len(valid_scan_ids)
    sim_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_sim(scene_3d_embs[i], scene_text_embs[j])

    # Compute retrieval metrics
    ranks = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    print(f"\n{'Scan ID':<40} {'Rank':>5} {'Sim(self)':>10} {'Sim(best)':>10} {'Best Match'}")
    print("-" * 100)

    for i in range(n):
        row = sim_matrix[i]
        sorted_indices = row.argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

        best_idx = sorted_indices[0].item()
        self_sim = row[i].item()
        best_sim = row[best_idx].item()

        if rank == 1:
            top1_correct += 1
        if rank <= 3:
            top3_correct += 1
        if rank <= 5:
            top5_correct += 1

        match_str = "SELF" if best_idx == i else valid_scan_ids[best_idx][:20]
        print(f"{valid_scan_ids[i]:<40} {rank:>5} {self_sim:>10.4f} {best_sim:>10.4f} {match_str}")

    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)

    print(f"\n{'=' * 60}")
    print(f"SCENE RETRIEVAL RESULTS ({n} scenes)")
    print(f"{'=' * 60}")
    print(f"  Recall@1:    {top1_correct}/{n} = {top1_correct/n*100:.1f}%")
    print(f"  Recall@3:    {top3_correct}/{n} = {top3_correct/n*100:.1f}%")
    print(f"  Recall@5:    {top5_correct}/{n} = {top5_correct/n*100:.1f}%")
    print(f"  Mean Rank:   {mean_rank:.1f}")
    print(f"  Median Rank: {median_rank:.1f}")

    # ---- Test 2: Text query retrieval ----
    print(f"\n\n=== TEST 2: Text Query -> Scene Retrieval ===\n")

    queries = [
        "a kitchen with cabinets and a sink",
        "a bedroom with a bed and nightstands",
        "a living room with a sofa and table",
        "a bathroom with a toilet and bathtub",
        "an office with a desk and chair",
    ]

    for query in queries:
        q_emb = encode_text(model, query)
        sims = []
        for i in range(n):
            sims.append((valid_scan_ids[i], cosine_sim(q_emb, scene_3d_embs[i]).item()))
        sims.sort(key=lambda x: -x[1])

        print(f"Query: \"{query}\"")
        for scan_id, sim in sims[:3]:
            scan_dir = osp.join(args.data_dir, scan_id)
            labels = load_semseg(scan_dir)
            obj_summary = ", ".join(sorted(set(labels.values()) - {'wall', 'floor', 'ceiling'}))
            print(f"  {sim:.4f}  {scan_id}  [{obj_summary}]")
        print()

    # Save results
    output = {
        'num_scenes': n,
        'scene_retrieval': {
            'recall_at_1': round(top1_correct / n * 100, 2),
            'recall_at_3': round(top3_correct / n * 100, 2),
            'recall_at_5': round(top5_correct / n * 100, 2),
            'mean_rank': round(mean_rank, 2),
            'median_rank': round(median_rank, 2),
        },
        'per_scene': [
            {
                'scan_id': valid_scan_ids[i],
                'description': scene_descs[i],
                'rank': int(ranks[i]),
                'self_similarity': round(sim_matrix[i, i].item(), 4),
            }
            for i in range(n)
        ],
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
