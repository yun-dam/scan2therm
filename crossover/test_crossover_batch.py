#!/usr/bin/env python3
"""
Run CrossOver 3D-text matching on all downloaded scans and summarize accuracy.

Usage:
    python scan2therm/test_crossover_batch.py \
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
from collections import defaultdict

from scan2therm.test_crossover import (
    load_model, load_scan, encode_object_3d, encode_text_labels,
    cosine_sim, load_global_id_to_label, NYU40_CLASSES, STRUCTURAL_NYU40
)


def run_single_scan(model, device, text_embeddings, data_dir, scan_id,
                    gid_to_label, min_points=50, skip_structural=True):
    """Run CrossOver on one scan, return per-object results."""
    scan = load_scan(data_dir, scan_id)
    unique_ids = np.unique(scan['object_ids'])
    results = []

    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        mask = scan['object_ids'] == obj_id
        count = mask.sum()
        if count < min_points:
            continue

        nyu40_gt = int(scan['nyu40_ids'][np.where(mask)[0][0]])
        global_id = int(scan['global_ids'][np.where(mask)[0][0]])

        if skip_structural and nyu40_gt in STRUCTURAL_NYU40:
            continue
        if nyu40_gt == 31:  # person
            continue

        gt_name = NYU40_CLASSES.get(nyu40_gt, 'unknown')
        gt_label = gid_to_label.get(global_id, gt_name)

        obj_emb = encode_object_3d(model, scan['points'][mask], scan['colors'][mask], device)
        if obj_emb is None:
            continue

        sims = {}
        for nyu_id, text_emb in text_embeddings.items():
            sims[nyu_id] = cosine_sim(obj_emb, text_emb).item()

        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        pred_id = ranked[0][0]
        top3_ids = [r[0] for r in ranked[:3]]
        top5_ids = [r[0] for r in ranked[:5]]

        results.append({
            'object_id': int(obj_id),
            'gt_nyu40': nyu40_gt,
            'gt_name': gt_name,
            'gt_label': gt_label,
            'pred_nyu40': int(pred_id),
            'pred_name': NYU40_CLASSES[pred_id],
            'correct_top1': pred_id == nyu40_gt,
            'correct_top3': nyu40_gt in top3_ids,
            'correct_top5': nyu40_gt in top5_ids,
            'num_points': int(count),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Batch test CrossOver vs ground truth')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='checkpoints/scene_crossover_scannet+scan3r.pth')
    parser.add_argument('--csv_path', type=str,
                        default='/home/yunda/CrossOver/3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv')
    parser.add_argument('--min_points', type=int, default=50)
    parser.add_argument('--output', type=str, default='crossover_batch_results.json')
    args = parser.parse_args()

    # Discover scans
    scan_ids = sorted([
        d for d in os.listdir(args.data_dir)
        if osp.isdir(osp.join(args.data_dir, d))
        and osp.isfile(osp.join(args.data_dir, d, 'mesh.refined.v2.obj'))
        and osp.isfile(osp.join(args.data_dir, d, 'labels.instances.annotated.v2.ply'))
    ])
    print(f"Found {len(scan_ids)} scans\n")

    # Load model once
    print("Loading CrossOver model...")
    model, device = load_model(args.ckpt)

    # Encode text labels once
    print("Encoding NYU40 class names...\n")
    text_embeddings = encode_text_labels(model, NYU40_CLASSES, device)

    # CSV labels
    gid_to_label = {}
    if osp.isfile(args.csv_path):
        gid_to_label = load_global_id_to_label(args.csv_path)

    # Run all scans
    all_results = []
    per_scan = []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for i, scan_id in enumerate(scan_ids):
        print(f"[{i+1}/{len(scan_ids)}] {scan_id}...", end=" ", flush=True)
        try:
            results = run_single_scan(
                model, device, text_embeddings,
                args.data_dir, scan_id, gid_to_label, args.min_points
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        correct1 = sum(r['correct_top1'] for r in results)
        correct3 = sum(r['correct_top3'] for r in results)
        correct5 = sum(r['correct_top5'] for r in results)
        total = len(results)
        acc1 = correct1 / total * 100 if total > 0 else 0

        print(f"{total} objects, top1={acc1:.0f}%")

        per_scan.append({
            'scan_id': scan_id,
            'num_objects': total,
            'top1_correct': correct1,
            'top3_correct': correct3,
            'top5_correct': correct5,
            'top1_accuracy': round(acc1, 2),
        })

        for r in results:
            gt = r['gt_name']
            per_class_total[gt] += 1
            if r['correct_top1']:
                per_class_correct[gt] += 1

        all_results.extend(results)

    # Overall summary
    total_obj = len(all_results)
    total_top1 = sum(r['correct_top1'] for r in all_results)
    total_top3 = sum(r['correct_top3'] for r in all_results)
    total_top5 = sum(r['correct_top5'] for r in all_results)

    print(f"\n{'=' * 60}")
    print(f"OVERALL RESULTS ({len(scan_ids)} scans, {total_obj} objects)")
    print(f"{'=' * 60}")
    print(f"  Top-1 Accuracy: {total_top1}/{total_obj} = {total_top1/total_obj*100:.1f}%")
    print(f"  Top-3 Accuracy: {total_top3}/{total_obj} = {total_top3/total_obj*100:.1f}%")
    print(f"  Top-5 Accuracy: {total_top5}/{total_obj} = {total_top5/total_obj*100:.1f}%")

    # Per-class breakdown
    print(f"\n{'Class':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for cls in sorted(per_class_total.keys()):
        c = per_class_correct.get(cls, 0)
        t = per_class_total[cls]
        print(f"{cls:<20} {c:>8} {t:>8} {c/t*100:>9.1f}%")
    print(f"{'=' * 60}")

    # Save
    output = {
        'num_scans': len(scan_ids),
        'num_objects': total_obj,
        'top1_accuracy': round(total_top1 / total_obj * 100, 2) if total_obj else 0,
        'top3_accuracy': round(total_top3 / total_obj * 100, 2) if total_obj else 0,
        'top5_accuracy': round(total_top5 / total_obj * 100, 2) if total_obj else 0,
        'per_class': {
            cls: {
                'correct': per_class_correct.get(cls, 0),
                'total': per_class_total[cls],
                'accuracy': round(per_class_correct.get(cls, 0) / per_class_total[cls] * 100, 2)
            }
            for cls in sorted(per_class_total.keys())
        },
        'per_scan': per_scan,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
