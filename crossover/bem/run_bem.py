#!/usr/bin/env python3
"""
CLI entry point for the CrossOver-BEM pipeline.

Usage:
    python bem/run_bem.py \
        --scan_data_dir /path/to/3RScan \
        --scan_id 0a4b8ef6-... \
        --output_dir bem_output/

    # With CrossOver inference:
    python bem/run_bem.py \
        --scan_data_dir /path/to/3RScan \
        --process_dir /path/to/preprocessed \
        --ckpt_scene checkpoints/scene_crossover.pth \
        --ckpt_instance checkpoints/instance_crossover.pth \
        --scan_id 0a4b8ef6-... \
        --run_inference \
        --output_dir bem_output/
"""

import os
import os.path as osp
import sys
import argparse
import json
import logging

# Ensure project root is on path
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

import numpy as np
import torch
from omegaconf import OmegaConf

from bem.pipeline import CrossOverBEMPipeline

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CrossOver-BEM: Extract thermal properties and geometry from 3RScan scans'
    )

    # Required
    parser.add_argument('--scan_data_dir', type=str, required=True,
                        help='Root directory of 3RScan dataset')
    parser.add_argument('--scan_id', type=str, required=True,
                        help='Scan ID to process (e.g., 0a4b8ef6-...)')

    # Output
    parser.add_argument('--output_dir', type=str, default='bem_output',
                        help='Directory to write output JSON files')

    # Optional: preprocessed data / inference
    parser.add_argument('--process_dir', type=str, default='',
                        help='Preprocessed features directory (for CrossOver inference)')
    parser.add_argument('--ckpt_scene', type=str, default='',
                        help='Scene-level model checkpoint path')
    parser.add_argument('--ckpt_instance', type=str, default='',
                        help='Instance-level model checkpoint path')
    parser.add_argument('--run_inference', action='store_true',
                        help='Run CrossOver model inference (requires process_dir and ckpt)')

    # Config overrides
    parser.add_argument('--config', type=str, default='',
                        help='Path to YAML config file (overrides CLI args)')
    parser.add_argument('--csv_path', type=str,
                        default='/home/yunda/3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv',
                        help='Path to 3RScan semantic class mapping CSV')
    parser.add_argument('--label_file_name', type=str,
                        default='labels.instances.align.annotated.v2.ply',
                        help='PLY annotation filename')
    parser.add_argument('--min_points', type=int, default=50,
                        help='Minimum points per object to include')
    parser.add_argument('--skip_structural', action='store_true', default=True,
                        help='Skip wall/floor/ceiling objects')
    parser.add_argument('--no_skip_structural', dest='skip_structural',
                        action='store_false',
                        help='Include wall/floor/ceiling objects')
    parser.add_argument('--room_type', type=str, default='default',
                        choices=['bedroom', 'kitchen', 'office', 'bathroom',
                                 'living_room', 'default'],
                        help='Room type for validation ranges')
    parser.add_argument('--no_validate', action='store_true',
                        help='Skip thermal validation')

    # Model dimensions (for inference mode)
    parser.add_argument('--modalities', nargs='+',
                        default=['rgb', 'point', 'cad', 'referral'],
                        help='Modalities for instance inference')
    parser.add_argument('--input_dim_3d', type=int, default=384)
    parser.add_argument('--input_dim_2d', type=int, default=1536)
    parser.add_argument('--input_dim_1d', type=int, default=768)
    parser.add_argument('--out_dim', type=int, default=768)

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build config
    if args.config and osp.isfile(args.config):
        cfg = OmegaConf.load(args.config)
        # CLI args override config file
        cli_cfg = OmegaConf.create(vars(args))
        cfg = OmegaConf.merge(cfg, cli_cfg)
    else:
        cfg = OmegaConf.create(vars(args))

    # Validate inputs
    scans_dir = osp.join(cfg.scan_data_dir, 'scans')
    scan_dir = osp.join(scans_dir, cfg.scan_id)
    if not osp.isdir(scan_dir):
        log.error(f"Scan directory not found: {scan_dir}")
        sys.exit(1)

    ply_path = osp.join(scan_dir, cfg.label_file_name)
    if not osp.isfile(ply_path):
        log.error(f"PLY file not found: {ply_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Run pipeline
    log.info(f"Processing scan: {cfg.scan_id}")
    pipeline = CrossOverBEMPipeline(cfg)
    output = pipeline.run(
        scan_id=cfg.scan_id,
        room_type=cfg.room_type,
        validate=not cfg.get('no_validate', False),
    )

    # Write output JSON
    output_path = osp.join(cfg.output_dir, f'bem_{cfg.scan_id}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f"Output written to {output_path}")

    # Print summary
    summary = output.get('summary', {})
    print(f"\n{'=' * 60}")
    print(f"BEM Pipeline Results: {cfg.scan_id}")
    print(f"{'=' * 60}")
    print(f"  Objects processed:    {summary.get('num_objects', 0)}")
    print(f"  Total thermal mass:   {summary.get('total_thermal_mass_JK', 0):.1f} J/K")
    print(f"  Total surface area:   {summary.get('total_surface_area_m2', 0):.2f} m^2")

    if 'validation' in output:
        v = output['validation']
        status = 'PASS' if v['room_valid'] else 'WARN'
        print(f"  Validation:           {status} ({v['num_warnings']} warnings)")

    print(f"  Output:               {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
