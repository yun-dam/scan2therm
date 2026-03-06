#!/usr/bin/env python3
"""
CrossOver embedding-based CAD retrieval for scan2therm.

Replaces the label-based ShapeNet lookup (cad_geometry.py) with learned
CrossOver embeddings: each scanned object is matched to the most
geometrically similar ShapeNet CAD model via cosine similarity in the
shared 768-dim embedding space.

Usage (standalone):
    python scan2therm/crossover_cad_geometry.py \
        --scene_list scan2therm/office_scenes_105.txt \
        --data_dir scan2therm/object_images \
        --shapenet_dir Scan2CAD/Assets/shapenet-sample \
        --out_dir scan2therm/object_images \
        --ckpt checkpoints/instance_crossover_scannet+scan3r+multiscan+arkitscenes.pth \
        --i2pmae_ckpt checkpoints/pointbind_i2pmae.pt
"""

import os
import os.path as osp
import argparse
import json
import numpy as np
import torch
import trimesh
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from scan2therm.extract_objects import extract_objects
from scan2therm.cad_geometry import (
    compute_cad_geometry, STRUCTURAL_LABELS, LABEL_TO_SYNSET,
    CAD_AREA_RATIO_THRESHOLD,
)
from scan2therm.point_cloud_utils import load_obj, sample_faces
from util.point_cloud import sample_and_normalize_pcl, get_object_loc_box
from scan2therm.scan3r_utils import load_ply_data


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_crossover_model(ckpt_path, i2pmae_ckpt=None, device='cuda'):
    """Load InstanceCrossOverModel with checkpoint weights.

    Args:
        ckpt_path: Directory containing model.safetensors (HuggingFace download)
        i2pmae_ckpt: Path to pointbind_i2pmae.pt. If provided, monkey-patches
                     the hardcoded path before model init.
        device: torch device string.

    Returns:
        model on device, in eval mode.
    """
    from omegaconf import DictConfig

    # BLIP uses `from models.vit import ...` which requires third_party/BLIP on sys.path
    blip_path = osp.join(osp.dirname(__file__), '..', 'third_party', 'BLIP')
    if osp.isdir(blip_path) and blip_path not in sys.path:
        sys.path.insert(0, osp.abspath(blip_path))

    from model.instance_crossover import InstanceCrossOverModel
    from util import torch_util

    # Patch I2PMAE checkpoint path before model construction
    if i2pmae_ckpt is not None:
        InstanceCrossOverModel.__init__.__globals__  # ensure class is loaded
        _orig_init = InstanceCrossOverModel.__init__

        def _patched_init(self, args, dev='cuda'):
            _orig_init(self, args, dev)

        # Simpler: just set it on the class before construction
        # The model reads self.point_feature_extractor_ckpt in __init__
        # We'll subclass temporarily
        pass  # handled below via direct attribute override

    model_args = DictConfig({
        'out_dim': 768,
        'input_dim_3d': 384,
        'input_dim_2d': 1536,
        'input_dim_1d': 768,
    })

    # Override the hardcoded I2PMAE path before model init
    if i2pmae_ckpt is not None:
        _orig_init = InstanceCrossOverModel.__init__

        def _patched_init(self, args, dev='cuda'):
            # Temporarily replace the hardcoded path
            self.device = dev
            self.out_dim = args.out_dim
            self.point_feature_extractor_ckpt = Path(i2pmae_ckpt)
            # Now call rest of original init logic (skip the line that sets ckpt)
            _orig_init(self, args, dev)

        # The original __init__ sets self.point_feature_extractor_ckpt = Path('/drive/...')
        # then immediately calls self.loadFeatureExtractor which reads it.
        # We need to intercept. Safest: patch the class attribute default.
        InstanceCrossOverModel._i2pmae_ckpt_override = Path(i2pmae_ckpt)

        _real_init = InstanceCrossOverModel.__init__

        def _new_init(self, args, dev='cuda'):
            _real_init(self, args, dev)

        # Actually the simplest approach: just override after construction
        # won't work because loadFeatureExtractor is called IN __init__.
        # So let's directly monkey-patch the source attribute:
        import model.instance_crossover as _mic
        _orig_path = _mic.Path
        _target = Path(i2pmae_ckpt)

        # The line in __init__ is:
        #   self.point_feature_extractor_ckpt = Path('/drive/pretrained-models/pointbind_i2pmae.pt')
        # We can't easily intercept that. Instead, let's just temporarily
        # modify the class after init and reload the extractor... but that's wasteful.
        #
        # Best approach: subclass and override __init__
        pass

    # Use subclass approach to override I2PMAE path
    from model.instance_crossover import InstanceCrossOverModel as _Base

    class _PatchedModel(_Base):
        def __init__(self, args, dev='cuda'):
            # We need to set point_feature_extractor_ckpt BEFORE
            # loadFeatureExtractor is called. The parent __init__ sets it
            # then immediately calls loadFeatureExtractor.
            # So we override __init__ fully, copying the parent logic.
            import torch.nn as nn
            from common.constants import ModalityType
            from modules.layers.pointnet import PointTokenizeEncoder
            from modules.basic_modules import get_mlp_head
            from modules.build import build_module
            from third_party.BLIP.models.blip import blip_feature_extractor

            nn.Module.__init__(self)
            self.device = dev
            self.out_dim = args.out_dim

            self.modalities = ['point', 'cad', 'rgb', 'referral']
            self.feat_dims = {
                ModalityType.POINT: args.input_dim_3d,
                ModalityType.CAD: args.input_dim_3d,
                ModalityType.RGB: args.input_dim_2d,
                ModalityType.REF: args.input_dim_1d,
            }

            self.point_feature_extractor_name = 'I2PMAE'
            # Use the user-provided path instead of hardcoded /drive/...
            if i2pmae_ckpt is not None:
                self.point_feature_extractor_ckpt = Path(i2pmae_ckpt)
            else:
                self.point_feature_extractor_ckpt = Path(
                    '/drive/pretrained-models/pointbind_i2pmae.pt')
            self.point_feature_extractor = self.loadFeatureExtractor("3D")

            self.modality_encoders = nn.ModuleDict({
                ModalityType.POINT: PointTokenizeEncoder(
                    hidden_size=self.feat_dims[ModalityType.POINT]),
                ModalityType.CAD: PointTokenizeEncoder(
                    use_attn=False,
                    hidden_size=self.feat_dims[ModalityType.CAD]),
            })

            self.encoder2D = build_module(
                "2D", 'DinoV2', ckpt='dinov2_vitg14', device=self.device)

            self.encoder1D = blip_feature_extractor(
                pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
                image_size=224, vit='large').to(self.device)

            self.modality_projections = nn.ModuleDict({
                ModalityType.POINT: get_mlp_head(
                    self.feat_dims[ModalityType.POINT],
                    self.feat_dims[ModalityType.POINT], self.out_dim),
                ModalityType.CAD: get_mlp_head(
                    self.feat_dims[ModalityType.CAD],
                    self.feat_dims[ModalityType.CAD], self.out_dim),
                ModalityType.RGB: get_mlp_head(
                    self.feat_dims[ModalityType.RGB],
                    self.feat_dims[ModalityType.RGB] // 2, self.out_dim),
                ModalityType.REF: get_mlp_head(
                    self.feat_dims[ModalityType.REF],
                    self.feat_dims[ModalityType.REF], self.out_dim),
            })

    model = _PatchedModel(model_args, device)
    model.eval()
    model.to(device)

    # Load CrossOver checkpoint weights
    from util import torch_util
    torch_util.load_weights(model, ckpt_path, device)

    return model


# ---------------------------------------------------------------------------
# CAD embedding database
# ---------------------------------------------------------------------------

def build_cad_database(model, shapenet_dir, cache_path, n_samples=5000,
                       device='cuda'):
    """Build or load cached CAD embedding database from ShapeNet models.

    Walks all synset dirs in shapenet_dir, loads each model_normalized.obj,
    samples n_samples surface points, encodes via CrossOver's CAD encoder,
    and caches the result as .npz.

    Args:
        model: InstanceCrossOverModel (eval mode, on device).
        shapenet_dir: Root ShapeNet directory (synset/model_id/models/).
        cache_path: Path for .npz cache file.
        n_samples: Points to sample per CAD mesh.
        device: torch device string.

    Returns:
        dict: {f"{synset_id}/{model_id}": np.array(768,)} embeddings.
    """
    if osp.isfile(cache_path):
        print(f"  Loading CAD embedding cache: {cache_path}")
        data = np.load(cache_path)
        cad_db = {k: data[k] for k in data.files}
        print(f"  Loaded {len(cad_db)} CAD embeddings from cache")
        return cad_db

    print(f"  Building CAD embedding database from {shapenet_dir} ...")
    cad_db = {}

    if not osp.isdir(shapenet_dir):
        print(f"  [WARN] ShapeNet dir not found: {shapenet_dir}")
        return cad_db

    synset_dirs = sorted([
        d for d in os.listdir(shapenet_dir)
        if osp.isdir(osp.join(shapenet_dir, d))
    ])

    for synset_id in tqdm(synset_dirs, desc="  ShapeNet synsets"):
        synset_path = osp.join(shapenet_dir, synset_id)
        model_dirs = sorted([
            d for d in os.listdir(synset_path)
            if osp.isdir(osp.join(synset_path, d))
        ])

        for model_id in model_dirs:
            obj_path = osp.join(
                synset_path, model_id, 'models', 'model_normalized.obj')
            if not osp.isfile(obj_path):
                continue

            try:
                vertices, faces = load_obj(obj_path)
                if len(vertices) == 0 or len(faces) == 0:
                    continue

                points = sample_faces(vertices, faces, n_samples=n_samples)
                key = f"{synset_id}/{model_id}"

                # Encode via CAD encoder (single object)
                cad_points = [points]  # list of np arrays
                cad_mask = torch.ones(1, 1).bool().to(device)

                with torch.no_grad():
                    embeddings = model.encode_cad_objects(cad_points, cad_mask)
                    # (1, 1, 768) → (768,)
                    emb = embeddings[0, 0].cpu().numpy()
                    # L2 normalize
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm

                cad_db[key] = emb

            except Exception as e:
                print(f"  [WARN] Failed to process {synset_id}/{model_id}: {e}")
                continue

    print(f"  Built {len(cad_db)} CAD embeddings, saving to {cache_path}")
    os.makedirs(osp.dirname(cache_path) or '.', exist_ok=True)
    np.savez_compressed(cache_path, **cad_db)
    return cad_db


# ---------------------------------------------------------------------------
# Per-scene: encode scanned objects
# ---------------------------------------------------------------------------

def encode_scene_objects(model, vertices, objects, device='cuda'):
    """Encode scanned objects via CrossOver's point encoder.

    Args:
        model: InstanceCrossOverModel (eval mode).
        vertices: structured array from load_ply_data.
        objects: list of object dicts from extract_objects().
        device: torch device string.

    Returns:
        dict: {object_id: np.array(768,)} L2-normalized embeddings.
    """
    obj_points_list = []
    obj_ids = []

    for obj in objects:
        label = obj['label'].lower().strip()
        if label in STRUCTURAL_LABELS:
            continue

        obj_id = obj['object_id']
        mask = vertices['objectId'] == obj_id
        if mask.sum() < 50:
            continue

        points = np.column_stack([
            vertices['x'][mask],
            vertices['y'][mask],
            vertices['z'][mask],
        ])

        obj_points_list.append(points)
        obj_ids.append(obj_id)

    if not obj_points_list:
        return {}

    # Encode all objects via the point encoder
    point_mask = torch.ones(1, len(obj_points_list)).bool().to(device)

    with torch.no_grad():
        embeddings = model.encode_point_objects(obj_points_list, point_mask)
        # (1, N, 768)
        embs = embeddings[0].cpu().numpy()

    result = {}
    for i, obj_id in enumerate(obj_ids):
        emb = embs[i]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        result[obj_id] = emb

    return result


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_best_cad(obj_embedding, cad_db):
    """Find the best matching CAD model via cosine similarity.

    Args:
        obj_embedding: (768,) L2-normalized embedding.
        cad_db: dict {synset/model_id: (768,) embedding}.

    Returns:
        (synset_id, model_id, score) or (None, None, -1).
    """
    if not cad_db:
        return None, None, -1.0

    keys = list(cad_db.keys())
    db_matrix = np.stack([cad_db[k] for k in keys])  # (M, 768)

    # Cosine similarity (both are L2-normalized)
    scores = db_matrix @ obj_embedding  # (M,)
    best_idx = int(np.argmax(scores))
    best_key = keys[best_idx]
    best_score = float(scores[best_idx])

    parts = best_key.split('/')
    synset_id = parts[0]
    model_id = parts[1]
    return synset_id, model_id, best_score


# ---------------------------------------------------------------------------
# Process a single scan
# ---------------------------------------------------------------------------

def process_scan_crossover(scan_id, model, cad_db, data_dir, shapenet_dir,
                           out_dir, csv_path=None, objects_json_path=None,
                           device='cuda'):
    """Process one scan: encode objects, retrieve CAD, compute geometry.

    Args:
        scan_id: 3RScan scan identifier.
        model: InstanceCrossOverModel (eval mode).
        cad_db: dict from build_cad_database().
        data_dir: directory with PLY files per scan.
        shapenet_dir: ShapeNet models root.
        out_dir: output directory (objects_cad.json per scan).
        csv_path: optional 3RScan semantic classes CSV.
        objects_json_path: optional 3DSSG objects.json.
        device: torch device string.

    Returns:
        result dict or None.
    """
    # Find PLY file (same logic as cad_geometry.py)
    scan_data_dir = data_dir
    ply_in_out = osp.join(out_dir, scan_id,
                          'labels.instances.annotated.v2.ply')
    ply_in_data = osp.join(data_dir, scan_id,
                           'labels.instances.annotated.v2.ply')

    if osp.isfile(ply_in_out):
        scan_data_dir = out_dir
        label_file = 'labels.instances.annotated.v2.ply'
    elif osp.isfile(osp.join(out_dir, scan_id,
                             'labels.instances.align.annotated.v2.ply')):
        scan_data_dir = out_dir
        label_file = 'labels.instances.align.annotated.v2.ply'
    elif osp.isfile(ply_in_data):
        scan_data_dir = data_dir
        label_file = 'labels.instances.annotated.v2.ply'
    else:
        print(f"  [SKIP] No PLY found for {scan_id}")
        return None

    # Extract objects (labels, materials, OBB, etc.)
    try:
        objects = extract_objects(
            data_dir=scan_data_dir,
            scan_id=scan_id,
            label_file=label_file,
            csv_path=csv_path,
            objects_json_path=objects_json_path,
            min_points=50,
            skip_structural=False,
        )
    except Exception as e:
        print(f"  [ERROR] extract_objects failed for {scan_id}: {e}")
        return None

    # Load vertices for point encoding
    vertices = load_ply_data(scan_data_dir, scan_id, label_file)

    # Encode all non-structural objects
    obj_embeddings = encode_scene_objects(model, vertices, objects, device)

    cad_hit = 0
    cad_miss = 0

    for obj in objects:
        label = obj['label'].lower().strip()

        if label in STRUCTURAL_LABELS:
            obj['cad_source'] = 'scanned_geometry'
            continue

        obj_id = obj['object_id']

        if obj_id not in obj_embeddings:
            obj['cad_source'] = 'scanned_geometry'
            cad_miss += 1
            continue

        obj_emb = obj_embeddings[obj_id]
        synset_id, model_id, score = retrieve_best_cad(obj_emb, cad_db)

        if synset_id is None:
            obj['cad_source'] = 'scanned_geometry'
            cad_miss += 1
            continue

        # Load the matched CAD model and compute scaled geometry
        obj_path = osp.join(shapenet_dir, synset_id, model_id,
                            'models', 'model_normalized.obj')
        if not osp.isfile(obj_path):
            obj['cad_source'] = 'scanned_geometry'
            cad_miss += 1
            continue

        try:
            loaded = trimesh.load(obj_path)
            if isinstance(loaded, trimesh.Scene):
                meshes = [g for g in loaded.geometry.values()
                          if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    obj['cad_source'] = 'scanned_geometry'
                    cad_miss += 1
                    continue
                cad_mesh = trimesh.util.concatenate(meshes)
            else:
                cad_mesh = loaded

            cad_extents = cad_mesh.bounding_box.extents
            scan_obb = np.array(obj['obb_dimensions'])

            if np.any(scan_obb < 1e-4):
                obj['cad_source'] = 'scanned_geometry'
                cad_miss += 1
                continue

            area, volume, scale = compute_cad_geometry(
                scan_obb, cad_mesh, cad_extents)

            # Guard: if CAD area >> scanned area, the OBB is likely from bad
            # instance segmentation and CAD scaling amplifies the error.
            scanned_area = obj['surface_area_m2']
            if scanned_area > 0 and area / scanned_area > CAD_AREA_RATIO_THRESHOLD:
                print(f"  [FALLBACK] obj {obj_id} ({label}): "
                      f"CAD area {area:.1f} m² >> scanned {scanned_area:.1f} m² "
                      f"(ratio {area/scanned_area:.1f}x > {CAD_AREA_RATIO_THRESHOLD}x), "
                      f"keeping scanned geometry")
                obj['cad_source'] = 'scanned_geometry (cad_ratio_fallback)'
                cad_miss += 1
                continue

            obj['surface_area_m2'] = round(area, 4)
            obj['volume_m3'] = round(volume, 4)
            obj['cad_source'] = f"crossover:{synset_id}/{model_id}"
            obj['crossover_score'] = round(score, 4)
            cad_hit += 1

        except Exception as e:
            print(f"  [WARN] CAD geometry failed for obj {obj_id} "
                  f"({label}): {e}")
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

    print(f"  {scan_id}: {len(objects)} objects, {cad_hit} CrossOver-matched, "
          f"{cad_miss} scan-only → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Main (standalone)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CrossOver embedding-based CAD retrieval for scan2therm')
    parser.add_argument('--scene_list', type=str, required=True,
                        help='Text file with one scan_id per line')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing scan subdirs with PLY')
    parser.add_argument('--shapenet_dir', type=str, required=True,
                        help='ShapeNet models directory')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory (objects_cad.json per scan)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='CrossOver checkpoint directory (with model.safetensors)')
    parser.add_argument('--i2pmae_ckpt', type=str,
                        default='checkpoints/pointbind_i2pmae.pt',
                        help='Path to I2PMAE weights (pointbind_i2pmae.pt)')
    parser.add_argument('--cad_cache', type=str,
                        default='scan2therm/cad_embeddings.npz',
                        help='Path for CAD embedding cache (.npz)')
    parser.add_argument('--csv_path', type=str,
                        default='3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv')
    parser.add_argument('--objects_json', type=str,
                        default='3DSSG/files/objects.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.scene_list) as f:
        scan_ids = [line.strip() for line in f if line.strip()]
    print(f"CrossOver CAD retrieval — {len(scan_ids)} scenes")

    # Load model
    print("Loading CrossOver model ...")
    model = load_crossover_model(args.ckpt, args.i2pmae_ckpt, args.device)

    # Build/load CAD database
    cad_db = build_cad_database(
        model, args.shapenet_dir, args.cad_cache, device=args.device)

    csv_path = args.csv_path if osp.isfile(args.csv_path) else None
    objects_json = args.objects_json if osp.isfile(args.objects_json) else None

    total_objects = 0
    total_cad = 0

    for i, scan_id in enumerate(scan_ids):
        print(f"[{i+1}/{len(scan_ids)}] {scan_id}")
        result = process_scan_crossover(
            scan_id=scan_id,
            model=model,
            cad_db=cad_db,
            data_dir=args.data_dir,
            shapenet_dir=args.shapenet_dir,
            out_dir=args.out_dir,
            csv_path=csv_path,
            objects_json_path=objects_json,
            device=args.device,
        )
        if result:
            total_objects += result['num_objects']
            total_cad += result['num_cad_augmented']

    print(f"\nDone. {total_objects} objects total, {total_cad} CrossOver-matched.")


if __name__ == '__main__':
    main()
