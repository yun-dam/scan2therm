"""
End-to-end BEM pipeline: CrossOver inference -> geometry extraction -> thermal mapping.

Produces a structured JSON with per-object thermal properties and geometry
suitable for EnergyPlus InternalMass modeling.
"""

import os.path as osp
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from bem.thermal_database import ThermalDatabase, ThermalMaterial, NYU40_TO_THERMAL_CATEGORY
from bem.geometry_extractor import GeometryExtractor, ObjectGeometry
from bem.validation import ThermalDecayValidator

log = logging.getLogger(__name__)

# Default 3RScan semantic class mapping CSV
DEFAULT_CSV_PATH = '/home/yunda/3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv'


class CrossOverBEMPipeline:
    """Orchestrates geometry extraction, thermal mapping, and optional CrossOver inference."""

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: OmegaConf config with keys:
                - scan_data_dir: 3RScan dataset root (contains scan subdirs)
                - process_dir: Preprocessed features root
                - label_file_name: PLY annotation filename
                - csv_path: Path to 3RScan semantic class mapping CSV
                - ckpt_scene: (optional) Scene-level checkpoint path
                - ckpt_instance: (optional) Instance-level checkpoint path
                - modalities: List of modalities for instance inference
                - model dims: input_dim_3d, input_dim_2d, input_dim_1d, out_dim
                - min_points: Minimum points per object
                - skip_structural: Whether to skip wall/floor/ceiling
                - run_inference: Whether to run CrossOver model inference
        """
        self.cfg = cfg
        self.scan_data_dir = cfg.scan_data_dir
        self.process_dir = cfg.get('process_dir', '')
        self.label_file_name = cfg.get(
            'label_file_name', 'labels.instances.align.annotated.v2.ply'
        )
        csv_path = cfg.get('csv_path', DEFAULT_CSV_PATH)

        # Initialize thermal database
        self.thermal_db = ThermalDatabase(csv_path=csv_path)

        # Initialize validator
        self.validator = ThermalDecayValidator()

        # Model inference settings
        self._run_inference = cfg.get('run_inference', False)
        self._scene_model = None
        self._instance_model = None

    # ---- CrossOver inference (optional) ----

    def run_scene_inference(self, scan_id: str) -> Dict[str, Any]:
        """Run scene-level CrossOver inference.

        Follows the pattern in single_inference/scene_inference.py.

        Returns:
            Dict with 'embeddings' per modality and 'masks'.
        """
        from accelerate import Accelerator
        from model.scene_crossover import SceneCrossOverModel
        from single_inference import datasets
        from util import torch_util

        accelerator = Accelerator()

        if self._scene_model is None:
            model = SceneCrossOverModel(self.cfg, accelerator.device)
            model = accelerator.prepare(model)
            model.eval()
            model.to(accelerator.device)
            torch_util.load_weights(model, self.cfg.ckpt_scene, accelerator.device)
            self._scene_model = model

        dataset = datasets.Scan3RInferDataset(
            self.scan_data_dir, self.process_dir
        )
        data_dict = dataset[scan_id]

        with torch.no_grad():
            output = self._scene_model(data_dict)

        # Convert to numpy
        result = {'embeddings': {}, 'masks': output['masks']}
        for modality in output['embeddings']:
            result['embeddings'][modality] = (
                output['embeddings'][modality].cpu().numpy()
            )

        log.info(f"Scene inference complete for {scan_id}: "
                 f"{list(result['embeddings'].keys())} modalities")
        return result

    def run_instance_inference(self, scan_id: str) -> Dict[str, Any]:
        """Run instance-level CrossOver inference.

        Follows the pattern in single_inference/instance_inference.py.

        Returns:
            Dict with per-object 'embeddings', 'masks', 'object_ids', 'label_ids'.
        """
        from accelerate import Accelerator
        from model.scenelevel_enc import SceneLevelEncoder
        from single_inference.instance_inference import load_single_scan_data
        from util import torch_util
        from copy import deepcopy

        accelerator = Accelerator()
        modalities = self.cfg.get('modalities', ['rgb', 'point', 'cad', 'referral'])

        if self._instance_model is None:
            model_config = DictConfig({
                'point': {'embed_dim': self.cfg.get('input_dim_3d', 384)},
                'cad': {'embed_dim': self.cfg.get('input_dim_3d', 384)},
                'image': {'embed_dim': self.cfg.get('input_dim_2d', 1536)},
                'referral': {'embed_dim': self.cfg.get('input_dim_1d', 768)},
                'out_dim': self.cfg.get('out_dim', 768),
            })
            model = SceneLevelEncoder(model_config, modalities)
            model = accelerator.prepare(model)
            model.eval()
            model.to(accelerator.device)
            torch_util.load_weights(
                model, self.cfg.ckpt_instance, accelerator.device
            )
            self._instance_model = model

        data_dict = load_single_scan_data(self.process_dir, scan_id, modalities)

        # Batch dimension
        batch_data = _add_batch_dim(data_dict)
        batch_data = _move_to_device(batch_data, accelerator.device)

        with torch.no_grad():
            output = self._instance_model(batch_data)

        result = {
            'embeddings': {},
            'masks': {},
            'object_ids': batch_data['objects']['object_ids'][0].cpu().numpy(),
            'label_ids': batch_data['label_ids'][0].cpu().numpy(),
        }
        for modality in output['embeddings']:
            result['embeddings'][modality] = (
                output['embeddings'][modality][0].cpu().numpy()
            )
            result['masks'][modality] = (
                batch_data['masks'][modality][0].cpu().numpy()
            )

        log.info(f"Instance inference complete for {scan_id}: "
                 f"{result['object_ids'].shape[0]} objects")
        return result

    # ---- Geometry extraction ----

    def extract_geometry(self, scan_id: str) -> List[ObjectGeometry]:
        """Extract per-object geometry from 3RScan PLY.

        Returns:
            List of ObjectGeometry dataclasses.
        """
        extractor = GeometryExtractor(
            data_dir=osp.join(self.scan_data_dir, 'scans'),
            scan_id=scan_id,
            label_file_name=self.label_file_name,
        )
        return extractor.extract_all_objects(
            min_points=self.cfg.get('min_points', 50),
            skip_structural=self.cfg.get('skip_structural', True),
        )

    # ---- Thermal mapping ----

    def map_objects_to_thermal(
        self, object_geometries: List[ObjectGeometry]
    ) -> List[Dict[str, Any]]:
        """Map each object to its thermal material properties.

        Returns:
            List of dicts with 'object_id', 'thermal_category', 'material', 'geometry'.
        """
        results = []
        for geom in object_geometries:
            category, material = self.thermal_db.lookup_by_nyu40(geom.nyu40_id)

            if category is None:
                # Try via globalId as fallback
                category, material = self.thermal_db.lookup_by_global_id(geom.global_id)

            obj_entry = {
                'object_id': geom.object_id,
                'label_name': geom.label_name,
                'nyu40_id': geom.nyu40_id,
                'global_id': geom.global_id,
                'thermal_category': category,
                'material': material.to_dict() if material else None,
                'geometry': {
                    'volume_m3': geom.volume_m3,
                    'surface_area_m2': geom.surface_area_m2,
                    'centroid': geom.centroid,
                    'obb_dimensions': geom.obb_dimensions,
                    'num_points': geom.num_points,
                },
            }
            results.append(obj_entry)

        return results

    # ---- Main entry point ----

    def run(self, scan_id: str, room_type: str = 'default',
            validate: bool = True) -> Dict[str, Any]:
        """Run the full BEM pipeline for a scan.

        Args:
            scan_id: 3RScan scan identifier.
            room_type: Room type for validation (bedroom/kitchen/office/...).
            validate: Whether to run thermal decay validation.

        Returns:
            JSON-serializable dict with per-object thermal + geometry data.
        """
        log.info(f"Starting BEM pipeline for scan {scan_id}")

        # Step 1: Optional CrossOver inference
        inference_data = {}
        if self._run_inference:
            if self.cfg.get('ckpt_scene'):
                inference_data['scene'] = self.run_scene_inference(scan_id)
            if self.cfg.get('ckpt_instance'):
                inference_data['instance'] = self.run_instance_inference(scan_id)

        # Step 2: Geometry extraction
        log.info("Extracting per-object geometry...")
        geometries = self.extract_geometry(scan_id)
        log.info(f"Extracted geometry for {len(geometries)} objects")

        # Step 3: Thermal mapping
        log.info("Mapping objects to thermal properties...")
        objects = self.map_objects_to_thermal(geometries)
        log.info(f"Mapped {len(objects)} objects to thermal categories")

        # Step 4: Compute summary statistics
        total_thermal_mass = 0.0
        total_surface_area = 0.0
        for obj in objects:
            mat = obj.get('material')
            geom = obj.get('geometry', {})
            if mat and geom.get('volume_m3'):
                total_thermal_mass += (
                    mat['density'] * mat['specific_heat'] * geom['volume_m3']
                )
            total_surface_area += geom.get('surface_area_m2', 0)

        output = {
            'scan_id': scan_id,
            'zone_name': f'Zone_{scan_id}',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'objects': objects,
            'summary': {
                'total_thermal_mass_JK': total_thermal_mass,
                'total_surface_area_m2': total_surface_area,
                'num_objects': len(objects),
            },
        }

        # Step 5: Validation
        if validate:
            log.info("Running thermal validation...")
            validation = self.validator.validate_room(output, room_type)
            output['validation'] = {
                'room_valid': validation['valid'],
                'total_thermal_mass_expected_range': validation['expected_range'],
                'num_warnings': validation['num_warnings'],
                'message': validation['message'],
            }

        log.info(f"BEM pipeline complete: {len(objects)} objects, "
                 f"total thermal mass = {total_thermal_mass:.0f} J/K")
        return output


# ---- Helpers for batching / device transfer ----

def _add_batch_dim(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add batch dimension to tensors, following instance_inference.py pattern."""
    batch = {}
    for key, value in data_dict.items():
        if key in ('pcl_coords', 'pcl_feats'):
            continue
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        elif isinstance(value, dict):
            batch[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    batch[key][sub_key] = {}
                    for nk, nv in sub_value.items():
                        if isinstance(nv, torch.Tensor):
                            batch[key][sub_key][nk] = nv.unsqueeze(0)
                        else:
                            batch[key][sub_key][nk] = [nv]
                elif isinstance(sub_value, torch.Tensor):
                    batch[key][sub_key] = sub_value.unsqueeze(0)
                else:
                    batch[key][sub_key] = [sub_value]
        else:
            batch[key] = [value]
    return batch


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors to device."""
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.float64:
            obj = obj.float()
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_move_to_device(item, device) for item in obj]
    return obj
