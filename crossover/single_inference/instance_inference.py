import os.path as osp
import argparse
from accelerate import Accelerator
from datetime import timedelta
import torch
from accelerate.utils import InitProcessGroupKwargs
import random
import numpy as np
import logging as log
import itertools
from typing import Dict, Any, List
from omegaconf import DictConfig

import sys
sys.path.append(osp.abspath('.'))

from model.scenelevel_enc import SceneLevelEncoder
from common.load_utils import load_npz_as_dict
from util import torch_util

from copy import deepcopy

DEFAULT_CONFIG = {
    'dataset': 'MultiScan',
    'process_dir': '/drive/dumps/multimodal-spaces/preprocess_feats/MultiScan',
    'ckpt': '/drive/dumps/multimodal-spaces/runs/new_runs/instance_crossover_scannet+scan3r+multiscan+arkitscenes.pth',
    'scan_id': 'scene_00004_00',
    'modalities': ['rgb', 'point', 'cad', 'referral'],
    'input_dim_3d': 384,
    'input_dim_2d': 1536,
    'input_dim_1d': 768,
    'out_dim': 768
}

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def load_single_scan_data(process_dir: str, scan_id: str, 
                         modalities: List[str]) -> Dict[str, Any]:
    """
    Load preprocessed data for a single scan, similar to ScanBase.__getitem__ but for one scan.
    
    Args:
        process_dir: Path to processed features directory
        scan_id: Scan ID to load
        modalities: List of modalities to load
        voxel_size: Voxel size for point cloud processing
        
    Returns:
        Dictionary containing scan data in the same format as ScanBase
    """
    scan_process_dir = osp.join(process_dir, 'scans', scan_id)
    
    # Load preprocessed data
    scan_objects_data = load_npz_as_dict(osp.join(scan_process_dir, 'objectsDataMultimodal.npz'))
    
    scene_dict = {}
    scene_dict['objects'] = {'inputs': {}, 'masks': {}, 'object_locs': {}}
    scene_dict['masks'] = {}
    
    for modality_name in modalities:
        inputs = torch.from_numpy(scan_objects_data['inputs'][modality_name]).float()
        
        if len(inputs.shape) > 2:
            mask = (inputs == 0).all(dim=2).all(dim=1)
        else:
            mask = torch.all(inputs == 0, dim=1)
        scene_dict['objects']['inputs'][modality_name] = inputs
        
        mask = ~mask
        scene_dict['objects']['masks'][modality_name] = mask
        
        if modality_name in scan_objects_data['object_locs']:
            scene_dict['objects']['object_locs'][modality_name] = torch.from_numpy(
                scan_objects_data['object_locs'][modality_name]
            ).float()
    
    num_objects = len(scan_objects_data['object_id2idx'].keys())
    obj_id_to_label_id_map = scan_objects_data['object_id_to_label_id_map']
    
    object_idx2id = {v: k for k, v in scan_objects_data['object_id2idx'].items()}
    object_ids = np.array([object_idx2id[object_idx] for object_idx in range(num_objects)])
    label_ids = np.array([obj_id_to_label_id_map[object_id] for object_id in object_ids])
    
    scene_dict['objects']['object_ids'] = torch.from_numpy(object_ids.astype(np.int32))
    scene_dict['objects']['label_ids'] = torch.from_numpy(label_ids.astype(np.int32))
    scene_dict['objects']['num_objects'] = object_ids.shape[0]
    
    scene_dict['scan_id'] = scan_id
    
    scene_dict['masks'] = deepcopy(scene_dict['objects']['masks'])
    scene_dict['label_ids'] = deepcopy(scene_dict['objects']['label_ids'])
    
    del scene_dict['objects']['masks']
    del scene_dict['objects']['label_ids']
    
    return scene_dict

def calculate_instance_similarities_single_scene(data_dict: Dict[str, Any], 
                                                 modalities: List[str],
                                                 logger) -> Dict[str, Dict[str, float]]:
    """
    Calculate cross-modal instance similarities within a single scene.
    Similar to ObjectRetrieval.eval() but for a single scene.
    
    Args:
        data_dict: Single scene data dict with embeddings and masks
        modalities: List of modalities to compare
        logger: Logger for output
    
    Returns:
        Dictionary containing similarity metrics for each modality combination
    """
    modality_combinations = list(itertools.combinations(modalities, 2))
    same_element_combinations = [(item, item) for item in modalities]
    modality_combinations.extend(same_element_combinations)
    
    metrics = {}
    for modality_combination in modality_combinations:
        src_modality, ref_modality = modality_combination
        metrics[src_modality + '_' + ref_modality] = {}
        metrics[src_modality + '_' + ref_modality]['instance'] = {
            'recall_top1': 0, 'recall_top3': 0, 'count': 0
        }
        metrics[src_modality + '_' + ref_modality]['scene'] = {
            'r@25': 0, 'r@50': 0, 'r@75': 0, 'r@100': 0, 'count': 0
        }
        metrics[src_modality + '_' + ref_modality]['matched_pairs'] = []
    
    for modality_combination in modality_combinations:
        src_modality, ref_modality = modality_combination
        
        if (src_modality not in data_dict['embeddings'] or 
            ref_modality not in data_dict['embeddings']):
            continue
            
        mask = torch.logical_and(
            data_dict['masks'][src_modality], 
            data_dict['masks'][ref_modality]
        )
        
        a_embed = data_dict['embeddings'][src_modality]
        b_embed = data_dict['embeddings'][ref_modality]
        
        if mask.sum() == 0:
            continue
        
        a_embed = a_embed[mask]
        b_embed = b_embed[mask]
        
        sim = torch.softmax(a_embed @ b_embed.t(), dim=-1)
        rank_list = torch.argsort(1.0 - sim, dim=1)
        top_k_indices = rank_list[:, :3]
        
        correct_index = torch.arange(a_embed.shape[0]).unsqueeze(1).to(top_k_indices.device)
        all_matches = top_k_indices == correct_index
        
        top1_matches = rank_list[:, 0]  # Top-1 match for each source object
        
        valid_object_ids = data_dict['object_ids'][mask].cpu().numpy()
        
        matched_pairs = [] # ---> [source_obj_id, matched_obj_id]
        for i, top1_idx in enumerate(top1_matches.cpu().numpy()):
            src_obj_id = valid_object_ids[i]
            matched_obj_id = valid_object_ids[top1_idx]
            matched_pairs.append([int(src_obj_id), int(matched_obj_id)])
        
        metrics[src_modality + '_' + ref_modality]['matched_pairs'] = matched_pairs
        
        recall_top1 = all_matches[:, 0].float().mean()
        recall_top3 = all_matches.any(dim=1).float().mean()
        
        ratio = all_matches[:, 0].float().sum() / a_embed.shape[0]
        
        if ratio >= 0.75:
            metrics[src_modality + '_' + ref_modality]['scene']['r@75'] += 1
        if ratio >= 0.5:
            metrics[src_modality + '_' + ref_modality]['scene']['r@50'] += 1
        if ratio >= 0.25:
            metrics[src_modality + '_' + ref_modality]['scene']['r@25'] += 1
        
        metrics[src_modality + '_' + ref_modality]['instance']['recall_top1'] += recall_top1
        metrics[src_modality + '_' + ref_modality]['instance']['recall_top3'] += recall_top3
        
        metrics[src_modality + '_' + ref_modality]['instance']['count'] += 1
        metrics[src_modality + '_' + ref_modality]['scene']['count'] += 1
    
    # Log results
    logger.info('Instance Matching ---')
    for modality_combination in modality_combinations:
        src_modality, ref_modality = modality_combination
        count = metrics[src_modality + '_' + ref_modality]['instance']['count']
        if count == 0:
            continue
        message = src_modality + '_' + ref_modality
        
        inst_recall_top1 = metrics[src_modality + '_' + ref_modality]['instance']['recall_top1'] / count * 100.
        inst_recall_top3 = metrics[src_modality + '_' + ref_modality]['instance']['recall_top3'] / count * 100.
        
        message += f'-> top1_recall: {inst_recall_top1:.2f} | top3_recall: {inst_recall_top3:.2f}'
        logger.info(message)
    
    # Scene Level Matching
    logger.info('Scene Level Matching ---')
    for modality_combination in modality_combinations:
        src_modality, ref_modality = modality_combination
        count = metrics[src_modality + '_' + ref_modality]['scene']['count']
        
        if count == 0:
            continue
        
        message = src_modality + '_' + ref_modality
        
        scene_recall_r25 = metrics[src_modality + '_' + ref_modality]['scene']['r@25'] / count * 100
        scene_recall_r50 = metrics[src_modality + '_' + ref_modality]['scene']['r@50'] / count * 100
        scene_recall_r75 = metrics[src_modality + '_' + ref_modality]['scene']['r@75'] / count * 100
        
        message += f'-> r@25: {scene_recall_r25:.2f} | r@50: {scene_recall_r50:.2f} | r@75: {scene_recall_r75:.2f}'
        
        logger.info(message)
    
    logger.info('Matched Object ID Pairs (Top-1 Matches) ---')
    for modality_combination in modality_combinations:
        src_modality, ref_modality = modality_combination
        matched_pairs = metrics[src_modality + '_' + ref_modality]['matched_pairs']
        
        if len(matched_pairs) == 0:
            continue
            
        pairs_str = ', '.join([f'[{pair[0]}, {pair[1]}]' for pair in matched_pairs])
        message = f'{src_modality.upper()} to {ref_modality.upper()}: {pairs_str}'
        logger.info(message)
    
    return metrics

def run_inference(args, scan_id=None):
    """Run instance-level cross-modal retrieval inference using preprocessed data"""
    
    if scan_id is None:
        raise ValueError("scan_id must be provided for instance inference")
    
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    kwargs = [init_kwargs]
    accelerator = Accelerator(kwargs_handlers=kwargs)
    
    model_config = DictConfig({
        'point': {'embed_dim': args.input_dim_3d},
        'cad': {'embed_dim': args.input_dim_3d},
        'image': {'embed_dim': args.input_dim_2d},
        'referral': {'embed_dim': args.input_dim_1d},
        'out_dim': args.out_dim
    })
    
    model = SceneLevelEncoder(model_config, args.modalities)
    model = accelerator.prepare(model)
    model.eval()
    model.to(accelerator.device)
    
    torch_util.load_weights(model, args.ckpt, accelerator.device)
    
    log.info(f"Running instance inference for scan: {scan_id}")
    log.info(f"Dataset: {args.dataset}")
    log.info(f"Available modalities: {args.modalities}")
    log.info(f"Process directory: {args.process_dir}")
    
    data_dict = load_single_scan_data(
        args.process_dir, scan_id, args.modalities
    )
    
    batch_data = {}
    for key, value in data_dict.items():
        if key not in ['pcl_coords', 'pcl_feats']:
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.unsqueeze(0)  
            elif isinstance(value, dict):
                batch_data[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        batch_data[key][sub_key] = {}
                        for nested_key, nested_value in sub_value.items():
                            if isinstance(nested_value, torch.Tensor):
                                batch_data[key][sub_key][nested_key] = nested_value.unsqueeze(0)
                            else:
                                batch_data[key][sub_key][nested_key] = [nested_value]
                    elif isinstance(sub_value, torch.Tensor):
                        batch_data[key][sub_key] = sub_value.unsqueeze(0)
                    else:
                        batch_data[key][sub_key] = [sub_value]
            else:
                batch_data[key] = [value]
    
    
    def move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            if obj.dtype == torch.float64:
                obj = obj.float() 
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_device(item, device) for item in obj]
        else:
            return obj
    
    batch_data = move_to_device(batch_data, accelerator.device)
    
    # Run inference
    log.info("Running model inference...")
    with torch.no_grad():
        output = model(batch_data)
    
    output_dict = {
        'scan_id': scan_id,
        'embeddings': {},
        'masks': {}
    }
    output_dict['label_ids'] = batch_data['label_ids'][0] 
    output_dict['object_ids'] = batch_data['objects']['object_ids'][0] 
    
    for modality in output['embeddings']:
        output_dict['masks'][modality] = batch_data['masks'][modality][0] 
        output_dict['embeddings'][modality] = output['embeddings'][modality][0] 
    
    log.info("Calculating cross-modal instance similarities...")
    similarity_results = calculate_instance_similarities_single_scene(
        output_dict, args.modalities, log
    )
    
    # log.info(similarity_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Instance-level Cross-modal Retrieval Inference')
    parser.add_argument('--dataset', default=DEFAULT_CONFIG['dataset'], type=str,
                        help=f"Dataset name (default: {DEFAULT_CONFIG['dataset']})")
    parser.add_argument('--process_dir', default=DEFAULT_CONFIG['process_dir'], type=str,
                        help=f"Path to processed features directory (default: {DEFAULT_CONFIG['process_dir']})")
    parser.add_argument('--ckpt', default=DEFAULT_CONFIG['ckpt'], type=str,
                        help=f"Path to model checkpoint (default: {DEFAULT_CONFIG['ckpt']})")
    parser.add_argument('--scan_id', default=DEFAULT_CONFIG['scan_id'], type=str,
                        help=f"Scan ID to run inference on (default: {DEFAULT_CONFIG['scan_id']})")
    
    # Model configuration
    parser.add_argument('--modalities', nargs='+', default=DEFAULT_CONFIG['modalities'],
                        help=f"List of modalities to use (default: {DEFAULT_CONFIG['modalities']})")
    parser.add_argument('--input_dim_3d', default=DEFAULT_CONFIG['input_dim_3d'], type=int,
                        help=f"Input dimension for 3D features (default: {DEFAULT_CONFIG['input_dim_3d']})")
    parser.add_argument('--input_dim_2d', default=DEFAULT_CONFIG['input_dim_2d'], type=int,
                        help=f"Input dimension for 2D features (default: {DEFAULT_CONFIG['input_dim_2d']})")
    parser.add_argument('--input_dim_1d', default=DEFAULT_CONFIG['input_dim_1d'], type=int,
                        help=f"Input dimension for 1D features (default: {DEFAULT_CONFIG['input_dim_1d']})")
    parser.add_argument('--out_dim', default=DEFAULT_CONFIG['out_dim'], type=int,
                        help=f"Output embedding dimension (default: {DEFAULT_CONFIG['out_dim']})")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    args = parser.parse_args()
    
    if not args.scan_id:
        raise ValueError("scan_id is required for instance inference")
    
    if not osp.exists(args.process_dir):
        raise ValueError(f"Process directory does not exist: {args.process_dir}")
        
    scan_process_dir = osp.join(args.process_dir, 'scans', args.scan_id)
    if not osp.exists(scan_process_dir):
        raise ValueError(f"Scan directory does not exist: {scan_process_dir}")
    
    run_inference(args, scan_id=args.scan_id)