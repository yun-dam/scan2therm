import os.path as osp
import argparse
from accelerate import Accelerator
from datetime import timedelta
import torch
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm
import random
import numpy as np
import logging as log

import sys
sys.path.append(osp.abspath('.'))

from single_inference import datasets
from model.scene_crossover import SceneCrossOverModel
from util import torch_util

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

def run_inference(args, scan_id=None):
    if args.dataset == 'Scannet':
        dataset = datasets.ScannetInferDataset(args.data_dir, args.process_dir)
    elif args.dataset == 'Scan3R':
        dataset = datasets.Scan3RInferDataset(args.data_dir, args.process_dir)
    elif args.dataset == 'ARKitScenes':
        dataset = datasets.ARKitScenesInferDataset(args.data_dir, args.process_dir)
    elif args.dataset == 'MultiScan':
        dataset = datasets.MultiScanInferDataset(args.data_dir, args.process_dir)
    else:
        raise NotImplementedError('Dataset not implemented')
    
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    kwargs = [init_kwargs]
    accelerator = Accelerator(kwargs_handlers=kwargs)
    
    model = SceneCrossOverModel(args, accelerator.device)
    model = accelerator.prepare(model)
    model.eval()
    model.to(model.device)
    torch_util.load_weights(model, args.ckpt, accelerator.device)
    
    # # hack to calculate the number of parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # total_params += sum(p.numel() for p in model.encoder2D.dino_model.parameters())
    # total_params += sum(p.numel() for p in model.encoder1D.parameters())
    # print(f'Total number of parameters: {total_params}')
    # assert False
    
    if scan_id is not None:
        # Single scan inference
        data_dict = dataset[scan_id]
        with torch.no_grad():
            output = model(data_dict)
        
        output_np = {}
        for modality in output['embeddings']:
            output_np[modality] = output['embeddings'][modality].cpu().numpy()
        
        data = {'scene': [{'scan_id': scan_id, 'scene_embeds': output_np, 'masks': output['masks']}]}
        save_data = {
            'scene': data['scene']
        }
        np.savez(f'embed_{args.dataset.lower()}_{scan_id}.npz', **save_data)
        log.info(f'Saved embeddings for {scan_id}.')

    else:
        output_file = f'/drive/dumps/multimodal-spaces/v1.0_release/embed_{args.dataset.lower()}.npz'
        
        existing_data = {'scene': []}
        processed_scan_ids = set()
        
        if osp.exists(output_file):
            try:
                existing_npz = np.load(output_file, allow_pickle=True)
                existing_data = {'scene': existing_npz['scene'].tolist()}
                processed_scan_ids = {item['scan_id'] for item in existing_data['scene']}
                log.info(f'Loaded existing embeddings for {len(processed_scan_ids)} scans. Resuming from where we left off.')
            except Exception as e:
                log.warning(f'Could not load existing file {output_file}: {e}. Starting fresh.')
                existing_data = {'scene': []}
                processed_scan_ids = set()
        
        remaining_scans = [(idx, scan_id) for idx, scan_id in enumerate(dataset.scan_ids) 
                          if scan_id not in processed_scan_ids]
        
        if not remaining_scans:
            log.info('All scans already processed.')
            return
            
        log.info(f'Processing {len(remaining_scans)} remaining scans out of {len(dataset.scan_ids)} total scans.')
        
        for idx, scan_id in tqdm(remaining_scans, desc="Processing scans"):
            try:
                data_dict = dataset[idx]
                with torch.no_grad():
                    output = model(data_dict)
                    
                    output_np = {}
                    for modality in output['embeddings']:
                        output_np[modality] = output['embeddings'][modality].cpu().numpy()
                    
                    existing_data['scene'].append({
                        'scan_id': scan_id, 
                        'scene_embeds': output_np, 
                        'masks': output['masks']
                    })
                
                save_data = {
                    'scene': existing_data['scene']
                }
                np.savez_compressed(output_file, **save_data)
                log.info(f'Processed and saved scan {scan_id} ({len(existing_data["scene"])}/{len(dataset.scan_ids)} total).')
                
            except Exception as e:
                log.error(f'Error processing scan {scan_id}: {e}. Skipping and continuing.')
                continue
        
        log.info(f'Completed processing. Final embeddings saved for {len(existing_data["scene"])} scenes.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Inference')
    parser.add_argument('--dataset', default='Scan3R', type=str, required=False)
    parser.add_argument('--data_dir', default='/scratch/users/gauravp/datasets/Scan3R', type=str, required=False)
    parser.add_argument('--process_dir', default='/scratch/users/gauravp/dumps/preprocess_feats/Scan3R', type=str, required=False)
    parser.add_argument('--ckpt', default='/scratch/users/gauravp/ckpts/scene_crossover_scannet+scan3r+multiscan+arkitscenes_scratch.pth', type=str, required=False)
    parser.add_argument('--scan_id', default='', type=str, required=False)
    parser.add_argument('--input_dim_3d', default=512, type=int, required=False)
    parser.add_argument('--input_dim_2d', default=1536, type=int, required=False)
    parser.add_argument('--input_dim_1d', default=768, type=int, required=False)
    parser.add_argument('--out_dim', default=768, type=int, required=False)

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Parse the arguments
    args = parser.parse_args()
    run_inference(args, scan_id=None if args.scan_id == '' else args.scan_id)