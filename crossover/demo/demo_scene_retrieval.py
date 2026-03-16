import os.path as osp
import os
import argparse
from accelerate import Accelerator
from datetime import timedelta
import torch
from accelerate.utils import InitProcessGroupKwargs
import random
import numpy as np
import open3d as o3d
import logging as log
from PIL import Image
from torchvision import transforms as tvf
from typing import Dict, List, Tuple
import sys
sys.path.append(osp.abspath('.'))
from model.scene_crossover import SceneCrossOverModel
from util import torch_util
from common.load_utils import load_npz_as_dict, load_yaml
import albumentations as A

DEFAULT_CONFIG = {
    'query_path': './demo_data/dining_room/scene_cropped.ply',
    'database_path': './embed_scannet.npz',
    'query_modality': 'point',
    'database_modality': 'point',
    'ckpt': '/drive/dumps/multimodal-spaces/runs/new_runs/rgb/scene_crossover_scannet+scan3r+multiscan+arkitscenes_scratch.pth',
    'top_k': 5,
    'color_stats_path': '/drive/dumps/multimodal-spaces/preprocess_feats/Scannet/color_mean_std.yaml'
}

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

class SceneRetrieval:
    """Scene-level cross-modal retrieval"""
    
    def __init__(self, args):
        self.args = args
        self.setup_model()
        
        self.image_transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
    
    def setup_model(self):
        """Initialize and load the model"""
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = [init_kwargs]
        self.accelerator = Accelerator(kwargs_handlers=kwargs)
        
        self.model = SceneCrossOverModel(self.args, self.accelerator.device)
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        self.model.to(self.model.device)
        torch_util.load_weights(self.model, self.args.ckpt, self.accelerator.device)
        
        log.info(f"Model loaded from {self.args.ckpt}")

    def encode_query(self, query_path: str, query_modality: str, color_stats_path: str, voxel_size: float = 0.02, image_size: List[int] = [224, 224]) -> torch.Tensor:
        """Encode query and return embedding"""
        
        if query_modality == 'point':
            return self._encode_point_query(query_path, color_stats_path, voxel_size)
        elif query_modality == 'rgb':
            return self._encode_rgb_query(query_path, image_size)
        elif query_modality == 'floorplan':
            return self._encode_floorplan_query(query_path, image_size)
        elif query_modality == 'referral':
            return self._encode_referral_query(query_path)
        else:
            raise NotImplementedError(f'Query modality {query_modality} not implemented')

    def _encode_point_query(self, path: str, color_stats_path: str, voxel_size: float = 0.02) -> torch.Tensor:
        """Encode point cloud query"""
        assert path.endswith('.ply'), 'Point Cloud Path should be a .ply file!'

        color_mean_std = load_yaml(color_stats_path)
        print(color_stats_path, color_mean_std)
        color_mean, color_std = (
            tuple(color_mean_std["mean"]),
            tuple(color_mean_std["std"]),
        )
        normalize_color = A.Normalize(mean=color_mean, std=color_std)
        assert path.endswith('.ply'), 'Point Cloud Path should be a .ply file!'
        pcd = o3d.io.read_point_cloud(path) 
        points = np.asarray(pcd.points)
        feats = np.asarray(pcd.colors)*255.0
        feats = feats.round()
        pseudo_image = feats.astype(np.uint8)[np.newaxis, :, :]
        feats = np.squeeze(normalize_color(image=pseudo_image)["image"])
        
        coords, feats = torch_util.convert_to_sparse_tensor(points, feats, voxel_size)
        with torch.no_grad():
            embed = self.model.encode_3d(coords, feats).cpu()
        
        return embed
    
    def _encode_rgb_query(self, path: str, image_size: List[int]) -> torch.Tensor:
        """Encode RGB image query"""
        assert os.isdir(path), 'RGB Path should be a directory'
        
        image_filenames = os.listdir(path)
        image_data = None
        
        for image_filename in image_filenames:
            image = Image.open(osp.join(path, image_filename))
            image = image.resize((image_size[1], image_size[0]), Image.BICUBIC)
            image_pt = self.image_transform(image).unsqueeze(0)
            image_data = image_pt if image_data is None else torch.cat((image_data, image_pt), dim=0)

        with torch.no_grad():
            embed = self.model.encode_rgb(image_data.to(self.model.device)).cpu()
        
        return embed
    
    def _encode_floorplan_query(self, path: str, image_size: List[int]) -> torch.Tensor:
        """Encode floorplan query"""
        assert os.isfile(path), 'Floorplan Path should be a file'
        
        floorplan_img = Image.open(path)
        floorplan_img = floorplan_img.resize((image_size[1], image_size[0]), Image.BICUBIC)
        floorplan_data = self.image_transform(floorplan_img).unsqueeze(0)
        
        with torch.no_grad():
            embed = self.model.encode_floorplan(floorplan_data.to(self.model.device)).cpu()
        
        return embed
    
    def _encode_referral_query(self, path: str) -> torch.Tensor:
        """Encode text referral query"""
        assert os.isfile(path), 'Referral Path should be a text file'
        with open(path, 'r') as f:
            text = f.read()
        
        text = [text]
        with torch.no_grad():
            embed = self.model.encode_1d(text).cpu()
        
        return embed
    
    def load_database(self, database_path: str) -> Dict:
        """Load database embeddings"""
        database_embeds = load_npz_as_dict(database_path)
        log.info(f'Loaded database embeddings from {database_path}')
        return database_embeds
    
    def retrieve(
        self,
        query_path: str,
        query_modality: str,
        database_path: str,
        database_modality: str,
        color_stats_path: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Main retrieval function
        Returns list of (scan_id, similarity_score) tuples
        """
        
        log.info(f"Query: {query_path} ({query_modality})")
        log.info(f"Database: {database_path}")
        log.info(f"Target modality: {database_modality}")
        
        # Encode query
        query_embed = self.encode_query(query_path, query_modality, color_stats_path)
        log.info(f"Query embedding shape: {query_embed.shape}")
        
        # Load database
        database_embeds = self.load_database(database_path)
        
        scene_data = database_embeds['scene']
        if isinstance(scene_data, np.ndarray):
            scene_data = scene_data.tolist()
        
        query_database_embeds = torch.from_numpy(np.array([embed_dict['scene_embeds'][database_modality].reshape(-1,) for embed_dict in scene_data]))
        embed_masks = torch.tensor([embed_dict['masks'][database_modality] for embed_dict in scene_data])
        
        query_database_embeds = query_database_embeds[embed_masks]
        
        if len(query_database_embeds) == 0:
            log.warning("No valid embeddings found in database")
            return []
        
        # Compute similarities
        sim = torch.softmax(query_embed @ query_database_embeds.t(), dim=-1)
        rank_list = torch.argsort(1.0 - sim, dim=1)
        top_k_indices = rank_list[0, :top_k]
        
        retrieved_scan_ids = [scene_data[idx]['scan_id'] for idx in top_k_indices][:5]
        message = f'Query modality: {query_modality}, Database modality: {database_modality}'
        message += f"\n Top 5 retrieved scans: {retrieved_scan_ids}"
        log.info(message)

def main():
    parser = argparse.ArgumentParser(
        description='Cross Modal Scene Retrieval',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Query arguments with defaults from config
    parser.add_argument('--query_path', type=str, default=DEFAULT_CONFIG['query_path'],
                       help=f'Path to query (point cloud, images, floorplan, or text) - default: {DEFAULT_CONFIG["query_path"]}')
    parser.add_argument('--query_modality', type=str, default=DEFAULT_CONFIG['query_modality'],
                       choices=['point', 'rgb', 'floorplan', 'referral'],
                       help=f'Query modality - default: {DEFAULT_CONFIG["query_modality"]}')
    parser.add_argument('--database_path', type=str, default=DEFAULT_CONFIG['database_path'],
                       help=f'Path to database embeddings - default: {DEFAULT_CONFIG["database_path"]}')
    parser.add_argument('--database_modality', type=str, default=DEFAULT_CONFIG['database_modality'],
                       choices=['point', 'rgb', 'floorplan', 'referral'],
                       help=f'Database modality to match against - default: {DEFAULT_CONFIG["database_modality"]}')
    parser.add_argument('--color_stats_path', type=str, default=DEFAULT_CONFIG['color_stats_path'],
                       help=f'Path to color statistics - default: {DEFAULT_CONFIG["color_stats_path"]}')

    # Model arguments with defaults from config
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CONFIG['ckpt'],
                       help=f'Path to model checkpoint - default: {DEFAULT_CONFIG["ckpt"]}')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=DEFAULT_CONFIG['top_k'],
                       help=f'Number of top results to return - default: {DEFAULT_CONFIG["top_k"]}')
    
    # Model dimensions
    parser.add_argument('--input_dim_3d', type=int, default=512)
    parser.add_argument('--input_dim_2d', type=int, default=1536)
    parser.add_argument('--input_dim_1d', type=int, default=768)
    parser.add_argument('--out_dim', type=int, default=768)
    
    args = parser.parse_args()
    
    # Print configuration being used
    log.info("=== Scene Retrieval Configuration ===")
    log.info(f"Model checkpoint: {args.ckpt}")
    log.info(f"Query: {args.query_path} ({args.query_modality})")
    log.info(f"Database: {args.database_path}")
    log.info(f"Database modality: {args.database_modality}")
    log.info(f"Top-K: {args.top_k}")
    log.info("=====================================")
    
    # Basic validation
    if not osp.exists(args.query_path) and args.query_modality != 'referral':
        log.warning(f"Query path does not exist: {args.query_path}")
    if not osp.exists(args.database_path):
        log.warning(f"Database path does not exist: {args.database_path}")
    if not osp.exists(args.ckpt):
        log.warning(f"Checkpoint does not exist: {args.ckpt}")
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Run retrieval
    retriever = SceneRetrieval(args)
    retriever.retrieve(
        args.query_path,
        args.query_modality,
        args.database_path,
        args.database_modality,
        args.color_stats_path,
        args.top_k
    )
    
    
if __name__ == '__main__':
    main()