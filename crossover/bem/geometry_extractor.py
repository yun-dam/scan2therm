"""
Per-object geometry extraction from 3RScan annotated PLY files.

Computes volume, surface area, centroid, and OBB dimensions for each
object instance using convex hull analysis.
"""

import os.path as osp
import numpy as np
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import trimesh

from util.scan3r import load_ply_data
from util.point_cloud import get_object_loc_box

log = logging.getLogger(__name__)

# NYU40 IDs for structural elements to skip
STRUCTURAL_NYU40_IDS = {1, 2, 22}  # wall, floor, ceiling


@dataclass
class ObjectGeometry:
    """Geometric measurements for a single object instance."""
    object_id: int
    label_name: str
    global_id: int
    nyu40_id: int
    centroid: List[float]       # [x, y, z] in meters
    obb_dimensions: List[float] # [length, width, height] in meters
    volume_m3: float
    surface_area_m2: float
    num_points: int

    def to_dict(self) -> dict:
        return asdict(self)


class GeometryExtractor:
    """Extracts per-object geometry from 3RScan PLY scans."""

    def __init__(self, data_dir: str, scan_id: str,
                 label_file_name: str = 'labels.instances.align.annotated.v2.ply'):
        """
        Args:
            data_dir: Root directory of 3RScan dataset (contains scan_id subdirs).
            scan_id: Scan identifier (e.g., '0a4b8ef6-...').
            label_file_name: PLY filename with semantic annotations.
        """
        self.data_dir = data_dir
        self.scan_id = scan_id
        self.label_file_name = label_file_name
        self.vertices: Optional[np.ndarray] = None

    def load_scan_data(self) -> np.ndarray:
        """Load annotated PLY data for the scan.

        Returns:
            Structured numpy array with fields:
            (x, y, z, red, green, blue, objectId, globalId, NYU40, Eigen13, RIO27)
        """
        log.info(f"Loading PLY data for scan {self.scan_id}")
        self.vertices = load_ply_data(self.data_dir, self.scan_id, self.label_file_name)
        log.info(f"Loaded {len(self.vertices)} vertices")
        return self.vertices

    def extract_object_geometry(self, object_id: int) -> Optional[ObjectGeometry]:
        """Extract geometry for a single object instance.

        Args:
            object_id: The objectId value from the PLY file.

        Returns:
            ObjectGeometry or None if too few points.
        """
        if self.vertices is None:
            raise RuntimeError("Call load_scan_data() first.")

        mask = self.vertices['objectId'] == object_id
        obj_points = np.column_stack([
            self.vertices['x'][mask],
            self.vertices['y'][mask],
            self.vertices['z'][mask],
        ])

        if len(obj_points) < 4:
            return None

        # Get label info from first vertex of this object
        idx = np.where(mask)[0][0]
        global_id = int(self.vertices['globalId'][idx])
        nyu40_id = int(self.vertices['NYU40'][idx])

        # Use existing utility for centroid and OBB
        obj_loc, obj_box = get_object_loc_box(obj_points)
        centroid = obj_loc[:3].tolist()
        obb_dims = obj_box[3:6].tolist()  # [size_x, size_y, size_z]

        # Compute volume and surface area via convex hull
        volume, surface_area = self._compute_hull_geometry(obj_points, obb_dims)

        # Build a label name from global_id (numeric fallback)
        label_name = f"globalId_{global_id}"

        return ObjectGeometry(
            object_id=int(object_id),
            label_name=label_name,
            global_id=global_id,
            nyu40_id=nyu40_id,
            centroid=centroid,
            obb_dimensions=obb_dims,
            volume_m3=volume,
            surface_area_m2=surface_area,
            num_points=len(obj_points),
        )

    def _compute_hull_geometry(self, points: np.ndarray,
                               obb_dims: List[float]) -> tuple:
        """Compute volume and surface area from convex hull, with OBB fallback.

        Args:
            points: (N, 3) point cloud.
            obb_dims: [dx, dy, dz] OBB dimensions for fallback.

        Returns:
            (volume_m3, surface_area_m2)
        """
        try:
            cloud = trimesh.PointCloud(points)
            hull = cloud.convex_hull
            return float(hull.volume), float(hull.area)
        except Exception:
            # Fallback: axis-aligned bounding box
            dx, dy, dz = obb_dims
            volume = dx * dy * dz
            surface_area = 2.0 * (dx * dy + dy * dz + dx * dz)
            return volume, surface_area

    def extract_all_objects(self, min_points: int = 50,
                            skip_structural: bool = True) -> List[ObjectGeometry]:
        """Extract geometry for all object instances in the scan.

        Args:
            min_points: Minimum point count to include an object.
            skip_structural: If True, skip wall/floor/ceiling (NYU40 IDs 1,2,22).

        Returns:
            List of ObjectGeometry for qualifying objects.
        """
        if self.vertices is None:
            self.load_scan_data()

        unique_ids = np.unique(self.vertices['objectId'])
        results = []

        for obj_id in unique_ids:
            if obj_id == 0:
                continue  # skip unlabeled

            mask = self.vertices['objectId'] == obj_id
            count = mask.sum()
            if count < min_points:
                continue

            if skip_structural:
                nyu40_id = int(self.vertices['NYU40'][np.where(mask)[0][0]])
                if nyu40_id in STRUCTURAL_NYU40_IDS:
                    continue

            geom = self.extract_object_geometry(obj_id)
            if geom is not None:
                results.append(geom)

        log.info(f"Extracted geometry for {len(results)} objects "
                 f"(from {len(unique_ids)} unique IDs)")
        return results
