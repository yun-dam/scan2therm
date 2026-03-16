"""
NYU40 to thermal material property lookup.

Maps 3RScan object labels to thermal properties via:
  PLY objectId -> globalId (528 classes) -> NYU40 ID (40 classes) -> thermal category -> (rho, Cp, k)
"""

import os.path as osp
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

from util.scan3r import read_label_mapping


@dataclass
class ThermalMaterial:
    """EnergyPlus-compatible thermal material properties."""
    name: str
    roughness: str  # VeryRough, Rough, MediumRough, MediumSmooth, Smooth, VerySmooth
    thickness: float  # m
    conductivity: float  # W/(m*K)
    density: float  # kg/m^3
    specific_heat: float  # J/(kg*K)
    thermal_absorptance: float  # 0-1
    solar_absorptance: float  # 0-1
    visible_absorptance: float  # 0-1

    def to_dict(self) -> dict:
        return asdict(self)


# ---- NYU40 ID -> thermal category mapping ----
# NYU40 classes: 1=wall, 2=floor, 3=cabinet, 4=bed, 5=chair, 6=sofa, 7=table,
# 8=door, 9=window, 10=bookshelf, 11=picture, 12=counter, 13=blinds, 14=desk,
# 15=shelves, 16=curtain, 17=dresser, 18=pillow, 19=mirror, 20=floor_mat,
# 21=clothes, 22=ceiling, 23=books, 24=refrigerator, 25=television, 26=paper,
# 27=towel, 28=shower_curtain, 29=box, 30=whiteboard, 31=person,
# 32=nightstand, 33=toilet, 34=sink, 35=lamp, 36=bathtub, 37=bag,
# 38=otherstructure, 39=otherfurniture, 40=otherprop

NYU40_TO_THERMAL_CATEGORY: Dict[int, str] = {
    1: 'structural_concrete',   # wall
    2: 'structural_concrete',   # floor
    3: 'wood_furniture',        # cabinet
    4: 'fabric_upholstery',     # bed
    5: 'wood_furniture',        # chair
    6: 'fabric_upholstery',     # sofa
    7: 'wood_furniture',        # table
    8: 'wood_furniture',        # door
    9: 'glass',                 # window
    10: 'wood_furniture',       # bookshelf
    11: 'glass',                # picture (framed glass)
    12: 'stone_countertop',     # counter
    13: 'fabric_light',         # blinds
    14: 'wood_furniture',       # desk
    15: 'wood_furniture',       # shelves
    16: 'fabric_light',         # curtain
    17: 'wood_furniture',       # dresser
    18: 'fabric_light',         # pillow
    19: 'glass',                # mirror
    20: 'fabric_light',         # floor_mat
    21: 'fabric_light',         # clothes
    22: 'structural_concrete',  # ceiling
    23: 'paper',                # books
    24: 'metal_appliance',      # refrigerator
    25: 'plastic_electronics',  # television
    26: 'paper',                # paper
    27: 'fabric_light',         # towel
    28: 'plastic_sheet',        # shower_curtain
    29: 'cardboard',            # box
    30: 'plastic_sheet',        # whiteboard
    31: None,                   # person (skip)
    32: 'wood_furniture',       # nightstand
    33: 'ceramic_fixture',      # toilet
    34: 'ceramic_fixture',      # sink
    35: 'plastic_electronics',  # lamp
    36: 'ceramic_fixture',      # bathtub
    37: 'fabric_light',         # bag
    38: 'structural_concrete',  # otherstructure
    39: 'wood_furniture',       # otherfurniture
    40: 'plastic_electronics',  # otherprop
}


# ---- Thermal material definitions with literature values ----
# Sources: ASHRAE Handbook of Fundamentals, EnergyPlus Engineering Reference,
# ISO 10456, MatWeb material property database

THERMAL_MATERIALS: Dict[str, ThermalMaterial] = {
    'wood_furniture': ThermalMaterial(
        name='Wood_Furniture_Oak',
        roughness='MediumSmooth',
        thickness=0.025,
        conductivity=0.15,      # W/(m*K) - oak/hardwood
        density=550.0,          # kg/m^3
        specific_heat=1630.0,   # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    ),
    'fabric_upholstery': ThermalMaterial(
        name='Fabric_Upholstery',
        roughness='Rough',
        thickness=0.10,
        conductivity=0.06,      # W/(m*K) - fabric + foam
        density=120.0,          # kg/m^3 - polyurethane foam composite
        specific_heat=1340.0,   # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.7,
        visible_absorptance=0.7,
    ),
    'fabric_light': ThermalMaterial(
        name='Fabric_Light',
        roughness='Rough',
        thickness=0.005,
        conductivity=0.04,      # W/(m*K)
        density=80.0,           # kg/m^3
        specific_heat=1340.0,   # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    ),
    'metal_appliance': ThermalMaterial(
        name='Metal_Appliance_Steel',
        roughness='Smooth',
        thickness=0.002,
        conductivity=45.0,      # W/(m*K) - steel
        density=7800.0,         # kg/m^3
        specific_heat=500.0,    # J/(kg*K)
        thermal_absorptance=0.3,
        solar_absorptance=0.3,
        visible_absorptance=0.3,
    ),
    'ceramic_fixture': ThermalMaterial(
        name='Ceramic_Fixture',
        roughness='MediumSmooth',
        thickness=0.015,
        conductivity=1.0,       # W/(m*K) - vitreous china
        density=2300.0,         # kg/m^3
        specific_heat=840.0,    # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.5,
        visible_absorptance=0.5,
    ),
    'glass': ThermalMaterial(
        name='Glass_Standard',
        roughness='VerySmooth',
        thickness=0.006,
        conductivity=0.9,       # W/(m*K) - soda-lime glass
        density=2500.0,         # kg/m^3
        specific_heat=840.0,    # J/(kg*K)
        thermal_absorptance=0.84,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    ),
    'stone_countertop': ThermalMaterial(
        name='Stone_Countertop_Granite',
        roughness='MediumRough',
        thickness=0.030,
        conductivity=2.8,       # W/(m*K) - granite
        density=2650.0,         # kg/m^3
        specific_heat=790.0,    # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.55,
        visible_absorptance=0.55,
    ),
    'plastic_electronics': ThermalMaterial(
        name='Plastic_ABS',
        roughness='Smooth',
        thickness=0.005,
        conductivity=0.17,      # W/(m*K) - ABS plastic
        density=1050.0,         # kg/m^3
        specific_heat=1400.0,   # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.5,
        visible_absorptance=0.5,
    ),
    'plastic_sheet': ThermalMaterial(
        name='Plastic_Sheet_PVC',
        roughness='Smooth',
        thickness=0.003,
        conductivity=0.16,      # W/(m*K) - PVC
        density=1380.0,         # kg/m^3
        specific_heat=1000.0,   # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.5,
        visible_absorptance=0.5,
    ),
    'paper': ThermalMaterial(
        name='Paper_Cellulose',
        roughness='MediumRough',
        thickness=0.003,
        conductivity=0.05,      # W/(m*K)
        density=700.0,          # kg/m^3
        specific_heat=1400.0,   # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    ),
    'cardboard': ThermalMaterial(
        name='Cardboard_Corrugated',
        roughness='MediumRough',
        thickness=0.005,
        conductivity=0.065,     # W/(m*K)
        density=200.0,          # kg/m^3
        specific_heat=1400.0,   # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    ),
    'structural_concrete': ThermalMaterial(
        name='Concrete_Standard',
        roughness='MediumRough',
        thickness=0.20,
        conductivity=1.4,       # W/(m*K) - medium-weight concrete
        density=2300.0,         # kg/m^3
        specific_heat=880.0,    # J/(kg*K)
        thermal_absorptance=0.9,
        solar_absorptance=0.6,
        visible_absorptance=0.6,
    ),
}


class ThermalDatabase:
    """Lookup thermal properties for 3RScan objects via NYU40 label mapping."""

    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: Path to '3RScan.v2 Semantic Classes - Mapping.csv'.
                      If provided, enables lookup_by_global_id.
        """
        self._global_to_nyu40: Optional[Dict[int, int]] = None
        if csv_path is not None:
            self._load_global_to_nyu40(csv_path)

    def _load_global_to_nyu40(self, csv_path: str) -> None:
        """Parse the 3RScan mapping CSV to build globalId -> NYU40 ID dict."""
        assert osp.isfile(csv_path), f"CSV not found: {csv_path}"
        # The CSV has header row: Global ID,Label,,NYU40 Mapping,...
        # The unnamed 3rd column contains the numeric NYU40 ID.
        # read_label_mapping works on named columns, so we parse manually
        # for the unnamed numeric NYU40 column.
        import csv
        self._global_to_nyu40 = {}
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # Skip the description header (rows 0-1) and column header (row 2)
                if i < 3:
                    continue
                if len(row) < 3:
                    continue
                try:
                    global_id = int(row[0].strip())
                    nyu40_id = int(row[2].strip())
                    self._global_to_nyu40[global_id] = nyu40_id
                except (ValueError, IndexError):
                    continue

    def lookup_by_nyu40(self, nyu40_id: int) -> Tuple[Optional[str], Optional[ThermalMaterial]]:
        """
        Look up thermal category and material by NYU40 ID.

        Returns:
            (category_name, ThermalMaterial) or (None, None) if unmapped.
        """
        category = NYU40_TO_THERMAL_CATEGORY.get(nyu40_id)
        if category is None:
            return None, None
        material = THERMAL_MATERIALS.get(category)
        return category, material

    def lookup_by_global_id(self, global_id: int) -> Tuple[Optional[str], Optional[ThermalMaterial]]:
        """
        Look up thermal properties via globalId -> NYU40 -> thermal category.
        Requires csv_path to have been provided at init.

        Returns:
            (category_name, ThermalMaterial) or (None, None) if unmapped.
        """
        if self._global_to_nyu40 is None:
            raise RuntimeError(
                "Global-to-NYU40 mapping not loaded. "
                "Provide csv_path when constructing ThermalDatabase."
            )
        nyu40_id = self._global_to_nyu40.get(global_id)
        if nyu40_id is None:
            return None, None
        return self.lookup_by_nyu40(nyu40_id)
