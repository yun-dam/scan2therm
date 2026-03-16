"""
BEM (Building Energy Modeling) integration for CrossOver.

Extracts per-object thermal properties and geometry from 3RScan indoor scans,
producing structured JSON output for EnergyPlus InternalMass modeling.
"""

from bem.thermal_database import ThermalMaterial, ThermalDatabase
from bem.geometry_extractor import ObjectGeometry, GeometryExtractor
from bem.pipeline import CrossOverBEMPipeline
from bem.validation import ThermalDecayValidator
