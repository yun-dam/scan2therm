"""
Thermal decay sanity checks for BEM output validation.

Validates that per-object thermal time constants and room-level thermal mass
fall within physically plausible ranges based on published reference data.
"""

import logging
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

# Default convective heat transfer coefficient for indoor air (W/(m^2*K))
DEFAULT_H_CONV = 8.0

# Expected thermal time constant ranges per category (seconds)
# tau = (rho * Cp * V) / (h * A)
EXPECTED_TAU_RANGES: Dict[str, Tuple[float, float]] = {
    'wood_furniture':       (300.0, 7200.0),
    'fabric_upholstery':    (60.0, 3600.0),
    'fabric_light':         (5.0, 300.0),
    'metal_appliance':      (10.0, 600.0),
    'ceramic_fixture':      (300.0, 10800.0),
    'glass':                (30.0, 1800.0),
    'stone_countertop':     (600.0, 14400.0),
    'plastic_electronics':  (30.0, 1200.0),
    'plastic_sheet':        (10.0, 600.0),
    'paper':                (5.0, 300.0),
    'cardboard':            (5.0, 300.0),
    'structural_concrete':  (3600.0, 86400.0),
}

# Expected total thermal mass ranges per room type (J/K)
# Total thermal mass = sum over objects of (rho * Cp * V)
EXPECTED_ROOM_THERMAL_MASS: Dict[str, Tuple[float, float]] = {
    'bedroom':     (50_000.0, 500_000.0),
    'kitchen':     (80_000.0, 800_000.0),
    'office':      (40_000.0, 400_000.0),
    'bathroom':    (30_000.0, 300_000.0),
    'living_room': (60_000.0, 600_000.0),
    'default':     (20_000.0, 1_000_000.0),
}


class ThermalDecayValidator:
    """Validates physical plausibility of BEM thermal outputs."""

    def __init__(self, h_conv: float = DEFAULT_H_CONV):
        """
        Args:
            h_conv: Convective heat transfer coefficient (W/(m^2*K)).
        """
        self.h_conv = h_conv

    def compute_thermal_time_constant(self, density: float,
                                      specific_heat: float,
                                      volume: float,
                                      surface_area: float) -> float:
        """Compute lumped thermal time constant.

        tau = (rho * Cp * V) / (h * A)

        Args:
            density: Material density (kg/m^3).
            specific_heat: Specific heat capacity (J/(kg*K)).
            volume: Object volume (m^3).
            surface_area: Object surface area (m^2).

        Returns:
            Time constant in seconds.
        """
        if surface_area <= 0 or self.h_conv <= 0:
            return float('inf')
        return (density * specific_heat * volume) / (self.h_conv * surface_area)

    def validate_single_object(self, obj: dict) -> Dict[str, object]:
        """Validate a single object entry from pipeline output.

        Args:
            obj: Object dict with 'thermal_category', 'material', 'geometry' keys.

        Returns:
            Dict with 'valid' bool, 'tau_seconds', 'expected_range', 'message'.
        """
        category = obj.get('thermal_category')
        mat = obj.get('material', {})
        geom = obj.get('geometry', {})

        density = mat.get('density', 0)
        specific_heat = mat.get('specific_heat', 0)
        volume = geom.get('volume_m3', 0)
        surface_area = geom.get('surface_area_m2', 0)

        if volume <= 0 or surface_area <= 0:
            return {
                'valid': False,
                'tau_seconds': None,
                'expected_range': None,
                'message': f"Object {obj.get('object_id')}: zero volume or surface area",
            }

        tau = self.compute_thermal_time_constant(density, specific_heat,
                                                 volume, surface_area)

        expected = EXPECTED_TAU_RANGES.get(category)
        if expected is None:
            return {
                'valid': True,
                'tau_seconds': tau,
                'expected_range': None,
                'message': f"Object {obj.get('object_id')}: no reference range for '{category}'",
            }

        tau_min, tau_max = expected
        valid = tau_min <= tau <= tau_max
        status = "OK" if valid else "OUT OF RANGE"
        msg = (f"Object {obj.get('object_id')} ({category}): "
               f"tau={tau:.1f}s, expected [{tau_min:.0f}, {tau_max:.0f}]s - {status}")

        return {
            'valid': valid,
            'tau_seconds': tau,
            'expected_range': list(expected),
            'message': msg,
        }

    def validate_room(self, summary_json: dict,
                      room_type: str = 'default') -> Dict[str, object]:
        """Validate room-level thermal mass against expected ranges.

        Args:
            summary_json: The pipeline output JSON (with 'summary' and 'objects').
            room_type: One of bedroom/kitchen/office/bathroom/living_room/default.

        Returns:
            Dict with 'valid', 'total_thermal_mass_JK', 'expected_range',
            'num_objects', 'object_results', 'message'.
        """
        objects = summary_json.get('objects', [])
        summary = summary_json.get('summary', {})
        total_mass = summary.get('total_thermal_mass_JK', 0)

        expected = EXPECTED_ROOM_THERMAL_MASS.get(
            room_type, EXPECTED_ROOM_THERMAL_MASS['default']
        )
        mass_min, mass_max = expected
        room_valid = mass_min <= total_mass <= mass_max

        # Validate individual objects
        object_results = []
        num_warnings = 0
        for obj in objects:
            result = self.validate_single_object(obj)
            object_results.append(result)
            if not result['valid']:
                num_warnings += 1
                log.warning(result['message'])

        status = "OK" if room_valid else "OUT OF RANGE"
        msg = (f"Room ({room_type}): total_thermal_mass={total_mass:.0f} J/K, "
               f"expected [{mass_min:.0f}, {mass_max:.0f}] J/K - {status}. "
               f"{num_warnings}/{len(objects)} objects with warnings.")

        if room_valid:
            log.info(msg)
        else:
            log.warning(msg)

        return {
            'valid': room_valid,
            'total_thermal_mass_JK': total_mass,
            'expected_range': list(expected),
            'num_objects': len(objects),
            'num_warnings': num_warnings,
            'object_results': object_results,
            'message': msg,
        }
