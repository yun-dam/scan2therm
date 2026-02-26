#!/usr/bin/env python3
"""
Inject per-object InternalMass into an EnergyPlus IDF from a scan2therm mapping JSON.

For each object in each zone:
  1. Creates a Material with thermal properties from the material library
  2. Creates a Construction referencing that Material
  3. Creates an InternalMass in the zone using that Construction and the object's surface area

Existing generic InternalMass objects are preserved. Per-object entries are added
alongside them.

Usage:
    python scan2therm/inject_internal_mass.py \
        --json scan2therm/small_office_mapping.json \
        --idf scan2therm/energyplus/small-office.idf \
        --idd scan2therm/energyplus/Energy+.idd \
        --output scan2therm/energyplus/small-office-bem.idf
"""

import argparse
import json
import os.path as osp
import sys

from eppy.modeleditor import IDF

# Material library — matches categories in the mapping JSON.
# Values from scan2therm/material_library.txt (ASHRAE 2005 HOF & CIBSE Guide A)
MATERIAL_LIBRARY = {
    "Wood": {
        "Conductivity": 0.15,
        "Density": 608,
        "Specific_Heat": 1630,
        "Roughness": "MediumSmooth",
    },
    "Concrete": {
        "Conductivity": 1.95,
        "Density": 2240,
        "Specific_Heat": 900,
        "Roughness": "MediumRough",
    },
    "Metal": {
        "Conductivity": 45.28,
        "Density": 7824,
        "Specific_Heat": 500,
        "Roughness": "Smooth",
    },
    "Books": {
        "Conductivity": 0.15,
        "Density": 850,
        "Specific_Heat": 1400,
        "Roughness": "MediumSmooth",
    },
    "Gypsum": {
        "Conductivity": 0.16,
        "Density": 800,
        "Specific_Heat": 1090,
        "Roughness": "MediumSmooth",
    },
    "Plastic": {
        "Conductivity": 0.2,
        "Density": 1000,
        "Specific_Heat": 1500,
        "Roughness": "Smooth",
    },
    "Fabric": {
        "Conductivity": 0.06,
        "Density": 288,
        "Specific_Heat": 1380,
        "Roughness": "Rough",
    },
}

# Default thickness (m) per category — representative effective thickness
# for internal mass objects (Volume / Surface_Area used when available)
DEFAULT_THICKNESS = {
    "Wood": 0.025,
    "Concrete": 0.10,
    "Metal": 0.002,
    "Books": 0.05,
    "Gypsum": 0.013,
    "Plastic": 0.005,
    "Fabric": 0.01,
}


def compute_thickness(obj):
    """Compute effective thickness from Volume / Surface_Area, with fallback."""
    geom = obj["Geometry"]
    volume = geom.get("Volume_m3", 0)
    area = geom.get("Surface_Area_m2", 0)
    category = obj["Material_Info"]["Category"]

    if volume > 0 and area > 0:
        return volume / area
    return DEFAULT_THICKNESS.get(category, 0.025)


def inject(idf, mapping):
    """Inject Material, Construction, and InternalMass objects from mapping JSON."""
    zones = mapping["Zones"]

    # Keep existing InternalMass objects
    existing = [im.Name for im in idf.idfobjects["INTERNALMASS"]]
    print(f"Keeping {len(existing)} existing InternalMass object(s): {existing}")

    # Track which materials/constructions we've already added to avoid duplicates
    added_materials = set()
    added_constructions = set()
    total_im = 0

    for zone in zones:
        zone_id = zone["Zone_ID"]
        zone_name = zone["Zone_Name"]
        print(f"\nZone: {zone_name} ({zone_id})")

        for obj in zone["Objects"]:
            obj_id = obj["ID"]
            obj_type = obj["Type"]
            category = obj["Material_Info"]["Category"]
            surface_area = obj["Geometry"]["Surface_Area_m2"]
            thickness = compute_thickness(obj)

            mat_props = MATERIAL_LIBRARY.get(category)
            if mat_props is None:
                print(f"  WARNING: Unknown material category '{category}' "
                      f"for {obj_id}, skipping")
                continue

            # Material name: e.g. "Mat_Z001_O001_Wood_0.0294m"
            mat_name = f"Mat_{obj_id}_{category}_{thickness:.4f}m"

            new_mat = idf.newidfobject("MATERIAL")
            new_mat.Name = mat_name
            new_mat.Roughness = mat_props["Roughness"]
            new_mat.Thickness = round(thickness, 6)
            new_mat.Conductivity = mat_props["Conductivity"]
            new_mat.Density = mat_props["Density"]
            new_mat.Specific_Heat = mat_props["Specific_Heat"]
            new_mat.Thermal_Absorptance = 0.9
            new_mat.Solar_Absorptance = 0.7
            new_mat.Visible_Absorptance = 0.7
            added_materials.add(mat_name)

            # Construction name: e.g. "Constr_Z001_O001_Wood_0.0294m"
            constr_name = f"Constr_{obj_id}_{category}_{thickness:.4f}m"

            new_constr = idf.newidfobject("CONSTRUCTION")
            new_constr.Name = constr_name
            new_constr.Outside_Layer = mat_name
            added_constructions.add(constr_name)

            # InternalMass — one per object
            im_name = f"{zone_name}_{obj_id}_{obj_type}"
            new_im = idf.newidfobject("INTERNALMASS")
            new_im.Name = im_name
            new_im.Construction_Name = constr_name
            new_im.Zone_or_ZoneList_Name = zone_name
            new_im.Surface_Area = round(surface_area, 4)
            total_im += 1

            print(f"  {im_name}: {category}, "
                  f"t={thickness:.4f}m, A={surface_area}m2")

    print(f"\nTotal: {len(added_materials)} Material(s), "
          f"{len(added_constructions)} Construction(s), "
          f"{total_im} InternalMass object(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Inject per-object InternalMass into EnergyPlus IDF"
    )
    parser.add_argument("--json", required=True,
                        help="Path to scan2therm mapping JSON")
    parser.add_argument("--idf", required=True,
                        help="Path to input EnergyPlus IDF")
    parser.add_argument("--idd", required=True,
                        help="Path to Energy+.idd")
    parser.add_argument("--output", required=True,
                        help="Path to write modified IDF")
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.json, "JSON"), (args.idf, "IDF"), (args.idd, "IDD")]:
        if not osp.isfile(path):
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    # Load mapping JSON
    with open(args.json) as f:
        mapping = json.load(f)

    # Initialize eppy with IDD
    IDF.setiddname(args.idd)
    idf = IDF(args.idf)

    # Verify zone names exist in IDF
    idf_zone_names = {z.Name for z in idf.idfobjects["ZONE"]}
    for zone in mapping["Zones"]:
        zn = zone["Zone_Name"]
        if zn not in idf_zone_names:
            print(f"WARNING: Zone '{zn}' from JSON not found in IDF. "
                  f"Available: {idf_zone_names}")

    # Inject
    inject(idf, mapping)

    # Save
    idf.saveas(args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
