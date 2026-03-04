#!/usr/bin/env python3
"""
scan2therm pipeline: 3RScan raw data → EnergyPlus InternalMass injection.

End-to-end pipeline that processes 3RScan scenes through four stages:
  1. Extract cropped 2D object images from RGB sequences
  2. Extract per-object geometry + augment with CAD-based surface area & volume
  3. Classify materials via Vertex AI Gemini VLM
  4. Build zone mapping JSON and inject InternalMass into EnergyPlus IDF

Each stage is idempotent — it checks for existing outputs and skips if found.
Stages can be run selectively via --steps.

Usage:
    # Full pipeline (all 4 steps):
    python scan2therm/main.py \
        --scene_list scan2therm/office_scenes_105.txt \
        --rscan_dir 3DSSG/data/3RScan/data/3RScan \
        --out_dir scan2therm/object_images \
        --shapenet_dir Scan2CAD/Assets/shapenet-sample \
        --gcp_project hai-gcp-a-model \
        --idf scan2therm/energyplus/small-office.idf \
        --idd scan2therm/energyplus/Energy+.idd \
        --idf_output scan2therm/energyplus/small-office-bem.idf

    # Run only specific steps:
    python scan2therm/main.py ... --steps 2 3
    python scan2therm/main.py ... --steps 4 --zone_name Core_ZN

Raw inputs:
    3DSSG/data/3RScan/data/3RScan/{scan_id}/
        ├── mesh.refined.v2.obj          (3D mesh)
        ├── labels.instances.annotated.v2.ply  (instance labels)
        └── sequence/                    (RGB frames + camera poses)
            ├── _info.txt
            ├── frame-000000.color.jpg
            ├── frame-000000.pose.txt
            └── ...
    Scan2CAD/Assets/shapenet-sample/{synset}/{model_id}/models/model_normalized.obj
    3DSSG/files/objects.json             (3DSSG material/color/shape attributes)
    3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv
"""

import sys
import os
import os.path as osp
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Material category normalization (VLM raw output → inject_internal_mass categories)
# ---------------------------------------------------------------------------

MATERIAL_CATEGORY_MAP = {
    # Wood
    'wood': 'Wood', 'wooden': 'Wood', 'timber': 'Wood', 'bamboo': 'Wood',
    'plywood': 'Wood', 'mdf': 'Wood', 'particleboard': 'Wood', 'oak': 'Wood',
    'pine': 'Wood', 'walnut': 'Wood', 'veneer': 'Wood', 'laminate': 'Wood',
    # Concrete
    'concrete': 'Concrete', 'cement': 'Concrete', 'cinder block': 'Concrete',
    # Metal
    'metal': 'Metal', 'steel': 'Metal', 'iron': 'Metal', 'aluminum': 'Metal',
    'aluminium': 'Metal', 'brass': 'Metal', 'copper': 'Metal', 'chrome': 'Metal',
    'stainless steel': 'Metal',
    # Plastic
    'plastic': 'Plastic', 'acrylic': 'Plastic', 'vinyl': 'Plastic',
    'polycarbonate': 'Plastic', 'abs': 'Plastic', 'nylon': 'Plastic',
    'rubber': 'Plastic', 'silicone': 'Plastic', 'resin': 'Plastic',
    # Fabric / textile
    'fabric': 'Fabric', 'textile': 'Fabric', 'cloth': 'Fabric',
    'leather': 'Fabric', 'cotton': 'Fabric', 'polyester': 'Fabric',
    'upholstery': 'Fabric', 'velvet': 'Fabric', 'linen': 'Fabric',
    'mesh fabric': 'Fabric', 'padded': 'Fabric', 'foam': 'Fabric',
    'carpet': 'Fabric', 'felt': 'Fabric',
    # Glass
    'glass': 'Glass',
    # Books / paper
    'paper': 'Books', 'cardboard': 'Books', 'book': 'Books', 'books': 'Books',
    # Gypsum
    'gypsum': 'Gypsum', 'drywall': 'Gypsum', 'plaster': 'Gypsum',
    # Ceramic
    'ceramic': 'Ceramic', 'tile': 'Ceramic', 'porcelain': 'Ceramic',
    # Stone
    'stone': 'Stone', 'marble': 'Stone', 'granite': 'Stone',
}


def normalize_material_category(raw):
    """Map VLM/3DSSG raw material string to inject_internal_mass category."""
    if not raw:
        return 'Wood'  # safe default for furniture
    r = raw.strip().lower()
    # Direct match
    if r in MATERIAL_CATEGORY_MAP:
        return MATERIAL_CATEGORY_MAP[r]
    # Substring match
    for key, cat in MATERIAL_CATEGORY_MAP.items():
        if key in r:
            return cat
    return 'Wood'  # default


# ---------------------------------------------------------------------------
# Structural labels (skip in VLM + mapping output)
# ---------------------------------------------------------------------------

STRUCTURAL_LABELS = {
    'wall', 'floor', 'ceiling', 'door', 'window', 'curtain', 'blinds',
    'pipe', 'beam', 'column', 'railing', 'staircase',
}


# ===================================================================
# STEP 1: Extract 2D object images from RGB sequences
# ===================================================================

def step1_extract_images(scan_ids, rscan_dir, out_dir, frame_skip=10,
                         min_pixels=500):
    """Extract cropped 2D object images from RGB frames."""
    from scan2therm.extract_object_images import process_scan
    from tqdm import tqdm

    print("\n" + "=" * 60)
    print("STEP 1: Extract 2D object images")
    print("=" * 60)

    processed = 0
    skipped = 0

    for scan_id in tqdm(scan_ids, desc="Step 1"):
        scan_dir = osp.join(rscan_dir, scan_id)
        seq_dir = osp.join(scan_dir, 'sequence')

        # Skip if no sequence data
        if not osp.isdir(seq_dir):
            skipped += 1
            continue

        # Skip if images already extracted (check for any .jpg in out_dir)
        scan_out = osp.join(out_dir, scan_id)
        existing_imgs = list(Path(scan_out).glob('*.jpg')) if osp.isdir(scan_out) else []
        if existing_imgs:
            skipped += 1
            continue

        process_scan(scan_id, rscan_dir, out_dir, frame_skip, min_pixels)
        processed += 1

    print(f"Step 1 done: {processed} processed, {skipped} skipped (existing/no sequence)")


# ===================================================================
# STEP 2: Extract objects + CAD geometry
# ===================================================================

def step2_cad_geometry(scan_ids, data_dir, shapenet_dir, out_dir,
                       csv_path=None, objects_json_path=None, force=False):
    """Extract per-object geometry and augment with CAD surface area + volume."""
    from scan2therm.cad_geometry import process_scan, LABEL_TO_SYNSET

    print("\n" + "=" * 60)
    print("STEP 2: Extract objects + CAD geometry")
    print("=" * 60)

    # Report ShapeNet coverage
    available = set()
    if osp.isdir(shapenet_dir):
        for d in os.listdir(shapenet_dir):
            if osp.isdir(osp.join(shapenet_dir, d)):
                available.add(d)
    needed = set(LABEL_TO_SYNSET.values())
    missing = needed - available
    if missing:
        print(f"  ShapeNet: {len(available & needed)} categories available, "
              f"{len(missing)} missing (will use scanned geometry)")

    processed = 0
    skipped = 0

    for i, scan_id in enumerate(scan_ids):
        # Skip if objects_cad.json already exists
        objects_path = osp.join(out_dir, scan_id, 'objects_cad.json')
        if osp.isfile(objects_path) and not force:
            skipped += 1
            continue

        print(f"  [{i+1}/{len(scan_ids)}] {scan_id}")
        result = process_scan(
            scan_id=scan_id,
            data_dir=data_dir,
            shapenet_dir=shapenet_dir,
            out_dir=out_dir,
            csv_path=csv_path,
            objects_json_path=objects_json_path,
        )
        if result:
            processed += 1
        else:
            skipped += 1

    print(f"Step 2 done: {processed} processed, {skipped} skipped")


# ===================================================================
# STEP 3: VLM material classification (Vertex AI Gemini)
# ===================================================================

def step3_vlm_materials(scan_ids, data_dir, gcp_project, gcp_location='us-central1',
                        gemini_model='gemini-2.0-flash', delay=0.5):
    """Classify object materials using Vertex AI Gemini."""
    import vertexai
    from vertexai.generative_models import GenerativeModel

    from scan2therm.vlm_material_estimator_gemini import (
        process_scan, find_object_image, STRUCTURAL_LABELS as VLM_STRUCTURAL,
    )

    print("\n" + "=" * 60)
    print("STEP 3: VLM material classification (Gemini)")
    print("=" * 60)

    vertexai.init(project=gcp_project, location=gcp_location)
    model = GenerativeModel(gemini_model)

    total_updated = 0
    skipped_scans = 0

    for i, scan_id in enumerate(scan_ids):
        objects_path = osp.join(data_dir, scan_id, 'objects_cad.json')
        if not osp.isfile(objects_path):
            skipped_scans += 1
            continue

        # Check if VLM already ran (look for vlm_material in first non-structural object)
        with open(objects_path) as f:
            data = json.load(f)
        already_done = any(
            'vlm_material' in obj for obj in data['objects']
            if obj['label'].lower().strip() not in VLM_STRUCTURAL
        )
        if already_done:
            skipped_scans += 1
            continue

        print(f"  [{i+1}/{len(scan_ids)}] {scan_id}")
        result = process_scan(scan_id, data_dir, model)
        if result:
            total_updated += result['updated']
        if delay > 0:
            time.sleep(delay)

    print(f"Step 3 done: {total_updated} materials classified, "
          f"{skipped_scans} scans skipped")


# ===================================================================
# STEP 4: Build mapping JSON + inject into EnergyPlus IDF
# ===================================================================

def step4_build_and_inject(scan_ids, data_dir, zone_mapping=None,
                           zone_name='Core_ZN', zone_id='Z001',
                           idf_path=None, idd_path=None,
                           idf_output=None, mapping_output=None):
    """Build zone mapping JSON from objects_cad.json files, optionally inject into IDF.

    Args:
        zone_mapping: dict mapping zone_name -> list of scan_ids, e.g.
            {"Core_ZN": ["scan1", "scan2"], "Perimeter_ZN_1": ["scan3"]}
            If None, all scans go into a single zone (zone_name/zone_id).
    """

    print("\n" + "=" * 60)
    print("STEP 4: Build mapping JSON + inject into EnergyPlus IDF")
    print("=" * 60)

    # Build zone_mapping if not provided — all scans into one zone
    if zone_mapping is None:
        zone_mapping = {zone_name: scan_ids}

    ZONE_IDS = {}
    for i, zn in enumerate(zone_mapping.keys(), 1):
        ZONE_IDS[zn] = f'Z{i:03d}'

    zones = []
    total_objects = 0

    for zn, zn_scan_ids in zone_mapping.items():
        zid = ZONE_IDS[zn]
        zone_objects = []
        obj_counter = 0

        for scan_id in zn_scan_ids:
            if scan_id not in scan_ids:
                print(f"  [WARN] scan {scan_id} in zone {zn} not in scene list, skipping")
                continue

            objects_path = osp.join(data_dir, scan_id, 'objects_cad.json')
            if not osp.isfile(objects_path):
                continue

            with open(objects_path) as f:
                data = json.load(f)

            for obj in data['objects']:
                label = obj['label'].lower().strip()
                if label in STRUCTURAL_LABELS:
                    continue

                obj_counter += 1

                area = obj.get('surface_area_m2', 0)
                vol = obj.get('volume_m3', 0)
                if vol == 0 and area > 0:
                    obb = obj.get('obb_dimensions', [0, 0, 0])
                    vol = obb[0] * obb[1] * obb[2] if len(obb) == 3 else 0

                raw_material = obj.get('vlm_material',
                                       obj.get('material', 'wood'))
                category = normalize_material_category(raw_material)

                zone_objects.append({
                    'ID': f'{zid}_O{obj_counter:03d}',
                    'Type': obj['label'],
                    'Scan_ID': scan_id,
                    'Scan_Object_ID': obj['object_id'],
                    'Geometry': {
                        'Surface_Area_m2': round(area, 4),
                        'Volume_m3': round(vol, 4),
                    },
                    'Material_Info': {
                        'Category': category,
                        'Raw': raw_material,
                        'Source': obj.get('vlm_material_source',
                                         obj.get('material_source', 'fallback')),
                    },
                })

        zones.append({
            'Zone_ID': zid,
            'Zone_Name': zn,
            'Zone_Properties': {},
            'Objects': zone_objects,
        })
        total_objects += len(zone_objects)
        print(f"  Zone {zn} ({zid}): {len(zone_objects)} objects from {len(zn_scan_ids)} scans")

    total_scans = sum(len(sids) for sids in zone_mapping.values())
    mapping = {
        'Metadata': {
            'Project_Name': 'scan2therm',
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Description': f'{total_scans} 3RScan scenes → EnergyPlus InternalMass',
            'Num_Scans': total_scans,
            'Num_Objects': total_objects,
            'Num_Zones': len(zones),
        },
        'Zones': zones,
    }

    # Write mapping JSON
    if mapping_output is None:
        mapping_output = osp.join(data_dir, 'mapping.json')
    with open(mapping_output, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"  Mapping JSON: {len(all_objects)} objects → {mapping_output}")

    # Inject into EnergyPlus IDF if paths provided
    if idf_path and idd_path and idf_output:
        if not osp.isfile(idf_path):
            print(f"  [SKIP] IDF not found: {idf_path}")
            return mapping
        if not osp.isfile(idd_path):
            print(f"  [SKIP] IDD not found: {idd_path}")
            return mapping

        from scan2therm.inject_internal_mass import inject
        from eppy.modeleditor import IDF

        IDF.setiddname(idd_path)
        idf = IDF(idf_path)

        # Verify zone exists
        idf_zones = {z.Name for z in idf.idfobjects['ZONE']}
        if zone_name not in idf_zones:
            print(f"  WARNING: Zone '{zone_name}' not in IDF. "
                  f"Available: {idf_zones}")

        inject(idf, mapping)
        idf.saveas(idf_output)
        print(f"  IDF written: {idf_output}")

    return mapping


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='scan2therm: 3RScan → EnergyPlus InternalMass pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Extract 2D object images from RGB sequences
  2  Extract objects + CAD geometry → objects.json
  3  VLM material classification via Vertex AI Gemini
  4  Build mapping JSON + inject into EnergyPlus IDF

Examples:
  # Full pipeline:
  python scan2therm/main.py --scene_list scan2therm/office_scenes_105.txt ...

  # Steps 2-3 only (images already extracted):
  python scan2therm/main.py --steps 2 3 ...

  # Step 4 only (geometry + materials already done):
  python scan2therm/main.py --steps 4 --zone_name Core_ZN ...
""")
    # Scene selection
    parser.add_argument('--scene_list', type=str, required=True,
                        help='Text file with one scan_id per line')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Which steps to run (default: 1 2 3 4)')

    # Step 1: image extraction
    parser.add_argument('--rscan_dir', type=str,
                        default='3DSSG/data/3RScan/data/3RScan',
                        help='3RScan data root with sequence/ dirs')
    parser.add_argument('--frame_skip', type=int, default=10,
                        help='Process every Nth frame in step 1')

    # Step 2: CAD geometry
    parser.add_argument('--out_dir', type=str,
                        default='scan2therm/object_images',
                        help='Output dir (objects.json + images per scan)')
    parser.add_argument('--shapenet_dir', type=str,
                        default='Scan2CAD/Assets/shapenet-sample',
                        help='ShapeNet models directory')
    parser.add_argument('--csv_path', type=str,
                        default='3DSSG/files/3RScan.v2 Semantic Classes - Mapping.csv',
                        help='3RScan semantic classes CSV')
    parser.add_argument('--objects_json', type=str,
                        default='3DSSG/files/objects.json',
                        help='3DSSG objects.json')

    # Step 3: VLM materials
    parser.add_argument('--gcp_project', type=str,
                        default=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                        help='GCP project ID for Vertex AI')
    parser.add_argument('--gcp_location', type=str, default='us-central1',
                        help='Vertex AI region')
    parser.add_argument('--gemini_model', type=str, default='gemini-2.0-flash',
                        help='Gemini model name')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Seconds between Gemini API calls')

    # Step 4: IDF injection
    parser.add_argument('--zone_mapping', type=str, default=None,
                        help='JSON file mapping zone names to scan IDs '
                             '(e.g. {"Core_ZN": ["scan1"], "Perimeter_ZN_1": ["scan2"]})')
    parser.add_argument('--zone_name', type=str, default='Core_ZN',
                        help='Fallback zone name if --zone_mapping not provided')
    parser.add_argument('--zone_id', type=str, default='Z001',
                        help='Fallback zone ID if --zone_mapping not provided')
    parser.add_argument('--idf', type=str, default=None,
                        help='Input EnergyPlus IDF file')
    parser.add_argument('--idd', type=str, default=None,
                        help='EnergyPlus IDD file')
    parser.add_argument('--idf_output', type=str, default=None,
                        help='Output IDF file path')
    parser.add_argument('--mapping_output', type=str, default=None,
                        help='Output mapping JSON path (default: <out_dir>/mapping.json)')

    # General
    parser.add_argument('--force', action='store_true',
                        help='Re-run steps even if output exists')

    args = parser.parse_args()
    steps = set(args.steps)

    # Load scene list
    with open(args.scene_list) as f:
        scan_ids = [line.strip() for line in f if line.strip()]
    print(f"scan2therm pipeline — {len(scan_ids)} scenes, steps: {sorted(steps)}")

    # Resolve paths relative to CWD
    def resolve(p):
        return p if (p is None or osp.isabs(p)) else osp.join(os.getcwd(), p)

    rscan_dir = resolve(args.rscan_dir)
    out_dir = resolve(args.out_dir)
    shapenet_dir = resolve(args.shapenet_dir)
    csv_path = resolve(args.csv_path)
    objects_json = resolve(args.objects_json)

    # ── Step 1 ──
    if 1 in steps:
        if not osp.isdir(rscan_dir):
            print(f"\n[SKIP Step 1] 3RScan dir not found: {rscan_dir}")
        else:
            step1_extract_images(scan_ids, rscan_dir, out_dir,
                                 frame_skip=args.frame_skip)

    # ── Step 2 ──
    if 2 in steps:
        step2_cad_geometry(
            scan_ids, data_dir=out_dir, shapenet_dir=shapenet_dir,
            out_dir=out_dir,
            csv_path=csv_path if osp.isfile(csv_path) else None,
            objects_json_path=objects_json if osp.isfile(objects_json) else None,
            force=args.force,
        )

    # ── Step 3 ──
    if 3 in steps:
        if not args.gcp_project:
            print("\n[SKIP Step 3] No --gcp_project provided")
        else:
            step3_vlm_materials(
                scan_ids, data_dir=out_dir,
                gcp_project=args.gcp_project,
                gcp_location=args.gcp_location,
                gemini_model=args.gemini_model,
                delay=args.delay,
            )

    # ── Step 4 ──
    if 4 in steps:
        zone_mapping = None
        if args.zone_mapping:
            zm_path = resolve(args.zone_mapping)
            with open(zm_path) as f:
                zone_mapping = json.load(f)
            print(f"Loaded zone mapping: {len(zone_mapping)} zones from {zm_path}")

        step4_build_and_inject(
            scan_ids, data_dir=out_dir,
            zone_mapping=zone_mapping,
            zone_name=args.zone_name, zone_id=args.zone_id,
            idf_path=resolve(args.idf),
            idd_path=resolve(args.idd),
            idf_output=resolve(args.idf_output),
            mapping_output=resolve(args.mapping_output),
        )

    print("\n✓ Pipeline complete.")


if __name__ == '__main__':
    main()
