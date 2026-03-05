#!/usr/bin/env python3
"""
VLM material estimator using Vertex AI Gemini.

For each non-structural object in a scan, sends its cropped 2D image to
Gemini to classify the primary material. Updates objects.json in-place.

Uses the same constrained-material logic as run_vlm_batch.py:
  - Maps jpg filenames to object IDs via build_jpg_to_object_id()
  - Constrains VLM output to VALID_MATERIALS
  - Parses response with parse_material()

Usage:
    python scan2therm/vlm_material_estimator_gemini.py \
        --scene_list scan2therm/office_scenes_105.txt \
        --data_dir scan2therm/object_images \
        --project YOUR_GCP_PROJECT_ID \
        --location us-central1

Requires:
    pip install google-cloud-aiplatform
    gcloud auth application-default login
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path


STRUCTURAL_LABELS = {
    'wall', 'floor', 'ceiling', 'door', 'window', 'curtain', 'blinds',
    'pipe', 'beam', 'column', 'railing', 'staircase',
}

VALID_MATERIALS = ["Wood", "Metal", "Plastic", "Fabric", "Books", "Gypsum", "Concrete"]

MATERIAL_PROMPT = (
    "This is a mesh from a 3D room scan. "
    "Identify the primary material of the '{label}' object visible in the texture. "
    "Reply with with one of the following materials: Wood, Metal, Plastic, Fabric, Books, Gypsum, Concrete. "
    "Answer with exactly one of these materials."
    "If the object is not made of one of these materials, reply with the material "
    "that is most similar to the object in terms of thermodynamic properties."
)


def parse_material(text: str) -> str | None:
    """Extract the first valid material keyword from a free-form response."""
    for mat in VALID_MATERIALS:
        if mat.lower() in text.lower():
            return mat
    return None


def build_jpg_to_object_id(objects_path: Path) -> dict[str, tuple[str, int]]:
    """Return {jpg_stem: (label, object_id)} using label+counter ordering by object_id."""
    data = json.loads(objects_path.read_text())
    by_label = defaultdict(list)
    for obj in sorted(data["objects"], key=lambda x: x["object_id"]):
        by_label[obj["label"]].append(obj["object_id"])
    mapping = {}
    for label, oids in by_label.items():
        for i, oid in enumerate(oids, start=1):
            stem = label.replace(" ", "_") + f"{i:02d}"
            mapping[stem] = (label, oid)
    return mapping


def identify_material(model, label: str, image_path: str) -> str | None:
    """Ask Gemini to identify the primary material from a cropped object image."""
    from vertexai.generative_models import Image
    image = Image.load_from_file(image_path)
    prompt = MATERIAL_PROMPT.format(label=label)

    response = model.generate_content(
        [image, prompt],
        generation_config={"max_output_tokens": 30, "temperature": 0.1},
    )
    return parse_material(response.text.strip())


def process_scan(scan_id, data_dir, model, dry_run=False):
    """Classify materials for all objects in a scan via Gemini.

    Uses build_jpg_to_object_id() to map each jpg to its object,
    then calls Gemini and parses the result into a VALID_MATERIALS category.
    Updates objects.json in-place with vlm_material and vlm_material_source.
    """
    scan_dir = Path(data_dir) / scan_id
    objects_path = scan_dir / 'objects.json'

    if not objects_path.exists():
        print(f"  [SKIP] No objects.json for {scan_id}")
        return None

    stem_map = build_jpg_to_object_id(objects_path)

    # Run VLM on each jpg
    vlm_results: dict[int, str] = {}
    for jpg in sorted(scan_dir.glob("*.jpg")):
        stem = jpg.stem
        if stem not in stem_map:
            continue
        label, oid = stem_map[stem]

        if label.lower().strip() in STRUCTURAL_LABELS:
            continue

        if dry_run:
            print(f"    {jpg.name}: would query (label={label}, oid={oid})")
            continue

        try:
            material = identify_material(model, label, str(jpg))
            if material:
                vlm_results[oid] = material
                print(f"    {jpg.name}: {material}")
            else:
                print(f"    {jpg.name}: no valid material parsed")
        except Exception as e:
            print(f"    [WARN] Gemini failed for {jpg.name}: {e}")
            if 'quota' in str(e).lower() or '429' in str(e):
                print("    Backing off 30s for rate limit...")
                time.sleep(30)

    if dry_run:
        return {'updated': 0, 'skipped': 0}

    # Patch objects.json with VLM results
    with open(objects_path) as f:
        data = json.load(f)

    updated = 0
    for obj in data['objects']:
        oid = obj['object_id']
        if oid in vlm_results:
            obj['vlm_material'] = vlm_results[oid]
            obj['vlm_material_source'] = 'vlm_estimate'
            updated += 1

    with open(objects_path, 'w') as f:
        json.dump(data, f, indent=2)

    skipped = len(stem_map) - updated
    print(f"  {scan_id}: {updated} classified, {skipped} skipped")
    return {'updated': updated, 'skipped': skipped}


def find_object_image(scan_dir: Path, label: str, object_id: int):
    """Find the cropped image for an object.

    Images follow the naming convention: {label}{index}.jpg
    e.g. chair01.jpg, desk_chair02.jpg
    """
    label_slug = label.replace(' ', '_')
    for ext in ('jpg', 'png'):
        candidates = sorted(scan_dir.glob(f"{label_slug}*.{ext}"))
        if candidates:
            return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Classify object materials using Vertex AI Gemini')
    parser.add_argument('--scene_list', type=str, required=True,
                        help='Text file with one scan_id per line')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing scan subdirs with objects.json + images')
    parser.add_argument('--project', type=str,
                        default=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                        help='GCP project ID (or set GOOGLE_CLOUD_PROJECT)')
    parser.add_argument('--location', type=str, default='us-central1',
                        help='Vertex AI region')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                        help='Gemini model name')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be queried without calling Gemini')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Seconds between API calls (rate limiting)')
    args = parser.parse_args()

    if not args.project and not args.dry_run:
        raise ValueError("Provide --project or set GOOGLE_CLOUD_PROJECT")

    # Load scene list
    with open(args.scene_list) as f:
        scan_ids = [line.strip() for line in f if line.strip()]
    print(f"Processing {len(scan_ids)} scans with {args.model}")

    if not args.dry_run:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=args.project, location=args.location)
        model = GenerativeModel(args.model)
    else:
        model = None

    total_updated = 0
    total_skipped = 0

    for i, scan_id in enumerate(scan_ids):
        print(f"[{i+1}/{len(scan_ids)}] {scan_id}")
        result = process_scan(scan_id, args.data_dir, model, dry_run=args.dry_run)
        if result:
            total_updated += result['updated']
            total_skipped += result['skipped']
        if not args.dry_run and args.delay > 0:
            time.sleep(args.delay)

    print(f"\nDone. {total_updated} materials classified, {total_skipped} skipped.")


if __name__ == '__main__':
    main()
