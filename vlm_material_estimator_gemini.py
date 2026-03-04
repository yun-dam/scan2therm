#!/usr/bin/env python3
"""
VLM material estimator using Vertex AI Gemini.

For each non-structural object in a scan, sends its cropped 2D image to
Gemini to classify the primary material. Updates objects.json in-place.

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
import base64
import json
import os
import os.path as osp
import time
from pathlib import Path


STRUCTURAL_LABELS = {
    'wall', 'floor', 'ceiling', 'door', 'window', 'curtain', 'blinds',
    'pipe', 'beam', 'column', 'railing', 'staircase',
}

MATERIAL_PROMPT = (
    "This is a cropped image of a '{label}' from a 3D room scan. "
    "Identify the primary material of this object. "
    "Reply with one word or short lowercase phrase only with no punctuation "
    "(e.g. wood, metal, fabric, plastic, leather, glass, ceramic, stone, concrete, carpet, cardboard, padded)."
)


def identify_material(model, label: str, image_path: str) -> str:
    """Ask Gemini to identify the primary material from a cropped object image."""
    from vertexai.generative_models import Image
    image = Image.load_from_file(image_path)
    prompt = MATERIAL_PROMPT.format(label=label)

    response = model.generate_content(
        [image, prompt],
        generation_config={"max_output_tokens": 30, "temperature": 0.1},
    )
    return response.text.strip().lower().rstrip('.')


def process_scan(scan_id, data_dir, model, dry_run=False):
    """Classify materials for all objects in a scan via Gemini.

    Reads objects.json (from cad_geometry.py), sends each object's cropped
    image to Gemini, and updates the material + material_source fields.
    """
    scan_dir = Path(data_dir) / scan_id
    objects_path = scan_dir / 'objects.json'

    if not objects_path.exists():
        print(f"  [SKIP] No objects.json for {scan_id}")
        return None

    with open(objects_path) as f:
        data = json.load(f)

    updated = 0
    skipped = 0

    # Pre-build image assignment: map each label to its sorted image list,
    # then assign images round-robin to objects with that label.
    label_images = {}  # label -> list of Path
    label_counters = {}  # label -> next index to assign

    for obj in data['objects']:
        label = obj['label'].lower().strip()
        if label in STRUCTURAL_LABELS or label in label_images:
            continue
        label_slug = label.replace(' ', '_')
        imgs = sorted(scan_dir.glob(f"{label_slug}*.jpg")) + \
               sorted(scan_dir.glob(f"{label_slug}*.png"))
        label_images[label] = imgs
        label_counters[label] = 0

    for obj in data['objects']:
        label = obj['label'].lower().strip()

        # Skip structural elements
        if label in STRUCTURAL_LABELS:
            skipped += 1
            continue

        # Assign next available image for this label
        imgs = label_images.get(label, [])
        if not imgs:
            skipped += 1
            continue
        idx = label_counters[label]
        image_path = imgs[idx % len(imgs)]
        label_counters[label] = idx + 1

        if dry_run:
            print(f"    obj {obj['object_id']:3d} | {label:20s} | would query: {image_path.name}")
            continue

        try:
            material = identify_material(model, label, str(image_path))
            obj['vlm_material'] = material
            obj['vlm_material_source'] = 'gemini'
            updated += 1
        except Exception as e:
            print(f"    [WARN] Gemini failed for obj {obj['object_id']} ({label}): {e}")
            skipped += 1
            # Rate limit backoff
            if 'quota' in str(e).lower() or '429' in str(e):
                print("    Backing off 30s for rate limit...")
                time.sleep(30)

    if not dry_run:
        with open(objects_path, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"  {scan_id}: {updated} classified, {skipped} skipped")
    return {'updated': updated, 'skipped': skipped}


def find_object_image(scan_dir: Path, label: str, object_id: int):
    """Find the cropped image for an object.

    Images follow the naming convention: {label}{index}.jpg
    e.g. chair01.jpg, desk_chair02.jpg
    """
    # Normalize label for filename matching (spaces → underscores)
    label_slug = label.replace(' ', '_')

    # Try exact match patterns
    for ext in ('jpg', 'png'):
        # Try with various index patterns
        candidates = sorted(scan_dir.glob(f"{label_slug}*.{ext}"))
        if candidates:
            # If multiple images for this label, just use the first
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
