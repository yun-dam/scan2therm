import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import openai

from vlm_material_estimator import encode_image, identify_material

OBJECT_IMAGES = Path("object_images")
JSONS_IN = Path("jsons")
JSONS_OUT = Path("jsons_vlm_estimate")


def build_jpg_to_object_id(objects_json_path: Path) -> dict[str, tuple[str, int]]:
    """Return {jpg_stem: (label, object_id)} using label+counter ordering by object_id."""
    data = json.loads(objects_json_path.read_text())
    by_label = defaultdict(list)
    for obj in sorted(data["objects"], key=lambda x: x["object_id"]):
        by_label[obj["label"]].append(obj["object_id"])
    mapping = {}
    for label, oids in by_label.items():
        for i, oid in enumerate(oids, start=1):
            stem = label.replace(" ", "_") + f"{i:02d}"
            mapping[stem] = (label, oid)
    return mapping


def process_scan(scan_id: str, client: openai.OpenAI, model: str) -> None:
    obj_img_dir = OBJECT_IMAGES / scan_id
    json_out = JSONS_OUT / f"{scan_id}.json"

    if not json_out.exists():
        print(f"  [skip] no json for {scan_id}")
        return

    objects_json = obj_img_dir / "objects.json"
    if not objects_json.exists():
        print(f"  [skip] no objects.json for {scan_id}")
        return

    stem_map = build_jpg_to_object_id(objects_json)

    # Run VLM on each jpg
    vlm_results: dict[int, str] = {}
    for jpg in sorted(obj_img_dir.glob("*.jpg")):
        stem = jpg.stem
        if stem not in stem_map:
            print(f"  [warn] no mapping for {jpg.name}")
            continue
        label, oid = stem_map[stem]
        image_b64 = encode_image(jpg)
        material = identify_material(client, label, image_b64, model)
        vlm_results[oid] = material
        print(f"    {jpg.name}: {material}")

    # Patch the output JSON
    scan_json = json.loads(json_out.read_text())
    for zone in scan_json.get("Zones", []):
        for obj in zone.get("Objects", []):
            oid = obj.get("Scan_Object_ID")
            mi = obj["Material_Info"]
            if oid in vlm_results:
                mi["Forecast"] = vlm_results[oid]
                mi["Source"] = "vlm_estimate"
            else:
                mi["Forecast"] = None
    json_out.write_text(json.dumps(scan_json, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VLM material estimation on object_images/ and write to jsons_vlm_estimate/"
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument(
        "--scan", default=None, help="Process only this scan_id (for testing)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY")

    # Copy jsons/ → jsons_vlm_estimate/ (fresh copy each run)
    if JSONS_OUT.exists():
        shutil.rmtree(JSONS_OUT)
    shutil.copytree(JSONS_IN, JSONS_OUT)
    print(f"Copied {JSONS_IN} → {JSONS_OUT}")

    client = openai.OpenAI(api_key=api_key)

    if args.scan:
        scan_dirs = [OBJECT_IMAGES / args.scan]
    else:
        scan_dirs = sorted(p for p in OBJECT_IMAGES.iterdir() if p.is_dir())

    for scan_dir in scan_dirs:
        if not scan_dir.is_dir():
            print(f"[skip] {scan_dir} is not a directory")
            continue
        print(f"Processing {scan_dir.name} ...")
        process_scan(scan_dir.name, client, args.model)

    print("Done.")


if __name__ == "__main__":
    main()
