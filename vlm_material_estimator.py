import argparse
import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

import openai


def encode_image(image_path: Path) -> str:
    """Base64-encode an image for the vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


VALID_MATERIALS = ["Wood", "Metal", "Plastic", "Fabric", "Books", "Gypsum", "Concrete"]


def parse_material(text: str) -> str | None:
    """Extract the first valid material keyword from a free-form response."""
    for mat in VALID_MATERIALS:
        if mat.lower() in text.lower():
            return mat
    return None


def identify_material(
    client: openai.OpenAI, label: str, image_b64: str, model: str = "gpt-4o-mini"
) -> str | None:
    """Ask the model to identify the primary material; retries up to 3 times on error."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"This is a mesh from a 3D room scan. "
                        f"Identify the primary material of the '{label}' object visible in the texture. "
                        f"Reply with with one of the following materials: Wood, Metal, Plastic, Fabric, Books, Gypsum, Concrete. Answer with exactly one of these materials."
                        f"If the object is not made of one of these materials, reply with the material that is most similar to the object in terms of thermodynamic properties."
                    ),
                },
            ],
        }
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=10
            )
            raw = resp.choices[0].message.content.strip()
            return parse_material(raw)
        except Exception as e:
            print(f"  [warn] API error (attempt {attempt + 1}/3): {e}")
    return None


def estimate_materials(
    scan_dir: Path,
    client: openai.OpenAI,
    skip: set[str] | None = None,
    model: str = "gpt-4o-mini",
) -> dict[int, dict]:
    """Return {objectId: {label, material}} for non-structural objects in a scan."""
    if skip is None:
        skip = {"wall", "floor", "ceiling"}

    seg_groups = json.load(open(scan_dir / "semseg.v2.json")).get("segGroups", [])
    image_b64 = encode_image(scan_dir / "mesh.refined_0.png")

    results = {}
    for obj in seg_groups:
        oid = int(obj.get("objectId", obj.get("id", -1)))
        label = obj.get("label", "?")
        if label.lower() in skip:
            continue
        results[oid] = {
            "label": label,
            "material": identify_material(client, label, image_b64, model),
        }

    return results


# ── Demo: run on the first scan that has the required files ──────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Provide --api-key or set OPENAI_API_KEY")

    DATA = Path(__file__).resolve().parent / "3RScan_Data"
    scan_dir = next(
        (
            d
            for d in DATA.iterdir()
            if d.is_dir()
            and (d / "semseg.v2.json").exists()
            and (d / "mesh.refined_0.png").exists()
        ),
        None,
    )
    if not scan_dir:
        raise FileNotFoundError(
            "No 3RScan_Data/*/semseg.v2.json + mesh.refined_0.png found"
        )

    img_path = scan_dir / "mesh.refined_0.png"
    if shutil.which("imgcat"):
        subprocess.run(["imgcat", str(img_path)])
    else:
        subprocess.run(["open", str(img_path)])
        print(f"(opened {img_path.name} in Preview)")

    client = openai.OpenAI(api_key=args.api_key)
    results = estimate_materials(scan_dir, client)

    print(f"Scan: {scan_dir.name}")
    for oid, info in results.items():
        print(f"  [{oid:3d}] {info['label']:20s} → {info['material']}")
