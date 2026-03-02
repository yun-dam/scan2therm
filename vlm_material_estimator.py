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


def identify_material(
    client: openai.OpenAI, label: str, image_b64: str, model: str = "gpt-4o-mini"
) -> str:
    """Ask o4-mini to identify the primary material of a labeled object from a texture atlas."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This is a mesh from a 3D room scan. "
                            f"Identify the primary material of the '{label}' object visible in the texture. "
                            f"Reply with with one of the following materials: Wood, Metal, Plastic, Fabric, Books, Gypsum, Concrete"
                            f"If the object is not made of one of these materials, reply with the material that is most similar to the object."
                        ),
                    },
                ],
            }
        ],
        max_tokens=30,
    )
    return resp.choices[0].message.content.strip()


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
