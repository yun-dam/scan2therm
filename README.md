# 🌡️ scan2therm

**From 3D scans to thermal simulations** — an automated pipeline that converts indoor 3D scene scans into EnergyPlus thermal models.

## 🎯 What is this?

Buildings account for ~40% of global energy consumption, yet thermal simulations still rely on manually authored models. **scan2therm** bridges this gap by automatically extracting furniture and objects from real-world 3D scans, estimating their geometry and materials, and injecting them as thermal mass into EnergyPlus building energy models.

Given a set of [3RScan](https://waldjohannau.github.io/RIO/) indoor scene scans, the pipeline:

1. 📸 **Extracts object images** from RGB-D sequences
2. 📐 **Estimates geometry** (surface area & volume) using CAD model matching
3. 🧠 **Classifies materials** (wood, metal, fabric, etc.) via Vision-Language Models (Gemini)
4. 🏗️ **Injects InternalMass objects** into EnergyPlus IDF files for thermal simulation

## 🔧 Pipeline

The main entry point is **`main.py`** — a multi-scan pipeline with VLM-based material classification and optional [CrossOver](https://github.com/GradientSpaces/CrossOver) retrieval.

By default, CAD geometry is looked up via label → ShapeNet category mapping. With the `--use_crossover` flag, the pipeline uses learned [CrossOver](https://github.com/GradientSpaces/CrossOver) embeddings to retrieve the most geometrically similar CAD model for each scanned object — significantly improving geometry estimates.

## 📁 Project Structure

```
scan2therm/
├── main.py                        # Main pipeline (VLM + optional CrossOver)
├── baseline.py                    # v1 baseline (single-scan, 3DSSG materials)
├── extract_object_images.py       # Step 1: crop 2D object images from RGB
├── cad_geometry.py                # Step 2: label-based CAD geometry lookup
├── crossover_cad_geometry.py      # Step 2: CrossOver embedding-based retrieval
├── vlm_material_estimator_gemini.py  # Step 3: Gemini VLM material classifier
├── inject_internal_mass.py        # Step 4: EnergyPlus IDF injection
├── extract_objects.py             # Object extraction from 3RScan meshes
├── point_cloud_utils.py           # Point cloud helpers
├── scan3r_utils.py                # 3RScan data loading utilities
├── requirements.txt
├── energyplus/                    # Base IDF/IDD files
├── crossover/                     # CrossOver framework (bundled)
│   ├── model/                     # CrossOver model definitions
│   ├── modules/                   # Network modules (PointNet, BLIP, etc.)
│   ├── common/                    # Shared constants & utilities
│   ├── util/                      # Point cloud & torch utilities
│   └── third_party/               # BLIP vision encoder
└── office_scenes_105.txt          # Default scene list (105 office scenes)
```

## 🚀 Setup

### 1. Create conda environment

```bash
conda create -n scan2therm python=3.10 -y
conda activate scan2therm
```

### 2. Install PyTorch (with CUDA)

```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Data preparation

You will need:
- **[3RScan](https://waldjohannau.github.io/RIO/)** — RGB-D sequences with instance-annotated meshes
- **[ShapeNet](https://shapenet.org/)** — CAD models for geometry augmentation
- **EnergyPlus IDF/IDD** — base building model (included in `energyplus/`)
- *(Optional)* **[CrossOver checkpoints](https://github.com/GradientSpaces/CrossOver)** — for embedding-based CAD retrieval
- *(Optional)* **Google Cloud project** — for Vertex AI Gemini VLM access (Step 3)

## ▶️ Usage

### Full pipeline (recommended)

```bash
python main.py \
    --scene_list office_scenes_105.txt \
    --rscan_dir /path/to/3RScan \
    --shapenet_dir /path/to/ShapeNet \
    --gcp_project your-gcp-project
```

### With CrossOver retrieval

```bash
python main.py \
    --scene_list office_scenes_105.txt \
    --rscan_dir /path/to/3RScan \
    --shapenet_dir /path/to/ShapeNet \
    --use_crossover \
    --ckpt /path/to/instance_crossover.pth \
    --i2pmae_ckpt /path/to/pointbind_i2pmae.pt
```

### Run specific steps only

```bash
# Only geometry extraction (Step 2):
python main.py --steps 2 --rscan_dir ... --shapenet_dir ...

# Only VLM material classification (Step 3):
python main.py --steps 3 --gcp_project your-gcp-project
```

## 📊 Pipeline Steps Overview

```
3RScan RGB-D scenes
        │
        ▼
   ┌─────────┐
   │ Step 1   │  Extract 2D object crops from RGB frames
   └────┬────┘
        ▼
   ┌─────────┐    label-based ──► ShapeNet category lookup
   │ Step 2   │──
   └────┬────┘    --use_crossover ──► CrossOver embedding retrieval
        ▼
   ┌─────────┐
   │ Step 3   │  Gemini VLM classifies materials (wood, metal, fabric...)
   └────┬────┘
        ▼
   ┌─────────┐
   │ Step 4   │  Inject InternalMass into EnergyPlus IDF
   └────┬────┘
        ▼
  EnergyPlus IDF with thermal mass
```

## 🙏 Acknowledgements

- [CrossOver](https://github.com/GradientSpaces/CrossOver) — 3D scene understanding via cross-modal retrieval
- [3RScan](https://waldjohannau.github.io/RIO/) — real-world RGB-D indoor scan dataset
- [EnergyPlus](https://energyplus.net/) — building energy simulation engine
