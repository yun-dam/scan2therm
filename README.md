# scan2therm

Converts 3RScan indoor scene data into EnergyPlus thermal models by extracting object geometry, classifying materials with a VLM, and injecting InternalMass objects into IDF files.

## Project Structure

```
scan2therm/
  main.py              # v1 pipeline (single-scan, 3DSSG materials)
  main_v3.py           # v3 pipeline (multi-scan, VLM + optional CrossOver)
  cad_geometry.py      # Label-based CAD geometry lookup
  crossover_cad_geometry.py  # CrossOver embedding-based CAD retrieval
  extract_object_images.py   # Step 1: crop 2D object images from RGB
  extract_objects.py         # Object extraction from 3RScan meshes
  vlm_material_estimator_gemini.py  # Step 3: Gemini VLM material classifier
  inject_internal_mass.py    # Step 4: EnergyPlus IDF injection
  point_cloud_utils.py       # Point cloud helpers
  scan3r_utils.py            # 3RScan data loading utilities
  crossover/                 # CrossOver framework (CAD retrieval model)
```

## Setup

```bash
pip install -r requirements.txt
```

For CrossOver mode, also install PyTorch with CUDA support:
```bash
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Usage

### main_v3.py (recommended)

Four-stage pipeline: (1) extract images, (2) CAD geometry, (3) VLM materials, (4) IDF injection.

```bash
# Full pipeline (label-based CAD):
python scan2therm/main_v3.py \
    --scene_list scan2therm/office_scenes_105.txt \
    --rscan_dir <3RScan_data_path> \
    --shapenet_dir <ShapeNet_path> \
    --gcp_project <your_gcp_project>

# With CrossOver retrieval:
python scan2therm/main_v3.py \
    --scene_list scan2therm/office_scenes_105.txt \
    --rscan_dir <3RScan_data_path> \
    --shapenet_dir <ShapeNet_path> \
    --use_crossover \
    --ckpt checkpoints/instance_crossover_scannet+scan3r+multiscan+arkitscenes.pth \
    --i2pmae_ckpt checkpoints/pointbind_i2pmae.pt

# Run specific steps only:
python scan2therm/main_v3.py --steps 2 3 ...
```

### main.py (v1 baseline)

Single-scan pipeline using 3DSSG ground-truth materials.

```bash
python scan2therm/main.py \
    --rscan_root <3RScan_path> \
    --dssg_root <3DSSG_path> \
    --idf scan2therm/energyplus/SmallOffice.idf \
    --idd scan2therm/energyplus/Energy+.idd
```

## Data Requirements

- **3RScan**: RGB-D sequences with instance-annotated meshes
- **ShapeNet**: CAD models for geometry augmentation
- **CrossOver checkpoints** (optional): for embedding-based CAD retrieval
- **EnergyPlus IDF/IDD**: base building model for thermal injection
