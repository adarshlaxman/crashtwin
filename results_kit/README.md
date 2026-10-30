# Results kit (Git LFS)

`crashtwin_results_20261030.zip` (318 MB) is everything computed on the lab PC up to 30 Oct 2026, except the
raw CARLA clips and the datasets: feature caches, tracks, the 26 trained runs (checkpoints, predictions,
metrics), results tables and figures, indexes, splits, logs, CARLA run metadata and the YOLO11s weights.

Unzip it inside the project folder (the folder that contains `crashtwin/`):

```
tar -xf crashtwin\results_kit\crashtwin_results_20261030.zip -C C:\CARLA_Project      # Windows
unzip crashtwin/results_kit/crashtwin_results_20261030.zip -d ~/CARLA_Project          # Linux
```

Then `pipeline.py status` shows 26/26 experiments and work continues from there.
Git for Windows includes Git LFS; on Linux run `sudo apt install git-lfs && git lfs install` before cloning.

**Keep this repository private.** The kit contains features and predictions derived from the Nexar dataset
(licence under supervisor review) and the AGPL-3.0 YOLO11s weights.
