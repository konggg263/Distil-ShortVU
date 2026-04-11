# Kaggle experiment assets

- **Notebook:** `source/kd-experiment-500-videos.ipynb` (under this folder; run on Kaggle with GPU + dataset input).
- **Extracted features (500 videos):** `results/500_videos/features_500.json`

Train locally with:

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_local
```

Paths are relative to the **repository root** (`Distil-ShortVU/`).
