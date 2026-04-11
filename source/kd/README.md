# Knowledge Distillation (KD) — Distil-ShortVU

This folder contains the KD pipeline: **Teacher** (full multimodal + quality) → **soft targets** → **Student baseline** vs **Student + KD** → optional **ablation** and **explainability**.

## Quick start

- **Input:** a features JSON where every sample has `ecr`, `visual_emb`, `text_emb`, and `quality_scores`. Example: `source/kaggle_kd/results/500_videos/features_500.json` (see `source/kaggle_kd/README.md` for where Kaggle outputs live).
- **Smoke test:** add `--quick` (few epochs; checks that the code runs).
- **Full runs / reports:** omit `--quick`, increase `--teacher-epochs` and `--student-epochs`, and enable `--explain` plus `--ablation` for a complete experimental bundle.
- **If Student+KD underperforms the baseline:** try **lowering `--beta`**, **slightly lowering `--alpha`**, and raising student epochs; see **Hyperparameter tuning** below.
- **Accelerators:** use `--device cuda` or `--device mps` (Apple Silicon).

## Data format (JSON)

Each sample should have:

- `ecr` (float, label)
- `visual_emb` (1024-d, e.g. ImageBind)
- `text_emb` (384-d, e.g. MiniLM)
- `quality_scores`: `{ "aesthetic": float, "technical": float }` on scale ~0–10 (Teacher + auxiliary Student losses)

Optional for explainability / demo: `video_id`, `title`, `caption`.

Example path in repo: `source/kaggle_kd/results/500_videos/features_500.json`.

## Setup

From project root:

```bash
pip install -r requirements.txt
```

You need **PyTorch** and **scipy**. Set device with `--device`: `cpu`, `cuda`, or `mps`.

## 1. Main experiment (`run_experiment.py`)

### Quick run (smoke test)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd \
  --quick
```

`--quick` uses 15 teacher epochs and 20 student epochs. Use only to verify the code, **not** as final thesis numbers.

### Full run (recommended for reports)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd \
  --device cpu \
  --teacher-epochs 100 \
  --student-epochs 120 \
  --batch 32 \
  --alpha 0.3 \
  --beta 0.1 \
  --gamma 0.2 \
  --delta 0.2 \
  --explain --explain-n 20 \
  --ablation
```

- `--explain`: modality ablation on Student, internal aesthetic/technical scores, LLM-ready prompt → `explanations.json`.
- `--ablation`: train visual-only, text-only, concat-fusion variants; measure inference ms → `ablation_report.json`. Ablation training uses **half** of `--student-epochs`.

GPU / MPS examples:

```bash
python3 source/kd/run_experiment.py --data PATH/features.json --device cuda --save-dir results_kd
python3 source/kd/run_experiment.py --data PATH/features.json --device mps --save-dir results_kd
```

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to features JSON |
| `--save-dir` | `results_kd` | Checkpoints + reports |
| `--device` | `cpu` | `cpu` / `cuda` / `mps` |
| `--max-samples` | all | Subsample for debugging |
| `--teacher-hidden` | 512 | Teacher hidden size |
| `--teacher-blocks` | 4 | Teacher residual blocks |
| `--teacher-epochs` | 60 | Teacher epochs |
| `--teacher-lr` | 3e-4 | Teacher LR |
| `--student-hidden` | 256 | Student hidden size |
| `--student-epochs` | 80 | Student epochs (baseline + KD) |
| `--student-lr` | 5e-4 | Student LR |
| `--dropout` | 0.1 | Dropout |
| `--batch` | 32 | Batch size |
| `--alpha` | 0.5 | Weight for soft ECR (match teacher ECR) |
| `--beta` | 0.3 | Weight for representation (cosine) KD |
| `--gamma` | 0.2 | Weight for aesthetic auxiliary |
| `--delta` | 0.2 | Weight for technical auxiliary |
| `--quick` | off | Few epochs for smoke test |
| `--ablation` | off | Run ablation + inference timing |
| `--explain` | off | Run explainability export |
| `--explain-n` | 20 | Number of val samples to explain |

### Outputs (under `--save-dir`)

- `teacher_best.pth`, `student_baseline_best.pth`, `student_kd_best.pth`
- `experiment_report.json` — PLCC, SRCC, KTAU, MSE, MAE, param counts
- `ablation_report.json` if `--ablation`
- `explanations.json` if `--explain`
- `ablation_*.pth` for ablation variants (if saved)

## 2. Hyperparameter tuning (KD)

Student loss (summary):

`L = L_ecr_hard + alpha*L_ecr_soft + beta*L_repr + gamma*L_aes + delta*L_tech`

**When ECR labels are small and low-variance** (e.g. roughly 0–0.12 on your 500-video slice):

1. **Lower `beta`** (try 0.05–0.15): `L_repr` can dominate early; too large `beta` may hurt ECR-focused metrics.
2. **Slightly lower `alpha`** (0.2–0.4): balance teacher soft targets vs ground-truth ECR.
3. **Increase `--student-epochs`** (100–150) before concluding KD does not help.
4. **Strong teacher**: more `--teacher-epochs` or regularization; pick checkpoint by val MSE.
5. **Learning rates**: try `--student-lr` in `3e-4` … `1e-3`; teacher often stable at `1e-4` … `3e-4`.
6. **Batch size**: for ~500 samples, 16–32 is typical; very large batches can add noise to small-batch gradients.

Practical order: **enough epochs** → tune **`beta`** → **`alpha`** → **`gamma`, `delta`** (keep aux weights fixed if both heads matter equally for the thesis narrative).

## 3. Standalone explainability (`explainability.py`)

After training:

```bash
python3 source/kd/explainability.py \
  --model results_kd/student_kd_best.pth \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --out results_kd/explanations.json \
  --n 10 \
  --prompt \
  --device cpu
```

`--prompt` prints a sample LLM prompt. Match `--student-hidden` / `--teacher-hidden` to training (defaults 256 / 512).

## 4. Standalone ablation (`ablation_study.py`)

```bash
python3 source/kd/ablation_study.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --kd-model results_kd/student_kd_best.pth \
  --baseline-model results_kd/student_baseline_best.pth \
  --teacher-model results_kd/teacher_best.pth \
  --out results_kd/ablation_report.json \
  --epochs 60 \
  --batch 32 \
  --device cpu
```

You may omit `--kd-model` / `--baseline-model` / `--teacher-model` if you only want to train/compare sub-models from scratch (fewer pretrained baselines).

## 5. Feature extraction (`extract_features.py`)

For local runs with videos + CSV:

```bash
python3 source/kd/extract_features.py --help
```

On Kaggle, extract once to JSON, then use `run_experiment.py` locally on that JSON.

## 6. End-to-end sanity check (`test_pipeline.py`)

Generates synthetic JSON and runs a quick experiment; see the file for the exact command.

## File map

| File | Role |
|------|------|
| `models.py` | `TeacherModel`, `StudentModel` |
| `run_experiment.py` | Train + compare + optional ablation/explain |
| `explainability.py` | Modality ablation, quality heads, LLM prompt |
| `ablation_study.py` | Visual-only, text-only, concat fusion, inference time |
| `extract_features.py` | Heavy teachers → JSON features |
| `test_pipeline.py` | Synthetic quick test |

---

**TL;DR:** Run `run_experiment.py` with enough epochs and `--explain` + `--ablation`, then adjust `alpha` and `beta` using section 2.
