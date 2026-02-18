# Distil-ShortVU 🎥

**Distil-ShortVU** distills multimodal video understanding into a lightweight student model that predicts video quality scores (aesthetic, technical) and engagement rate (ECR) from short videos.

> 🍎 **Note:** This project is optimized for **Apple Silicon (M1/M2/M3)** using Metal Performance Shaders (MPS) for GPU acceleration, avoiding CUDA dependencies.

## 🚀 Features

*   **Multi-Pass Analysis Pipeline:**
    *   **Pass 1 (TimeChat):** Generates detailed captions and acts as a "Video Critic" to score aesthetic quality (1-10).
    *   **Pass 2 (Qwen2.5):** Generates a logical rationale explaining *why* the video received its score based on content, motion (ECR), and aesthetics.
*   **Motion Analysis:** Calculates ECR (Edge Change Ratio) to quantify visual motion.
*   **Mac Optimization:** Runs natively on macOS using `mps` (Metal).

## 📂 Project Structure

```text
source/
  ├── pipeline_fast.py     # Optimized data processing pipeline (~4s/video)
  ├── student_model_v2.py  # Student model architectures (MLP / Transformer / Full)
  ├── train_v2.py          # Training script
  ├── run_student_v2.py    # Inference script
  └── download_data.py     # Data download utility

checkpoints/               # ImageBind pretrained weights
checkpoints_v2/            # Trained student model weights

data/
  ├── train_data.csv       # Video metadata (Id, ECR, Title, Description)
  ├── train_videos/        # ~105k short videos
  └── train_processed_v2.json  # Pipeline output (embeddings + scores + captions)

third-party/
  └── ImageBind/           # Meta AI multimodal embedding model
```

## 🛠️ Installation

### Prerequisites
*   Python 3.10
*   Conda (Anaconda or Miniconda)
*   macOS with Apple Silicon (Recommended)

### Setup Environment

```bash
# 1. Create and activate conda environment
conda create -n longvu-env python=3.10
conda activate longvu-env

# 2. Install dependencies
pip install -r requirements.txt
```

**Key dependencies:** `torch==2.1.2`, `numpy==1.26.4`, `transformers==4.44.2`, `pyiqa`, `decord`.

## ⚙️ Pipeline & Models

### Feature Extraction
The pipeline extracts 3 core features per video:

| Component | Model | Output | Time/video |
|-----------|-------|--------|------------|
| **Scoring** | pyiqa (MUSIQ + TOPIQ) | Aesthetic & Technical scores (0-10) | ~2.0s |
| **Caption** | BLIP base | Text description | ~0.7s |
| **Embedding** | ImageBind | 1024-dim multimodal vector | ~1.1s |

### Student Models
Three architectures are available for the student model:

| Model | Params | Input | Best For |
|-------|--------|-------|----------|
| **StudentMLP** | 1.2M | Precomputed embeddings | Fast training & inference |
| **StudentTransformer** | 13.5M | Embeddings + optional text | Better accuracy |
| **ViralStudentV2** | 44M+ | Embeddings or raw video | Full capability |

### Performance Results (105k training samples)

| Metric | Value |
|--------|-------|
| **ECR Pearson Correlation** | **0.633** |
| ECR Spearman Correlation | 0.625 |
| ECR Kendall Tau | 0.453 |
| ECR MAE | 0.174 |
| Aesthetic Pearson | 0.868 |
| Technical Pearson | 0.845 |
| Binary Accuracy (ECR > 0.5) | 0.738 |

## 💻 Usage

### 1. Run Processing Pipeline

The script loads models sequentially to manage memory.

```bash
# Process all videos (resumes automatically)
python source/pipeline_fast.py --csv data/train_data.csv --videos data/train_videos

# Process with limit (e.g., first 1000 videos)
python source/pipeline_fast.py --csv data/train_data.csv --videos data/train_videos --max 1000

# Process validation set
python source/pipeline_fast.py --csv data/val_data.csv --videos data/val_videos --out data/val_processed_v2.json
```

### 2. Train Student Model

```bash
# Train MLP (Recommended - Fastest)
python source/train_v2.py \
  --data data/train_processed_v2.json \
  --val-data data/val_processed_v2.json \
  --model mlp --epochs 50 --batch 64 --lr 5e-4

# Train Transformer
python source/train_v2.py \
  --data data/train_processed_v2.json \
  --val-data data/val_processed_v2.json \
  --model transformer --epochs 30 --batch 128 --lr 3e-4
```

### 3. Inference

```bash
# From precomputed data (Fast)
python source/run_student_v2.py \
  --checkpoint checkpoints_v2/student_v2_mlp_best.pth \
  --data data/val_processed_v2.json \
  --output data/val_predictions.json

# From raw video file
python source/run_student_v2.py \
  --checkpoint checkpoints_v2/student_v2_mlp_best.pth \
  --video path/to/video.mp4
```

## ⚠️ Troubleshooting on Mac

*   **Flash Attention Error:** If you encounter errors related to `flash_attn`, ensure you are using the provided wrapper classes in `pipeline.py` which bypasses this requirement for MPS.
*   **Memory Issues:** The pipeline is designed to load/unload models sequentially. A minimum of **16GB Unified Memory** is recommended. For 8GB machines, ensure all other applications are closed.

## 📝 License

[Your License Here]
