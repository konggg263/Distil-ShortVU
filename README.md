# Distil-ShortVU 🎥# Distil-ShortVU 🎥# Distil-ShortVU 🎥



**Distil-ShortVU** distills multimodal video understanding into a lightweight student model that predicts video quality scores (aesthetic, technical) and engagement rate (ECR) from short videos.



> Optimized for **Apple Silicon (M1/M2/M3)** using Metal Performance Shaders (MPS).**Distil-ShortVU** distills multimodal video understanding into a lightweight student model that predicts video quality scores (aesthetic, technical) and engagement rate (ECR) from short videos.**Distil-ShortVU** is a comprehensive video quality assessment pipeline designed to analyze short videos. It leverages Multi-modal Large Language Models (MLLMs) to automatically caption, score, and generate rationales for video aesthetics and technical quality.



## Project Structure



```> Optimized for **Apple Silicon (M1/M2/M3)** using Metal Performance Shaders (MPS).> **Note:** This project is optimized for **Apple Silicon (M1/M2/M3)** using Metal Performance Shaders (MPS).

source/

  pipeline_fast.py     # Optimized data processing pipeline (~4s/video)

  student_model_v2.py  # Student model architectures (MLP / Transformer / Full)

  train_v2.py          # Training script## Project Structure## 🚀 Features

  run_student_v2.py    # Inference script

  download_data.py     # Data download utility

checkpoints/           # ImageBind pretrained weights

checkpoints_v2/        # Trained student model weights```- **Multi-Pass Analysis Pipeline:**

data/

  train_data.csv       # Video metadata (Id, ECR, Title, Description)source/  - **Pass 1 (TimeChat):** Generates detailed captions and acts as a "Video Critic" to score aesthetic quality (1-10).

  train_videos/        # ~105k short videos

  train_processed_v2.json  # Pipeline output (embeddings + scores + captions)  pipeline_fast.py     # Optimized data processing pipeline (~3.9s/video)  - **Pass 2 (Qwen2.5):** Generates a logical rationale explaining *why* the video received its score based on content, motion (ECR), and aesthetics.

third-party/

  ImageBind/           # Meta AI multimodal embedding model  student_model_v2.py  # Student model architectures (MLP / Transformer / Full)- **Motion Analysis:** Calculates ECR (Edge Change Ratio) to quantify visual motion.

```

  train_v2.py          # Training script- **Mac Optimization:** Runs natively on macOS using `mps` (Metal) for GPU acceleration, avoiding CUDA dependencies.

## Pipeline

  run_student_v2.py    # Inference script

The pipeline extracts 3 features per video:

  download_data.py     # Data download utility## 🛠️ Installation

| Component | Model | Output | Time/video |

|-----------|-------|--------|------------|checkpoints/           # ImageBind pretrained weights

| **Scoring** | pyiqa (MUSIQ + TOPIQ) | Aesthetic & Technical scores (0-10) | ~2.0s |

| **Caption** | BLIP base | Text description | ~0.7s |checkpoints_v2/        # Trained student model weights### Prerequisites

| **Embedding** | ImageBind | 1024-dim multimodal vector | ~1.1s |

data/- Python 3.10

### Run Pipeline

  train_data.csv       # Video metadata (Id, ECR, Title, Description)- Conda (Anaconda or Miniconda)

```bash

# Process all videos (resumes automatically)  train_videos/        # ~105k short videos- macOS with Apple Silicon (recommended for current config)

python source/pipeline_fast.py --csv data/train_data.csv --videos data/train_videos

  train_processed_v2.json  # Pipeline output (embeddings + scores + captions)

# Process with limit

python source/pipeline_fast.py --csv data/train_data.csv --videos data/train_videos --max 1000third-party/### Setup Environment



# Process validation set (no ECR labels)  ImageBind/           # Meta AI multimodal embedding model

python source/pipeline_fast.py --csv data/val_data.csv --videos data/val_videos --out data/val_processed_v2.json

`````````bash



## Student Model# 1. Create and activate conda environment



Three architectures available:## Pipelineconda create -n venv python=3.10



| Model | Params | Best ECR Corr | Best For |conda activate venv

|-------|--------|---------------|----------|

| **StudentMLP** | 1.2M | **0.633** | Fast training & inference |The pipeline extracts 3 features per video:

| **StudentTransformer** | 13.5M | 0.620 | Experimental |

| **ViralStudentV2** | 44M+ | - | Full capability (with LLM) |# 2. Install dependencies



### Results (105k training samples)| Component | Model | Output | Time/video |pip install -r requirements.txt



| Metric | Value ||-----------|-------|--------|------------|```

|--------|-------|

| ECR Pearson Correlation | **0.633** || **Scoring** | pyiqa (MUSIQ + TOPIQ) | Aesthetic & Technical scores (0-10) | ~2.0s |

| ECR Spearman Correlation | 0.625 |

| ECR Kendall Tau | 0.453 || **Caption** | BLIP base | Text description | ~0.7s |The script will:

| ECR MAE | 0.174 |

| Aesthetic Pearson | 0.868 || **Embedding** | ImageBind | 1024-dim multimodal vector | ~1.1s |1. Load **TimeChat-7B** to caption and score videos.

| Technical Pearson | 0.845 |

| Binary Accuracy (ECR > 0.5) | 0.738 |2. Unload TimeChat to free up RAM.

| Top-20% Overlap | 0.495 |

### Run Pipeline3. Load **Qwen2.5-7B** (via MLX) to generate rationales.

### Train

4. Save results incrementally to `data/train_processed.json`.

```bash

# Train MLP (recommended)```bash

python source/train_v2.py \

  --data data/train_full.json \# Process all videos (resumes automatically)## 🤖 Models Used

  --val-data data/val_full.json \

  --model mlp --epochs 15 --batch 128 --lr 5e-4python source/pipeline_fast.py --csv data/train_data.csv --videos data/train_videos



# Train Transformer| Component | Model | Role | Device |

python source/train_v2.py \

  --data data/train_full.json \# Process with limit|-----------|-------|------|--------|

  --val-data data/val_full.json \

  --model transformer --epochs 30 --batch 128 --lr 3e-4python source/pipeline_fast.py --csv data/train_data.csv --videos data/train_videos --max 1000| **Video VLM** | `TimeChat-7B` | Captioning & Aesthetic Scoring | MPS (GPU) |

```

```| **LLM** | `Qwen2.5-7B-Instruct` | Rationale Generation | MPS (MLX) |

### Inference



```bash

# From precomputed data (fast)## Student Model## ⚠️ Troubleshooting on Mac

python source/run_student_v2.py \

  --checkpoint checkpoints_v2/student_v2_mlp_best.pth \

  --data data/val_processed_v2.json \

  --output data/val_predictions.jsonThree architectures available:- **Flash Attention Error:** If you encounter errors related to `flash_attn`, ensure you are using the provided wrapper classes in `pipeline.py` which bypasses this requirement for MPS.



# From raw video- **Memory Issues:** The pipeline is designed to load/unload models sequentially. A minimum of **16GB Unified Memory** is recommended. For 8GB machines, close other applications.

python source/run_student_v2.py \

  --checkpoint checkpoints_v2/student_v2_mlp_best.pth \| Model | Params | Input | Best For |

  --video path/to/video.mp4

```|-------|--------|-------|----------|## 📝 License



## Setup| **StudentMLP** | 1.2M | Precomputed embeddings | Fast training & inference |



```bash| **StudentTransformer** | 13.5M | Embeddings + optional text | Better accuracy |[Your License Here]

conda create -n longvu-env python=3.10

conda activate longvu-env| **ViralStudentV2** | 44M+ | Embeddings or raw video | Full capability |```

pip install -r requirements.txt

```



Key dependencies: `torch==2.1.2`, `numpy==1.26.4`, `transformers==4.44.2`, `pyiqa`, `decord`### Train```



```bash
# Train MLP (recommended - fastest)
python source/train_v2.py --data data/train_processed_v2.json --model mlp --epochs 50 --batch 64

# Train Transformer
python source/train_v2.py --data data/train_processed_v2.json --model transformer --epochs 100

# With validation
python source/train_v2.py --data data/train_processed_v2.json --val-data data/val_processed_v2.json --model mlp --epochs 50
```

### Inference

```bash
# From precomputed data (fast)
python source/run_student_v2.py --checkpoint checkpoints_v2/student_v2_mlp_best.pth --data data/test.json

# From raw video
python source/run_student_v2.py --checkpoint checkpoints_v2/student_v2_mlp_best.pth --video path/to/video.mp4
```

## Setup

```bash
conda create -n longvu-env python=3.10
conda activate longvu-env
pip install -r requirements.txt
```

Key dependencies: `torch==2.1.2`, `numpy==1.26.4`, `transformers==4.44.2`, `pyiqa`, `decord`
