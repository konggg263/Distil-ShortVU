# Distil-ShortVU 🎥

**Distil-ShortVU** is a comprehensive video quality assessment pipeline designed to analyze short videos. It leverages Multi-modal Large Language Models (MLLMs) to automatically caption, score, and generate rationales for video aesthetics and technical quality.

> **Note:** This project is optimized for **Apple Silicon (M1/M2/M3)** using Metal Performance Shaders (MPS).

## 🚀 Features

- **Multi-Pass Analysis Pipeline:**
  - **Pass 1 (TimeChat):** Generates detailed captions and acts as a "Video Critic" to score aesthetic quality (1-10).
  - **Pass 2 (Qwen2.5):** Generates a logical rationale explaining *why* the video received its score based on content, motion (ECR), and aesthetics.
- **Motion Analysis:** Calculates ECR (Edge Change Ratio) to quantify visual motion.
- **Mac Optimization:** Runs natively on macOS using `mps` (Metal) for GPU acceleration, avoiding CUDA dependencies.

## 🛠️ Installation

### Prerequisites
- Python 3.10
- Conda (Anaconda or Miniconda)
- macOS with Apple Silicon (recommended for current config)

### Setup Environment

```bash
# 1. Create and activate conda environment
conda create -n venv python=3.10
conda activate venv

# 2. Install dependencies
pip install -r requirements.txt
```

The script will:
1. Load **TimeChat-7B** to caption and score videos.
2. Unload TimeChat to free up RAM.
3. Load **Qwen2.5-7B** (via MLX) to generate rationales.
4. Save results incrementally to `data/train_processed.json`.

## 🤖 Models Used

| Component | Model | Role | Device |
|-----------|-------|------|--------|
| **Video VLM** | `TimeChat-7B` | Captioning & Aesthetic Scoring | MPS (GPU) |
| **LLM** | `Qwen2.5-7B-Instruct` | Rationale Generation | MPS (MLX) |

## ⚠️ Troubleshooting on Mac

- **Flash Attention Error:** If you encounter errors related to `flash_attn`, ensure you are using the provided wrapper classes in `pipeline.py` which bypasses this requirement for MPS.
- **Memory Issues:** The pipeline is designed to load/unload models sequentially. A minimum of **16GB Unified Memory** is recommended. For 8GB machines, close other applications.

## 📝 License

[Your License Here]
```

```
