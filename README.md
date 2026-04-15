# Distil-ShortVU 🎥

**Distil-ShortVU** is a research framework for distilling heavy, multimodal video understanding models into lightweight, high-performance student models. It focuses on predicting short video engagement (ECR) and internal quality metrics (Aesthetic, Technical) with minimal parameter overhead.

> 🍎 **Note:** This project is highly optimized for **Apple Silicon (M1/M2/M3)** using Metal Performance Shaders (MPS), allowing for efficient local training and inference without NVIDIA GPUs.

---

## 🚀 Key Features

*   **Multimodal Knowledge Distillation (KD):** Distills a 6.1M parameter Teacher model into an 890K parameter Student model using Gated Cross-Attention.
*   **Multi-Task Learning:** Simultaneous training on Engagement (ECR) and Auxiliary tasks (Aesthetic & Technical quality).
*   **Explainability (XAI) Engine:** Mathematically-grounded Modality Ablation and Temporal Hook discovery (identifying the most engaging second of a video).
*   **Hardware Optimized:** Native support for `mps` on Mac and `cuda` on Linux/Windows.
*   **Ablation Suite:** Automated benchmarks comparing Gated architectures against Visual-only, Text-only, and blind Concatenation baselines.

---

## 📂 Project Structure (Current)

```text
source/kd/
  ├── run_experiment.py     # Main CLI orchestrator (Train + Compare + XAI)
  ├── models.py             # Architectures (Teacher & Gated-Student)
  ├── explainability.py     # XAI Engine (Ablation-based insights)
  ├── ablation_study.py     # Architecture benchmarking logic
  ├── extract_features.py   # Feature extraction script (usually run on GPU)
  └── test_pipeline.py      # Quick E2E test suite

results_kd_local/           # Default output directory for experiments
```

---

## 🛠️ Installation

### Setup Environment

```bash
# 1. Create conda environment
conda create -n distil-shortvu python=3.10
conda activate distil-shortvu

# 2. Install dependencies
pip install -r requirements.txt
```

*Required: `torch==2.1.2`, `torchvision`, `torchaudio`, `transformers`, `scipy`, `numpy`, `pandas`, `decord`.*

---

## ⚙️ Workflow & Pipeline

The pipeline is split into two stages to optimize for memory constraints:

1.  **Feature Extraction:** Run `extract_features.py` (ideally on Kaggle/GPU) to generate a `features.json` file containing ImageBind embeddings, text embeddings, and quality scores.
2.  **Local Distillation:** Run `run_experiment.py` on your local machine to train the teacher and distill the student.

### Training the Experiment

Run the complete pipeline (Teacher Training -> KD Student Training -> Ablation -> XAI Analysis):

```bash
python3 source/kd/run_experiment.py \
  --data path/to/features_5000.json \
  --save-dir results_kd_local \
  --device mps \
  --teacher-epochs 100 \
  --student-epochs 120 \
  --alpha 0.3 --beta 0.1 --gamma 0.2 --delta 0.2 \
  --explain --ablation
```

**Hyperparameters:**
*   `--alpha`: Weight for ECR Soft-Target Distillation.
*   `--beta`: Weight for Representation KD.
*   `--gamma/--delta`: Weights for Aesthetic/Technical auxiliary tasks.

---

## 🧠 Explainability (XAI)

The engine provides **Scientifically Grounded** explanations (No Hallucinations):

1.  **Modality Ablation:** Calculates the exact percentage impact of Visual vs. Text features on the final score.
2.  **Temporal Hook Discovery:** Analytically identifies which specific frame/second contributes most to the video's engagement drop if removed.
3.  **LLM Prompt Builder:** Automatically formats mathematical findings into a structured prompt for ChatGPT/Gemini to generate human-readable reports.

---

## 📊 Performance Benchmarks (Example)

| Model | Params | PLCC (ECR) | SRCC (ECR) | Gap to Teacher |
|-------|--------|------------|------------|----------------|
| **Teacher** | 6.1M | 0.428 | 0.395 | - |
| **Student + KD** | 890K | 0.385 | 0.362 | -4.3% |
| **Student Baseline** | 890K | 0.321 | 0.310 | -10.7% |

*KD typically recovers ~60% of the performance gap between a simple student and a large teacher.*

---

## 📝 License

This project is licensed under [Your License]. Developed as part of HCMUS Thesis Research.
