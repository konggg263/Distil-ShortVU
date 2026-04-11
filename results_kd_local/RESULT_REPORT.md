# RESULT_REPORT — Distil-ShortVU (KD Pipeline)

> Báo cáo tự động sinh từ `experiment_report.json` trong cùng thư mục.  
> **Lưu ý:** Mỗi lần chạy lại `run_experiment.py` số liệu thay đổi (seed, epoch, hyperparam). Cập nhật bằng cách chạy lại thí nghiệm và giữ file JSON mới.

---

## 1. Cấu hình dữ liệu

| Trường | Giá trị (lần chạy ghi trong JSON) |
|--------|-----------------------------------|
| `data_path` | `source/kaggle_kd/results/500_videos/features_500.json` |
| Train / Val | 400 / 100 |
| Biến mục tiêu | ECR (hồi quy) |

---

## 2. Tổng quan pipeline (`source/kd/run_experiment.py`)

1. **Load JSON** → `KDDataset`: `visual_emb` (1024), `text_emb` (384), `quality_scores` (aesthetic/technical scale 0–1), `ecr`.
2. **Train Teacher** — `TeacherModel`: visual + text + quality → cross-attention → residual → ECR + `hidden` (512). Loss: MSE với nhãn ECR.
3. **Sinh soft target** — `generate_teacher_targets`: mỗi mẫu thêm `teacher_ecr`, `teacher_hidden`.
4. **Train Student Baseline** — `StudentModel`: chỉ `ecr_hard` (các trọng số KD = 0).
5. **Train Student + KD** — cùng kiến trúc Student; loss: hard + α·soft ECR + β·repr (cosine) + γ·aesthetic + δ·technical.
6. **So sánh** — PLCC, SRCC, KTAU, MSE, MAE trên val.
7. **Tùy chọn** — `--ablation`, `--explain` → ghi thêm vào JSON và file `ablation_report.json`, `explanations.json`.

---

## 3. Kết quả chính (validation)

Nguồn: `results_kd_local/experiment_report.json` (snapshot hiện tại).

### 3.1 Bảng so sánh ba mô hình

| Model | Params | PLCC | SRCC | KTAU | MSE | MAE |
|-------|--------|------|------|------|-----|-----|
| **Teacher** | 6,114,305 | **0.2622** | **0.3104** | **0.2225** | 0.000943 | 0.0250 |
| **Student + KD** | 890,115 | 0.1443 | 0.1477 | 0.0973 | 0.000973 | 0.0265 |
| **Student Baseline** | 890,115 | **0.2388** | **0.2837** | **0.1927** | 0.000945 | 0.0260 |

- **Nén tham số:** Teacher → Student ≈ **6.9×** (6.11M → 0.89M).
- **`kd_gain_plcc` (snapshot):** **−0.0945** → Student+KD **thấp hơn** Baseline về PLCC trong lần chạy này. Cần điều chỉnh α, β, epoch hoặc early stopping theo PLCC thay vì chỉ MSE nếu mục tiêu báo cáo là tương quan tuyến tính.

**Trọng số KD đã dùng:** `ecr_hard=1.0`, `ecr_soft=0.3`, `kd_repr=0.1`, `aesthetic=0.2`, `technical=0.2`.

### 3.2 Ablation (cùng snapshot)

| Model | Params | PLCC | SRCC | Infer (ms/video) |
|-------|--------|------|------|-------------------|
| Teacher | 6,114,305 | 0.2622 | 0.3104 | 1.288 |
| Student + KD | 890,115 | 0.1443 | 0.1477 | 0.169 |
| Student Baseline | 890,115 | 0.2388 | 0.2837 | 0.164 |
| **Visual-only** | 822,785 | **0.3205** | 0.3189 | 0.130 |
| Text-only | 658,945 | 0.1917 | 0.2380 | 0.126 |
| Concat fusion | 1,120,001 | 0.2276 | 0.2473 | 0.198 |

**Đọc nhanh:**

- **Gated multimodal (Student baseline)** vs **Concat:** Baseline PLCC 0.2388 > Concat 0.2276 → gated tốt hơn concat trong snapshot này.
- **Visual-only** PLCC cao nhất bảng (0.32) — trên tập 500 mẫu và split này, nhánh visual-only (ablation) có thể khớp noise/đặc thù split; multimodal gated chưa vượt visual-only trong metric PLCC (cần kiểm tra lại trên nhiều seed hoặc tập lớn hơn).
- **Teacher** nhanh hơn visual-only ablation về tham số nhưng chậm hơn ~7–8× so với Student (ms/video).

### 3.3 Explainability (20 mẫu val, snapshot)

| Chỉ số | Giá trị |
|--------|---------|
| `visual_dominant_pct` | 20% |
| `avg_visual_importance` | 34.9% |
| `avg_text_importance` | 65.1% |
| `avg_predicted_ecr` | 0.0472 |
| `avg_aesthetic_10` | 5.22 |
| `avg_technical_10` | 4.24 |

Chi tiết từng video: `explanations.json`.

---

## 4. File artefact trong `results_kd_local/`

| File | Mô tả |
|------|--------|
| `experiment_report.json` | Metrics + ablation + explain summary (nguồn số liệu chính) |
| `teacher_best.pth` | Checkpoint Teacher (val MSE tốt nhất khi train) |
| `student_baseline_best.pth` | Student không KD |
| `student_kd_best.pth` | Student có KD |
| `ablation_report.json` | Bản sao/tóm ablation nếu chạy standalone |
| `ablation_*.pth` | Visual-only, text-only, concat |
| `explanations.json` | Ablation modality + quality heads + metadata |
| `RESULT_REPORT.md` | Tài liệu này |

---

## 5. Ánh xạ code — từng thành phần làm gì

| File | Vai trò |
|------|---------|
| `source/kd/models.py` | `TeacherModel`, `StudentModel`, loss trong `StudentModel.forward` |
| `source/kd/run_experiment.py` | Load data, train 3 phase, evaluate, optional ablation/explain |
| `source/kd/ablation_study.py` | Visual-only, text-only, concat; đo inference ms |
| `source/kd/explainability.py` | Zero-out visual/text, đọc head aesthetic/technical, prompt LLM |
| `source/kd/extract_features.py` | Offline: ImageBind, BLIP caption, MiniLM text, MUSIQ/TOPIQ → JSON |

---

## 6. Hạn chế & cách diễn giải với thầy

1. **Checkpoint chọn theo val MSE** có thể không trùng checkpoint tốt nhất về PLCC — Student+KD dễ bị kéo bởi `L_repr` và soft target trong khi PLCC tụt.
2. **ECR biên độ hẹp** (thường ~0–0.12) → PLCC tuyệt đối thấp, dao động giữa các lần chạy.
3. **Visual-only ablation** là kiến trúc khác (ít layer hơn multimodal); so sánh “công bằng” cần cùng capacity hoặc báo cáo rõ đây là ablation kiến trúc, không chỉ “tắt modality” trên cùng graph.

---

## 7. Next steps (đề xuất)

1. **Tune KD:** giảm `beta` (repr), giảm `alpha` (soft), hoặc tăng epoch; thử early stopping theo val PLCC/SRCC.
2. **Nhiều seed / K-fold** để báo cáo khoảng tin cậy.
3. **Tăng N mẫu** (ví dụ 2k–5k) sau khi extract trên Kaggle.
4. **Lưu thêm checkpoint** theo metric PLCC song song MSE.
5. Cập nhật lại **RESULT_REPORT.md** sau mỗi lần chạy ổn định (hoặc tự động hóa bằng script đọc JSON).

---

