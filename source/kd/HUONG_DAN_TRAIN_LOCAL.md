# Hướng dẫn Train Local (CPU) — Pipeline KD hoàn chỉnh

## Tổng quan pipeline hiện tại

Bạn đã thực hiện **Bước 1** (nặng nhất) trên Kaggle GPU:

```
[Kaggle GPU] Extract features (ImageBind + MUSIQ/TOPIQ + BLIP + MiniLM)
             ↓
             results/500_videos/features_500.json (~15 MB, 500 videos)
             ↓
[Local CPU]  Train Teacher → Student baseline → Student+KD → Ablation → Explainability
```

**File đã có sẵn:** `source/kaggle_kd/results/500_videos/features_500.json`  
→ Chứa 500 mẫu, mỗi mẫu có `visual_emb` (1024-d), `text_emb` (384-d), `quality_scores`, `ecr`.

**Bước còn lại (train local):**
- Train Teacher (6.1M params)
- Train Student baseline (890K params)
- Train Student + KD (890K params)
- (Tùy chọn) Ablation study: visual-only, text-only, concat fusion
- (Tùy chọn) Explainability: modality ablation + LLM prompt

---

## Câu hỏi: Train lại local bằng CPU có lâu và khả thi không?

### Thời gian ước tính (500 videos, CPU)

| Giai đoạn | Epochs | Thời gian (CPU) | Ghi chú |
|-----------|--------|-----------------|---------|
| Teacher | 100 | ~5–8 phút | 6.1M params, batch 32 |
| Student baseline | 120 | ~3–5 phút | 890K params |
| Student + KD | 120 | ~3–5 phút | 890K params + KD losses |
| Ablation (3 models) | 60 mỗi model | ~6–9 phút | Visual-only, text-only, concat |
| Explainability | N/A | <30 giây | Forward pass only |
| **Tổng cộng** | | **~20–30 phút** | Toàn bộ pipeline |

**Kết luận:** Hoàn toàn khả thi. Với 500 videos, CPU train xong toàn bộ trong **khoảng 30 phút**.

### So sánh Kaggle GPU vs Local CPU

| | Kaggle GPU (T4) | Local CPU |
|---|-----------------|-----------|
| **Feature extraction** | ~15–20 phút (ImageBind + BLIP + MUSIQ/TOPIQ) | Không cần (đã có JSON) |
| **Training** | ~5–10 phút | ~20–30 phút |
| **Tổng** | ~25–30 phút | ~20–30 phút (vì bỏ qua extraction) |
| **Ưu điểm** | Nhanh hơn cho training | Không giới hạn thời gian session, dễ debug |
| **Nhược điểm** | Giới hạn 12h/tuần GPU, phải rerun nếu timeout | Chậm hơn một chút |

**Khuyến nghị workflow:**

1. **Kaggle:** Extract features một lần → lưu `source/kaggle_kd/results/500_videos/features_500.json` (đã xong).
2. **Local:** Train + tune hyperparameters + ablation + explainability (linh hoạt, không giới hạn).

---

## Hướng dẫn chạy end-to-end (E2E) trên Local

### Bước 0: Chuẩn bị môi trường

```bash
cd /Users/top/Documents/HCMUS/KhoaLuan/Distil-ShortVU

# Cài đặt dependencies (nếu chưa)
pip install torch scipy numpy pandas tqdm

# Kiểm tra file features đã có
ls -lh source/kaggle_kd/results/500_videos/features_500.json
# Kết quả: ~15M, 500 samples
```

### Bước 1: Chạy thí nghiệm chính (Teacher + Student baseline + Student+KD)

#### 1.1. Chạy nhanh để kiểm tra (smoke test)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_local \
  --device cpu \
  --quick
```

**Kết quả:** ~2–3 phút, tạo 3 checkpoint trong `results_kd_local/`:
- `teacher_best.pth`
- `student_baseline_best.pth`
- `student_kd_best.pth`
- `experiment_report.json`

**Lưu ý:** `--quick` chỉ dùng 15 epoch Teacher và 20 epoch Student → kết quả PLCC thấp, chỉ để test code.

#### 1.2. Chạy đầy đủ (cho báo cáo luận văn)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_local \
  --device cpu \
  --teacher-epochs 100 \
  --student-epochs 120 \
  --batch 32 \
  --alpha 0.3 \
  --beta 0.1 \
  --gamma 0.2 \
  --delta 0.2
```

**Thời gian:** ~15–20 phút.

**Tham số quan trọng:**
- `--alpha 0.3`: trọng số soft ECR (khớp với Teacher)
- `--beta 0.1`: trọng số representation KD (cosine similarity giữa hidden states)
- `--gamma 0.2`, `--delta 0.2`: trọng số auxiliary tasks (aesthetic, technical)

**Kết quả:** File `experiment_report.json` chứa PLCC, SRCC, KTAU, MSE, MAE cho cả 3 models.

### Bước 2: Thêm Explainability (giải thích)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_local \
  --device cpu \
  --teacher-epochs 100 \
  --student-epochs 120 \
  --batch 32 \
  --alpha 0.3 \
  --beta 0.1 \
  --gamma 0.2 \
  --delta 0.2 \
  --explain --explain-n 20
```

**Thêm:** `--explain --explain-n 20`

**Kết quả:** File `explanations.json` chứa:
- Modality ablation: đóng góp của visual vs text (%)
- Điểm aesthetic/technical nội sinh từ Student
- Prompt mẫu gửi LLM (ChatGPT/Claude) để sinh giải thích bằng tiếng Việt

**Thời gian thêm:** ~30 giây.

### Bước 3: Thêm Ablation Study (so sánh kiến trúc)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_local \
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

**Thêm:** `--ablation`

**Kết quả:** File `ablation_report.json` chứa:
- Visual-only Student (chỉ dùng hình ảnh)
- Text-only Student (chỉ dùng text)
- Concat Fusion Student (không có gating)
- Thời gian inference (ms/video) cho từng model

**Thời gian thêm:** ~6–9 phút (train 3 models mới, mỗi model 60 epoch).

### Bước 4 (tùy chọn): Chạy Explainability độc lập

Nếu đã có checkpoint `student_kd_best.pth` và muốn chạy lại explainability với số mẫu khác:

```bash
python3 source/kd/explainability.py \
  --model results_kd_local/student_kd_best.pth \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --out results_kd_local/explanations_full.json \
  --n 50 \
  --prompt \
  --device cpu
```

**Kết quả:** In ra console:
- Modality ablation cho 50 videos
- Dataset summary (% visual-dominant, avg importance, avg ECR)
- Prompt LLM mẫu

### Bước 5 (tùy chọn): Chạy Ablation độc lập

```bash
python3 source/kd/ablation_study.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --kd-model results_kd_local/student_kd_best.pth \
  --baseline-model results_kd_local/student_baseline_best.pth \
  --teacher-model results_kd_local/teacher_best.pth \
  --out results_kd_local/ablation_full.json \
  --epochs 80 \
  --batch 32 \
  --device cpu
```

---

## Lệnh E2E hoàn chỉnh (một lần chạy hết)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_local \
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

**Thời gian:** ~25–30 phút.

**Kết quả:**
- `results_kd_local/teacher_best.pth`
- `results_kd_local/student_baseline_best.pth`
- `results_kd_local/student_kd_best.pth`
- `results_kd_local/experiment_report.json` (PLCC, SRCC, KTAU, MSE, MAE)
- `results_kd_local/explanations.json` (modality ablation + LLM prompts)
- `results_kd_local/ablation_report.json` (visual-only, text-only, concat, inference time)
- `results_kd_local/ablation_*.pth` (checkpoints cho ablation variants)

---

## Tinh chỉnh hyperparameters (nếu KD không hiệu quả)

Nếu `Student+KD` có PLCC **thấp hơn** `Student baseline`, thử:

### Chiến lược 1: Giảm `beta` (representation KD weight)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_beta005 \
  --device cpu \
  --teacher-epochs 100 \
  --student-epochs 120 \
  --alpha 0.3 \
  --beta 0.05 \
  --gamma 0.2 \
  --delta 0.2
```

**Lý do:** `L_repr` (cosine distance) thường lớn ở đầu training, làm chậm việc học ECR. Beta thấp hơn (0.05–0.1) giúp ECR loss dẫn đầu.

### Chiến lược 2: Tăng epochs

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_long \
  --device cpu \
  --teacher-epochs 150 \
  --student-epochs 150 \
  --alpha 0.3 \
  --beta 0.1
```

**Lý do:** KD cần nhiều epoch hơn để hội tụ, đặc biệt khi dataset nhỏ và ECR variance thấp.

### Chiến lược 3: Giảm `alpha` (soft ECR weight)

```bash
python3 source/kd/run_experiment.py \
  --data source/kaggle_kd/results/500_videos/features_500.json \
  --save-dir results_kd_alpha02 \
  --device cpu \
  --teacher-epochs 100 \
  --student-epochs 120 \
  --alpha 0.2 \
  --beta 0.1
```

**Lý do:** Nếu Teacher chưa đủ tốt (PLCC ~0.27), soft target có thể gây nhiễu. Alpha thấp hơn giúp Student tập trung vào nhãn thật.

---

## So sánh kết quả

Sau khi chạy, đọc `experiment_report.json`:

```bash
cat results_kd_local/experiment_report.json | python3 -m json.tool
```

Tìm các trường:
- `teacher.plcc`, `teacher.srcc`
- `student_baseline.plcc`, `student_baseline.srcc`
- `student_kd.plcc`, `student_kd.srcc`
- `kd_gain_plcc` (Student+KD - baseline)

**Mục tiêu:** `kd_gain_plcc > 0` và Student+KD gần Teacher hơn baseline.

---

## Checklist hoàn chỉnh

- [ ] Có file `source/kaggle_kd/results/500_videos/features_500.json` (15 MB, 500 samples)
- [ ] Chạy smoke test (`--quick`) để đảm bảo code không lỗi (~3 phút)
- [ ] Chạy full experiment với `--teacher-epochs 100 --student-epochs 120` (~20 phút)
- [ ] Bật `--explain` để có explainability report (+30 giây)
- [ ] Bật `--ablation` để có ablation study (+9 phút)
- [ ] Kiểm tra `kd_gain_plcc` trong `experiment_report.json`
- [ ] Nếu KD không hiệu quả, thử giảm `--beta` và tăng epochs
- [ ] (Tùy chọn) Chạy trên toàn bộ dataset (5000 videos) nếu cần kết quả tốt hơn

---

## Lưu ý quan trọng

1. **Dataset size:** 500 videos là đủ để chứng minh pipeline hoạt động, nhưng PLCC có thể thấp (~0.2–0.3) do:
   - ECR range hẹp (0–0.12)
   - Variance thấp
   → Nếu cần PLCC cao hơn (>0.5), cần tăng lên 2000–5000 videos.

2. **CPU vs GPU:**
   - CPU: ~30 phút cho 500 videos
   - GPU (T4): ~5–10 phút cho 500 videos
   - Nếu có GPU local (NVIDIA), thêm `--device cuda` để nhanh hơn 3–5 lần.

3. **Apple Silicon (M1/M2/M3):**
   ```bash
   python3 source/kd/run_experiment.py --device mps ...
   ```
   MPS nhanh hơn CPU ~2–3 lần.

4. **Kết quả tốt nhất:**
   - Teacher PLCC ~0.27–0.35 (với 500 videos)
   - Student baseline PLCC ~0.20–0.28
   - Student+KD PLCC ~0.22–0.30 (mục tiêu: > baseline)
   - Nếu KD không vượt baseline, xem phần "Tinh chỉnh hyperparameters".

---

## Tóm tắt workflow khuyến nghị

```
[ĐÃ XONG] Kaggle GPU: Extract features → `results/500_videos/features_500.json`
                      ↓
[BÂY GIỜ]  Local CPU: python3 source/kd/run_experiment.py \
                        --data source/kaggle_kd/results/500_videos/features_500.json \
                        --save-dir results_kd_local \
                        --device cpu \
                        --teacher-epochs 100 \
                        --student-epochs 120 \
                        --alpha 0.3 --beta 0.1 \
                        --explain --ablation
                      ↓
           Kết quả: experiment_report.json + explanations.json + ablation_report.json
                      ↓
           (Nếu cần) Tinh chỉnh alpha/beta, chạy lại (~20 phút/lần)
```

**Thời gian tổng:** ~30 phút cho một lần chạy đầy đủ.
