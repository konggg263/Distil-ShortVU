# Distil-ShortVU: Lộ Trình Hoàn Thiện Khóa Luận

## 📍 Bạn Đang Ở Đây

```
[✅ Done] ──────────────────────────────────────────── [🎯 Bảo vệ]
Pipeline → Features → Model → Training → Ablation → Report → Demo → Bảo vệ
  ✅        ✅        ✅      ⬅ ĐANG LÀM                           07/2026
```

---

## Phase 1: Chạy Notebook trên Kaggle (Tháng 4)
> **Trạng thái: Đang làm**

### Bước 1.1: Chạy lần 1 với 5000 videos
- Upload `distil-shortvu-final.ipynb` lên Kaggle
- Run All → kiểm tra ECR Pearson **phải > 0** (kỳ vọng 0.3-0.5)
- Nếu Pearson vẫn = 0 → báo lại để debug tiếp
- Thời gian: ~4-5 tiếng

### Bước 1.2: Tăng data lên 20k-50k videos
- Sửa Cell 13: `max_videos=None` hoặc `max_videos=50000`
- Re-run toàn bộ (feature extraction ~15-30 tiếng, cần dùng nhiều session Kaggle)
- Kỳ vọng: Pearson tăng lên 0.5-0.7

### Bước 1.3: Lưu kết quả
- Download các file output từ Kaggle:
  - `best_model.pth` (model weights)
  - `val_predictions.json` (predictions)
  - `ablation_results.json` (ablation study)
  - Tất cả `.png` (plots)
  - `distil-shortvu-final.ipynb` (notebook có outputs)

---

## Phase 2: Viết Báo Cáo Khóa Luận (Tháng 5-6)

### Cấu trúc báo cáo đề xuất (theo đề cương):

#### Chương 1: Giới thiệu (~5-8 trang)
- Bối cảnh và động lực nghiên cứu
- Mục tiêu đề tài
- Phạm vi và giới hạn
- Cấu trúc báo cáo

#### Chương 2: Cơ sở lý thuyết & Công trình liên quan (~10-15 trang)
- 2.1 Video engagement prediction
- 2.2 Multimodal learning & fusion methods
  - Concatenation, attention-based, gated fusion
- 2.3 Knowledge Distillation
  - KD cổ điển (Hinton 2015), FitNets, CRD
  - Representation distillation vs logit distillation
- 2.4 Các mô hình tiền huấn luyện sử dụng
  - ImageBind, MUSIQ, TOPIQ, BLIP, MiniLM
- 2.5 Bộ dữ liệu SnapUGC

#### Chương 3: Phương pháp đề xuất - Distil-ShortVU (~10-12 trang)
- 3.1 Tổng quan pipeline (hình vẽ kiến trúc tổng thể)
- 3.2 Teacher Ensemble & Feature Extraction
  - Visual: ImageBind → 1024-dim
  - Quality: MUSIQ + TOPIQ → aesthetic/technical scores
  - Caption: BLIP → text
  - Text: MiniLM → 384-dim
- 3.3 Student Model Architecture
  - Visual encoder, Text encoder
  - Gated Multimodal Fusion (công thức toán)
  - Multi-task heads (ECR, aesthetic, technical)
  - KD Projector (cosine similarity)
- 3.4 Hàm mất mát đa nhiệm vụ
  - L = L_ECR + λ₁·L_aes + λ₂·L_tech + λ₃·L_KD
- 3.5 Cơ chế giải thích (Ablation-based Explainability)

#### Chương 4: Thực nghiệm (~12-15 trang)
> **Đây là phần quan trọng nhất — dùng kết quả từ notebook**

- 4.1 Cài đặt thực nghiệm
  - Dataset: SnapUGC (số lượng train/val/test)
  - Hardware: Kaggle GPU (T4/P100)
  - Hyperparameters: bảng tổng hợp
- 4.2 Kết quả chính
  - Bảng metrics: Pearson, Spearman, Kendall, MAE, MSE
  - Training curves (copy từ notebook)
  - Scatter plot true vs predicted ECR
- 4.3 Ablation Study
  - 4.3.1 So sánh modality (bảng + biểu đồ từ notebook)
  - 4.3.2 So sánh fusion method (bảng + biểu đồ)
  - 4.3.3 So sánh training strategy (bảng + biểu đồ)
- 4.4 Phân tích bổ sung
  - Feature importance (visual vs text)
  - Gate weights distribution
  - Error analysis by ECR range
  - Inference time comparison (student vs teacher)
- 4.5 Giải thích dự đoán
  - Ví dụ explanations cho các video mẫu
- 4.6 Thảo luận kết quả
  - Trả lời 4 câu hỏi nghiên cứu trong đề cương:
    1. Teacher ensemble có giúp cải thiện? → So sánh Full KD vs ECR-only
    2. Gated fusion tốt hơn concat? → So sánh ablation 2
    3. Quality scores có hỗ trợ? → So sánh Multi-task vs ECR-only
    4. Ablation explainability có làm rõ? → Phân tích feature importance

#### Chương 5: Kết luận & Hướng phát triển (~3-5 trang)
- Tóm tắt kết quả
- Trả lời lại mục tiêu đề tài
- Hạn chế (chỉ 5k videos, chưa test trên TikTok/YouTube,...)
- Hướng phát triển (tăng data, thêm audio modality, real-time demo,...)

#### Tài liệu tham khảo
- Đã có sẵn 35 refs trong đề cương

---

## Phase 3: Xây Dựng Demo (Tháng 6)

> Đề cương ghi: "giao diện demo đơn giản cho phép tải video, hiển thị điểm dự đoán và giải thích"

### Option A: Gradio (Nhanh nhất, recommend)
```python
import gradio as gr

def predict_video(video_file):
    # 1. Extract features (ImageBind + BLIP + MUSIQ/TOPIQ + MiniLM)
    # 2. Run student model
    # 3. Generate explanation
    return f"""
    🎯 ECR: {ecr:.2f}
    🎨 Aesthetic: {aes:.1f}/10
    🔧 Technical: {tech:.1f}/10
    
    📊 Visual: {vis_pct:.0f}% | Text: {txt_pct:.0f}%
    
    {explanation}
    """

demo = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload short video"),
    outputs=gr.Textbox(label="Prediction & Explanation"),
    title="Distil-ShortVU: Video Engagement Predictor",
    description="Predict engagement rate of short videos using knowledge distillation"
)
demo.launch()
```

### Option B: Streamlit (Đẹp hơn)
- Upload video → hiển thị video + metrics + explanation + bar charts

### Yêu cầu cho demo:
- Load `best_model.pth` đã train
- Cần ImageBind + BLIP + pyiqa + MiniLM cho feature extraction
- Chạy được trên Mac (MPS) hoặc Colab (CUDA)

---

## Phase 4: Chuẩn Bị Bảo Vệ (Tháng 7)

### Slide bảo vệ (~20-25 slides):
1. Title slide
2. Bối cảnh & Động lực (2-3 slides)
3. Mục tiêu & Câu hỏi nghiên cứu (1 slide)
4. Cơ sở lý thuyết (3-4 slides)
5. Phương pháp đề xuất - Pipeline diagram (3-4 slides)
6. Kiến trúc DistilStudent (2 slides)
7. Kết quả thực nghiệm (4-5 slides)
   - Bảng metrics chính
   - Ablation study results
   - Feature importance
   - Sample explanations
8. Demo (1-2 slides hoặc live demo)
9. Kết luận & Hướng phát triển (1-2 slides)
10. Q&A

### Checklist trước bảo vệ:
- [ ] Notebook chạy end-to-end không lỗi
- [ ] Báo cáo hoàn chỉnh, đọc lại 2-3 lần
- [ ] Demo hoạt động
- [ ] Slide có đủ hình ảnh/biểu đồ
- [ ] Chuẩn bị trả lời câu hỏi phản biện

---

## 📅 Timeline Tổng Hợp

| Tháng | Việc cần làm | Ai |
|-------|-------------|-----|
| **04/2026** | Chạy notebook (5k) → fix bugs → chạy lại (20k+) | Cả hai |
| **05/2026** | Viết Chương 1-3 báo cáo + hoàn thiện ablation | Tuấn Công |
| **05/2026** | Viết Chương 4 (thực nghiệm) từ kết quả notebook | Anh Thư |
| **06/2026** | Xây demo Gradio/Streamlit | Anh Thư |
| **06/2026** | Viết Chương 5, hoàn thiện báo cáo, trực quan hóa | Tuấn Công |
| **07/2026** | Slide bảo vệ + rehearsal + nộp bài | Cả hai |

---

## 🎯 Ưu Tiên Ngay Bây Giờ

1. ⬜ Upload notebook lên Kaggle, bật GPU, Run All
2. ⬜ Kiểm tra ECR Pearson > 0 sau training
3. ⬜ Nếu OK → tăng data lên 20k+, chạy lại
4. ⬜ Download tất cả outputs (model, plots, results)
5. ⬜ Bắt đầu viết báo cáo Chương 3 (phương pháp) — có thể viết song song
