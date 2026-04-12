import os
import sys
import tempfile
import torch
import streamlit as st

# Setup Paths to Import local packages
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third-party", "ImageBind"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source"))

# PATCH for pytorchvideo using outdated torchvision imports
import torchvision
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as functional
    sys.modules['torchvision.transforms.functional_tensor'] = functional

# Import KD Pipeline models
from source.kd.models import StudentModel
from source.kd.explainability import find_engaging_hook_frame, ExplainabilityEngine
from source.kd.extract_features import VisualEncoder, CaptionGenerator, TextEncoder

st.set_page_config(page_title="ShortVU XAI Dashboard", page_icon="🎥", layout="wide")

# ==============================================================================
# CACHING HEAVY MODELS TO AVOID RELOADING ON EVERY UI INTERACTION
# ==============================================================================
@st.cache_resource(show_spinner="Đang tải các mô hình AI (ImageBind, BLIP, Student-KD)...")
def load_models():
    # Detect best device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    # Load 3 extraction models
    visual_encoder = VisualEncoder()
    caption_gen = CaptionGenerator()
    text_encoder = TextEncoder()
    
    # Load your best Student Model (Change path if needed)
    student_model = StudentModel(hidden_dim=256).to(device)
    possible_paths = [
        os.path.join(PROJECT_ROOT, "results_kd_local", "2000_videos", "student_kd_best.pth"),
        os.path.join(PROJECT_ROOT, "results_kd_local", "student_kd_best.pth")
    ]
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if model_path is None:
        st.error("❌ Không tìm thấy model weigths `student_kd_best.pth`")
        st.stop()
        
    try:
        student_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except TypeError:
        student_model.load_state_dict(torch.load(model_path, map_location=device))
    student_model.eval()

    return student_model, visual_encoder, caption_gen, text_encoder, device

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
st.title("🎥 Giao diện Chẩn đoán Sức hút Video (Explainable AI)")
st.markdown("""
Ứng dụng sử dụng mô hình **Knowledge Distillation Student** siêu nhẹ kết hợp kỹ thuật **Temporal Ablation** để dự báo và giải thích *Khoảnh khắc vàng (The Hook)* giữ chân người xem.
""")

try:
    student_model, visual_encoder, captioner, text_encoder, device = load_models()
    st.success(f"✅ Hệ thống AI đã sẵn sàng trên `{device.upper()}`")
except Exception as e:
    st.error(f"Loadding models failed: {e}")
    st.stop()

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Upload Video Tóp Tóp / Reels")
    uploaded_file = st.file_uploader("📂 Chọn một video ngắn Định dạng mp4", type=['mp4', 'mov', 'avi'])
    title_input = st.text_input("✏️ Tiêu đề video (Tùy chọn):", placeholder="Ví dụ: Review quán ăn sinh viên...")
    desc_input = st.text_area("📝 Mô tả video / Hashtags (Tùy chọn):", placeholder="#review #food")
    
    analyze_btn = st.button("🚀 Phân Tích Độ Thu Hút (ECR) & Tìm Hook", type="primary", use_container_width=True)
    
    if uploaded_file is not None:
        st.video(uploaded_file)

with col2:
    st.header("2. Kết Quả XAI (Explainable AI)")
    
    if analyze_btn and uploaded_file is not None:
        with st.spinner("⏳ Hệ thống đang cắt Frame & Trích xuất Vector..."):
            # Save uploaded file to temp file for Decord to read
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            try:
                # Dịch Caption từ Video và Text Embed
                st.write("Đang quét nội dung ra chữ (BLIP)...")
                video_caption = captioner.caption(video_path)
                st.write("Đang embedding text (MiniLM)...")
                text_emb = text_encoder.encode(title_input, desc_input, video_caption)
                t_tensor = torch.tensor(text_emb, dtype=torch.float32).to(device)

                # ==============================================
                # Chạy Temporal Ablation để tìm THE HOOK
                # ==============================================
                st.write("Đang giả lập triệt tiêu khung hình (Temporal Ablation)...")
                hook_info = find_engaging_hook_frame(
                    video_path=video_path,
                    student_model=student_model,
                    visual_encoder=visual_encoder,
                    captioner=captioner,
                    text_emb=t_tensor,
                    device=device
                )

                if hook_info is None:
                    st.error("Video bị lỗi hoặc quá ngắn.")
                else:
                    # ==============================================
                    # Sinh Báo Cáo Giải Thích
                    # ==============================================
                    st.write("Sắp kết thúc...")
                    # Trích xuất Baseline Visual Emb (lấy từ feature_extractor giống hook_frame logic)
                    from torchvision import transforms
                    from imagebind.models.imagebind_model import ModalityType
                    import numpy as np
                    from decord import VideoReader, cpu
                    from PIL import Image

                    vr = VideoReader(video_path, ctx=cpu(0))
                    total = len(vr)
                    indices = np.linspace(0, total - 1, 4, dtype=int)
                    frames = vr.get_batch(indices).asnumpy()
                    
                    preprocess = transforms.Compose([
                        transforms.Resize(224), transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                            [0.26862954, 0.26130258, 0.27577711]),
                    ])
                    tensors = [preprocess(Image.fromarray(f)) for f in frames]
                    vis_device = getattr(visual_encoder, 'device', 'cpu')
                    video_tensor = torch.stack(tensors).unsqueeze(0).to(vis_device)
                    with torch.no_grad():
                        embs = visual_encoder.model({ModalityType.VISION: video_tensor})
                    v_emb = embs[ModalityType.VISION].cpu().numpy().flatten()
                    v_emb = v_emb / (np.linalg.norm(v_emb) + 1e-8)
                    v_tensor = torch.tensor(v_emb, dtype=torch.float32).to(device)

                    video_meta = {"title": title_input, "caption": video_caption}
                    
                    engine = ExplainabilityEngine(student_model, device)
                    explanations = engine.explain(v_tensor, t_tensor, video_meta)
                    prompt = engine.generate_llm_prompt(explanation=explanations, temporal_ablation=hook_info)

                    # IN KẾT QUẢ
                    st.success("🎉 Hoàn tất phân tích Toán học!")
                    
                    st.markdown(f"### Điểm ECR dự đoán: `{explanations['predicted_ecr']:.4f}`")
                    st.info(f"**💡 Khoảnh Khắc Hook (Ăn Tiền Nhất):**\n"
                            f"- Nằm ở **giây thứ {hook_info['hook_time_sec']}** (Frame số {hook_info['hook_frame_index']}/4)\n"
                            f"- AI Nội dung: *\"{hook_info['hook_caption']}\"*\n"
                            f"- Nếu vô tình cắt bỏ khung hình này ở khâu Editor dựng phim, mức độ thu hút sẽ giảm ngay **~{hook_info['ecr_drop_pct']:.1f}%**.")
                    
                    st.markdown("### 🤖 ChatGPT XAI Prompt")
                    st.markdown("Góp ý từ trợ lý ảo dựa trên Data-Grounded (Nếu có ChatGPT API ở đây, nó sẽ nhận vào đoạn lệnh sau, bạn có thể copy tự hỏi ChatGPT hiện tại):")
                    st.code(prompt, language="markdown")
                    
            except Exception as e:
                st.error(f"Xảy ra lỗi trong lúc phân tích: {e}")
            finally:
                os.unlink(video_path)
    
    elif analyze_btn and uploaded_file is None:
        st.warning("Xin vui lòng upload một video trước!")
