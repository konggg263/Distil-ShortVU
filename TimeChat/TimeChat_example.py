import torch
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from timechat.models import TimeChat
from timechat.processors import Blip2ImageEvalProcessor
from timechat.conversation.conversation_video import Chat, default_conversation

# --- 1. Cấu hình và Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

model_config = {
    "arch": "timechat",
    "model_type": "pretrain_vicuna",
    "llama_model": "lmsys/vicuna-7b-v1.5", 
    "ckpt": "timechat_7b.pth",
    "image_size": 224,
    "num_query_token": 32,
    "vit_model": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
    
    # SỬA QUAN TRỌNG 1: Chuyển sang fp32 để tránh lỗi MPS crash
    "vit_precision": "fp32", 
    
    "freeze_vit": True,
    "freeze_qformer": True,
    "low_resource": False, 
    "device_8bit": 0,
    "lora_r": 0,
    "q_former_model": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
}

print("Loading model...")
try:
    model = TimeChat.from_config(model_config)
    
    # Đảm bảo model nằm trên device đúng
    model = model.to(device)
    
    # Ép kiểu toàn bộ model sang float32 để an toàn trên Mac
    model = model.float() 
    
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. Hàm xử lý video (Đã fix lỗi Numpy/Tensor và chuyển sang FP32) ---
def load_video_custom(video_path, n_fms=32):
    """
    Hàm này thay thế logic load video bị lỗi của thư viện.
    Nó chuyển đổi an toàn từ Decord -> Numpy -> PIL -> Tensor -> FP32.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, n_fms, dtype=int)
    frames_batch = vr.get_batch(frame_indices)
    
    # Chuyển về Numpy bất kể đầu ra là gì
    if hasattr(frames_batch, 'asnumpy'):
        frames = frames_batch.asnumpy()
    else:
        frames = frames_batch.numpy()
    
    img_size = 224
    processor = Blip2ImageEvalProcessor(image_size=img_size)
    
    video_list = []
    for frame in frames:
        # Chuyển Numpy (H,W,C) -> PIL Image
        pil_image = Image.fromarray(frame)
        processed_frame = processor(pil_image) 
        video_list.append(processed_frame)
    
    # Stack: (C, T, H, W) -> Batch: (1, C, T, H, W)
    video_tensor = torch.stack(video_list, dim=1) 
    
    # SỬA QUAN TRỌNG 2: Chuyển sang float32 (thay vì float16)
    video_tensor = video_tensor.unsqueeze(0).to(device).to(torch.float32)
    
    return video_tensor

# --- 3. Wrapper Chat (Đã sửa lỗi) ---
def ask_timechat(model, video_path, prompt, n_fms=32):
    # Tạo conversation mới
    conv = default_conversation.copy()
    img_list = []
    
    # Khởi tạo Chat wrapper
    chat = Chat(model, Blip2ImageEvalProcessor(), device=device)
    
    # --- BƯỚC QUAN TRỌNG: BYPASS HÀM BỊ LỖI ---
    print(f"Processing video: {video_path}...")
    
    # 1. Load video bằng hàm custom (đã convert sang FP32)
    video_tensor = load_video_custom(video_path, n_fms=n_fms)
    
    # 2. Encode video thủ công để lấy embeddings
    with torch.no_grad():
        # Encode video
        image_emb, _ = model.encode_videoQformer_visual(video_tensor)
    
    img_list.append(image_emb)
    
    # 3. Cập nhật trạng thái hội thoại
    conv.append_message(conv.roles[0], "<Video><ImageHere></Video>")
    print("Video processed.")
    
    # --- Kết thúc Bypass ---

    # Gửi câu hỏi
    chat.ask(prompt, conv)
    
    # Nhận câu trả lời
    # max_new_tokens: độ dài câu trả lời
    answer_text, _ = chat.answer(conv, img_list, max_new_tokens=300, num_beams=1, top_p=0.9)
    return answer_text

# --- 4. Chạy thử nghiệm ---
video_path = "../0ada4a54b0f74f86561075531f71ffaa.mp4"

print("\n=== Dense Video Captioning ===")
prompt_dvc = "Describe the video in detail."
response_dvc = ask_timechat(model, video_path, prompt_dvc)
print(f"Response: {response_dvc}")

print("\n=== Temporal Grounding ===")
query = "When does the dog jump?"
response_tg = ask_timechat(model, video_path, query)
print(f"Response: {response_tg}")

print("\n=== Highlight Detection ===")
response_hd = ask_timechat(model, video_path, "What are the key highlights?")
print(f"Response: {response_hd}")
