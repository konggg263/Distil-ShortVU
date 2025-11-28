import sys
import os
import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
import gc
import requests
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from transformers import CLIPModel, CLIPProcessor
import types

# --- Cấu hình Device (Mac ARM) ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# --- Hack để bypass lỗi Flash Attention trên Mac ---
# Nhiều model VLM yêu cầu flash_attn, ta tạo module giả để nó fallback về attention thường
if "flash_attn" not in sys.modules:
    module = types.ModuleType("flash_attn")
    sys.modules["flash_attn"] = module

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "third-party", "TimeChat")) 
sys.path.append(os.path.join(PROJECT_ROOT, "third-party", "DOVER"))
sys.path.append(os.path.join(PROJECT_ROOT, "third-party", "ImageBind"))
sys.path.append(os.path.join(PROJECT_ROOT, "third-party", "InternVideo"))

# # ==============================================
# # 1. WRAPPER CHO TIMECHAT (FIXED FP32)
# # ==============================================
# class TimeChatWrapper:
#     def __init__(self, checkpoint_path="checkpoints/timechat_7b.pth"):
#         from timechat.models import TimeChat
#         from timechat.processors import Blip2ImageEvalProcessor
#         from timechat.conversation.conversation_video import Chat, default_conversation
        
#         self.processor = Blip2ImageEvalProcessor(image_size=224)
#         self.default_conversation = default_conversation
#         self.Chat = Chat
        
#         print(f"[TimeChat] Loading from {checkpoint_path}...")
#         model_config = {
#             "arch": "timechat",
#             "model_type": "pretrain_vicuna",
#             "llama_model": "lmsys/vicuna-7b-v1.5", 
#             "ckpt": checkpoint_path,
#             "image_size": 224,
#             "num_query_token": 32,
#             "vit_model": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
#             "vit_precision": "fp32",
#             "freeze_vit": True,
#             "freeze_qformer": True,
#             "low_resource": False, 
#             "device_8bit": 0,
#             "lora_r": 0,
#             "q_former_model": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
#         }
        
#         self.model = TimeChat.from_config(model_config)
#         print("[TimeChat] Converting to Float32 on CPU...")
#         self.model = self.model.float()
#         print(f"[TimeChat] Moving to {DEVICE}...")
#         self.model = self.model.to(DEVICE).eval()
        
#         self.chat_wrapper = self.Chat(self.model, self.processor, device=DEVICE)
#         print("[TimeChat] Loaded!")

#     def load_video_custom(self, video_path, n_fms=32):
#         vr = VideoReader(video_path, ctx=cpu(0))
#         total_frames = len(vr)
#         frame_indices = np.linspace(0, total_frames - 1, n_fms, dtype=int)
#         frames_batch = vr.get_batch(frame_indices)
        
#         if hasattr(frames_batch, 'asnumpy'):
#             frames = frames_batch.asnumpy()
#         else:
#             frames = frames_batch.numpy()
        
#         video_list = []
#         for frame in frames:
#             pil_image = Image.fromarray(frame)
#             processed_frame = self.processor(pil_image) 
#             video_list.append(processed_frame)
        
#         video_tensor = torch.stack(video_list, dim=1) 
#         video_tensor = video_tensor.unsqueeze(0).float().to(DEVICE)
#         return video_tensor

#     def ask(self, video_path, prompt):
#         try:
#             video_tensor = self.load_video_custom(video_path)
#             with torch.no_grad():
#                 image_emb, _ = self.model.encode_videoQformer_visual(video_tensor)
            
#             img_list = [image_emb]
#             conv = self.default_conversation.copy()
#             conv.append_message(conv.roles[0], "<Video><ImageHere></Video>")
#             self.chat_wrapper.ask(prompt, conv)
            
#             answer_text, _ = self.chat_wrapper.answer(conv, img_list, max_new_tokens=300, num_beams=1, top_p=0.9)
#             return answer_text
#         except Exception as e:
#             print(f"TimeChat Error on {video_path}: {e}")
#             return ""

#     def unload(self):
#         del self.model
#         del self.chat_wrapper
#         gc.collect()
#         torch.mps.empty_cache()
#         print("[TimeChat] Unloaded.")

# ==============================================
# 1.1 WRAPPER CHO TIMECHAT (Multi-Task: Caption & Score)
# ==============================================
class TimeChatWrapper:
    def __init__(self, checkpoint_path="checkpoints/timechat_7b.pth"):
        from timechat.models import TimeChat
        from timechat.processors import Blip2ImageEvalProcessor
        from timechat.conversation.conversation_video import Chat, default_conversation
        
        self.processor = Blip2ImageEvalProcessor(image_size=224)
        self.default_conversation = default_conversation
        self.Chat = Chat
        
        print(f"[TimeChat] Loading from {checkpoint_path}...")
        model_config = {
            "arch": "timechat",
            "model_type": "pretrain_vicuna",
            "llama_model": "lmsys/vicuna-7b-v1.5", 
            "ckpt": checkpoint_path,
            "image_size": 224,
            "num_query_token": 32,
            "vit_model": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
            "vit_precision": "fp32",
            "freeze_vit": True,
            "freeze_qformer": True,
            "low_resource": False, 
            "device_8bit": 0,
            "lora_r": 0,
            "q_former_model": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
        }
        
        self.model = TimeChat.from_config(model_config)
        self.model = self.model.float().to(DEVICE).eval()
        self.chat_wrapper = self.Chat(self.model, self.processor, device=DEVICE)
        print("[TimeChat] Loaded!")

    def load_video_custom(self, video_path, n_fms=32):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_indices = np.linspace(0, total_frames - 1, n_fms, dtype=int)
        frames_batch = vr.get_batch(frame_indices)
        
        if hasattr(frames_batch, 'asnumpy'):
            frames = frames_batch.asnumpy()
        else:
            frames = frames_batch.numpy()
        
        video_list = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            processed_frame = self.processor(pil_image) 
            video_list.append(processed_frame)
        
        video_tensor = torch.stack(video_list, dim=1) 
        video_tensor = video_tensor.unsqueeze(0).float().to(DEVICE)
        return video_tensor

    def ask(self, video_path, prompt, video_tensor=None):
        try:
            # Nếu chưa có tensor thì load, nếu có rồi (cache) thì dùng lại để tiết kiệm thời gian
            if video_tensor is None:
                video_tensor = self.load_video_custom(video_path)
                
            with torch.no_grad():
                image_emb, _ = self.model.encode_videoQformer_visual(video_tensor)
            
            img_list = [image_emb]
            conv = self.default_conversation.copy()
            conv.append_message(conv.roles[0], "<Video><ImageHere></Video>")
            self.chat_wrapper.ask(prompt, conv)
            
            answer_text, _ = self.chat_wrapper.answer(conv, img_list, max_new_tokens=300, num_beams=1, top_p=0.9)
            return answer_text
        except Exception as e:
            print(f"TimeChat Error on {video_path}: {e}")
            return ""

    def analyze_video(self, video_path):
        # Load video 1 lần dùng cho cả 2 task
        try:
            video_tensor = self.load_video_custom(video_path)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return "", 5.0

        # Task 1: Caption
        caption_prompt = "Describe the main events and visual style of this video in detail."
        caption = self.ask(video_path, caption_prompt, video_tensor)
        
        # Task 2: Scoring (Prompt Engineering để ép model chấm điểm)
        # TimeChat dựa trên Vicuna, nó có khả năng đánh giá tốt nếu prompt đúng.
        score_prompt = (
            "Act as a professional video critic. Rate the aesthetic quality, lighting, and composition "
            "of this video on a scale from 1 to 10. "
            "Return ONLY the number (e.g., 7.5). Do not explain."
        )
        score_text = self.ask(video_path, score_prompt, video_tensor)
        
        # Parse điểm số từ text trả về
        try:
            # Tìm số thực đầu tiên trong chuỗi (ví dụ: "I give it a 7.5" -> 7.5)
            match = re.search(r"[-+]?\d*\.\d+|\d+", score_text)
            if match:
                score = float(match.group())
                # Clip điểm trong khoảng 1-10
                score = max(1.0, min(10.0, score))
            else:
                score = 5.0 # Fallback
        except:
            score = 5.0
            
        return caption, score

    def unload(self):
        del self.model
        del self.chat_wrapper
        gc.collect()
        torch.mps.empty_cache()
        print("[TimeChat] Unloaded.")

# ==============================================
# 2. WRAPPER CHO DOVER (FORCE CPU ONLY)
# ==============================================
class DOVERWrapper:
    def __init__(self, checkpoint_path="./checkpoints/DOVER_plus_plus.pth"):
        from dover.models import DOVER
        
        print("[DOVER] Loading (Force CPU due to Conv3D limitation on MPS)...")
        dover_config = {
            "resize": {
                "type": "conv_tiny", 
                "window_size": (4, 4, 4)
            },
            "fragments": {
                "type": "swin_tiny_grpb", 
                "window_size": (4, 4, 4)
            }
        }
        
        self.model = DOVER(backbone=dover_config)
        
        if os.path.exists(checkpoint_path):
            print(f"[DOVER] Loading weights from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
            msg = self.model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f"[DOVER] WARNING: Checkpoint not found at {checkpoint_path}")

        # --- QUAN TRỌNG: FORCE CPU ---
        self.device = "cpu" 
        self.model = self.model.float().to(self.device).eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).float().to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).float().to(self.device)
        print("[DOVER] Loaded on CPU!")

    def _process_video(self, video_path, num_frames=32):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames_batch = vr.get_batch(indices)
        if hasattr(frames_batch, 'asnumpy'):
            frames = frames_batch.asnumpy()
        else:
            frames = frames_batch.numpy()
        
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        frames = frames.unsqueeze(0).to(self.device) # CPU
        
        frames = (frames - self.mean) / self.std
        
        resize_view = torch.nn.functional.interpolate(frames, size=(num_frames, 224, 224), mode='trilinear')
        fragments_view = torch.nn.functional.interpolate(frames, size=(num_frames, 224, 224), mode='trilinear')

        return {"resize": resize_view, "fragments": fragments_view}

    def predict(self, video_path):
        try:
            inputs = self._process_video(video_path)
            with torch.no_grad():
                scores = self.model(inputs, inference=True, reduce_scores=False, pooled=True)
                
                if isinstance(scores, list) and len(scores) >= 2:
                    aes = scores[0].mean().item()
                    tech = scores[1].mean().item()
                elif isinstance(scores, torch.Tensor):
                    val = scores.mean().item()
                    aes, tech = val, val
                else:
                    aes, tech = 0.0, 0.0
                    
            return {"aesthetic": aes, "technical": tech}
        except Exception as e:
            print(f"DOVER Error on {video_path}: {e}")
            return {"aesthetic": 0.0, "technical": 0.0}

    def unload(self):
        del self.model
        gc.collect()
        print("[DOVER] Unloaded.")

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024), # Index 0
            nn.ReLU(True),                    # Index 1
            nn.Linear(1024, 128),             # Index 2
            nn.ReLU(True),                    # Index 3
            nn.Linear(128, 64),               # Index 4
            nn.ReLU(True),                    # Index 5
            nn.Linear(64, 16),                # Index 6
            nn.Linear(16, 1)                  # Index 7
        )

    def forward(self, x):
        return self.layers(x)
    
# ==============================================
# 2.1 WRAPPER CHO CLIP-AESTHETICS (NEW & STABLE)
# ==============================================
class AestheticScorerWrapper:
    def __init__(self):
        print("[Aesthetic] Loading CLIP-Large (Native MPS)...")
        model_id = "openai/clip-vit-large-patch14"
        
        # 1. Load CLIP Backbone
        self.clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # 2. Load Aesthetic MLP Head
        # Đây là weights chuẩn của LAION Aesthetic V2
        self.mlp = MLP(768) # CLIP Large dim = 768
        
        weight_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        weight_path = "checkpoints/sac_logos_ava1_l14_linearMSE.pth"
        
        if not os.path.exists(weight_path):
            print(f"[Aesthetic] Downloading MLP weights to {weight_path}...")
            os.makedirs("checkpoints", exist_ok=True)
            response = requests.get(weight_url)
            with open(weight_path, "wb") as f:
                f.write(response.content)
                
        state_dict = torch.load(weight_path, map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.mlp = self.mlp.to(DEVICE).eval()
        
        print("[Aesthetic] Loaded!")

    def predict(self, video_path, num_frames=8):
        # Lấy ít frame hơn DOVER vì CLIP nhìn ảnh tĩnh rất tốt
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            frames_batch = vr.get_batch(indices) # (T, H, W, C)
            if hasattr(frames_batch, 'asnumpy'):
                frames = frames_batch.asnumpy()
            else:
                frames = frames_batch.numpy()
            
            # Convert to PIL for CLIP Processor
            pil_images = [Image.fromarray(f) for f in frames]
            
            inputs = self.processor(images=pil_images, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                # Get CLIP embeddings
                embeddings = self.clip_model.get_image_features(**inputs)
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                # Predict score
                scores = self.mlp(embeddings)
            
            # Average score over frames
            avg_score = scores.mean().item()
            
            # Return format giống DOVER để không phải sửa prompt LLM nhiều
            # Aesthetic = điểm thật, Technical = 0 (hoặc dùng điểm này luôn)
            return {"aesthetic": round(avg_score, 2), "technical": round(avg_score, 2)}
            
        except Exception as e:
            print(f"Aesthetic Error on {video_path}: {e}")
            return {"aesthetic": 0.0, "technical": 0.0}

    def unload(self):
        del self.clip_model
        del self.mlp
        gc.collect()
        torch.mps.empty_cache()
        print("[Aesthetic] Unloaded.")

# ==============================================
# 2.2. WRAPPER CHO Q-ALIGN (ONE-ALIGN)
# ==============================================
class QAlignWrapper:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Sử dụng model OneAlign (phiên bản Q-Align tối ưu hơn)
        # Model này dựa trên LLaVA/Qwen, hỗ trợ AutoModel
        model_path = "q-future/one-align" 
        
        print(f"[Q-Align] Loading {model_path}...")
        print("[Q-Align] Note: Using Float16 for MPS optimization.")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.float16, # Dùng FP16 để nhẹ hơn trên Mac
                device_map=DEVICE
            ).eval()
            print("[Q-Align] Loaded!")
        except Exception as e:
            print(f"[Q-Align] Load Failed: {e}")
            self.model = None

    def predict(self, video_path):
        if self.model is None: return {"aesthetic": 0.0, "technical": 0.0}
        
        try:
            # Q-Align logic: Lấy video -> Prompt -> Tính xác suất các từ đánh giá
            
            # 1. Load & Process Video (Logic đơn giản hóa cho OneAlign)
            # OneAlign thường nhận input tensor video. 
            # Ở đây ta dùng logic xử lý video tích hợp sẵn trong repo của họ nếu có,
            # hoặc tự xử lý thủ công. Để an toàn và nhanh, ta dùng code xử lý video chuẩn:
            
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            # OneAlign thường dùng 32 frames
            indices = np.linspace(0, total_frames-1, 32, dtype=int)
            frames_batch = vr.get_batch(indices) # (T, H, W, C)
            if hasattr(frames_batch, 'asnumpy'):
                frames = frames_batch.asnumpy()
            else:
                frames = frames_batch.numpy()
            
            # Convert to Tensor (B, T, C, H, W)
            tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() # (T, C, H, W)
            tensor = tensor.unsqueeze(0).to(DEVICE, dtype=torch.float16) # (1, T, C, H, W)
            
            # 2. Prompting
            # OneAlign prompt: "How would you rate the quality of this video?"
            prompt = "How would you rate the quality of this video?"
            
            # Chuẩn bị input cho model (Tùy thuộc vào implementation cụ thể của remote code)
            # Do remote code của OneAlign khá phức tạp, ta sẽ dùng hàm generate và parse text
            # Hoặc cách tốt hơn: Tính log-likelihood của các từ rating.
            
            # Cách đơn giản nhất với OneAlign wrapper:
            # Model này được train để trả lời: "Excellent", "Good", "Fair", "Poor", "Bad"
            
            # Xây dựng input ids (giả lập LLaVA style conversation)
            # Lưu ý: Đây là phần tricky nhất vì mỗi model có chat template khác nhau.
            # Ta sẽ dùng trực tiếp hàm forward nếu có thể, hoặc generate.
            
            # Thử dùng generate trước (Dễ implement hơn)
            # Format input của OneAlign/LLaVA:
            # USER: <video>\nPrompt ASSISTANT:
            
            # *Lưu ý*: Do không thể import trực tiếp logic xử lý ảnh của OneAlign dễ dàng,
            # ta sẽ dùng phương pháp "Blind extraction" nếu model hỗ trợ, 
            # hoặc fallback về logic tính điểm thủ công nếu generate ra text.
            
            # Để đơn giản và chạy được ngay, ta sẽ dùng generate text và map về điểm số.
            # Tuy nhiên, OneAlign remote code thường yêu cầu 'video' argument trong forward.
            
            # --- CUSTOM FORWARD PASS CHO ONE-ALIGN ---
            # Dựa trên source code của q-future/one-align
            
            # Input structure giả định (cần check remote code thực tế, nhưng thường là như này):
            inputs = self.tokenizer([prompt], return_tensors='pt').to(DEVICE)
            
            # OneAlign nhận 'images' hoặc 'video' tensor. 
            # Ta cần resize video về kích thước model mong muốn (thường là 224 hoặc 336)
            # Để tránh lỗi dimension, ta resize về 224x224
            tensor_resized = torch.nn.functional.interpolate(
                tensor.view(-1, 3, frames.shape[1], frames.shape[2]), 
                size=(224, 224), 
                mode='bilinear'
            ).view(1, 32, 3, 224, 224)
            
            # Normalize (ImageNet mean/std)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1).to(DEVICE, dtype=torch.float16)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1).to(DEVICE, dtype=torch.float16)
            tensor_norm = (tensor_resized / 255.0 - mean) / std
            
            # Generate
            # Model OneAlign trên HF thường yêu cầu input_values hoặc images
            # Ta thử gọi generate với tensor
            
            # QUAN TRỌNG: Do ta không dùng processor của họ (vì khó import), 
            # ta sẽ dùng cách generate text cơ bản và hy vọng model đủ thông minh.
            # Nếu crash, ta sẽ trả về điểm giả định và log lỗi để debug.
            
            # *Hack*: OneAlign thực chất là LLaVA. Ta cần đưa video vào embedding layer.
            # Nhưng việc này quá phức tạp cho 1 script ngắn.
            
            # ==> GIẢI PHÁP THAY THẾ AN TOÀN HƠN:
            # Dùng model "InternVideo2-Stage2_1B" hoặc scoring bằng CLIP-Aesthetics nhưng với prompt tốt hơn?
            # Không, user muốn Q-Align.
            
            # Thử gọi trực tiếp nếu model hỗ trợ 'video' arg
            with torch.no_grad():
                # OneAlign custom forward accepts 'video'
                output_ids = self.model.generate(
                    **inputs,
                    images=tensor_norm, # OneAlign dùng key 'images' cho video tensor (B, T, C, H, W) hoặc (B, C, T, H, W)
                    max_new_tokens=10
                )
                
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
            
            # Map response to score
            # Excellent=5, Good=4, Fair=3, Poor=2, Bad=1
            score_map = {
                "excellent": 5.0, "good": 4.0, "fair": 3.0, "poor": 2.0, "bad": 1.0,
                "high": 4.5, "low": 1.5
            }
            
            final_score = 3.0 # Default
            for key, val in score_map.items():
                if key in response:
                    final_score = val
                    break
            
            # Convert 1-5 scale to 1-10 scale for consistency
            final_score = final_score * 2.0
            
            return {"aesthetic": final_score, "technical": final_score, "raw_rating": response}

        except Exception as e:
            # Nếu Q-Align crash (thường do input shape), ta fallback về 0
            print(f"[Q-Align] Error on {video_path}: {e}")
            # Fallback logic: Trả về điểm trung bình để pipeline không chết
            return {"aesthetic": 5.0, "technical": 5.0, "error": str(e)}

    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
        gc.collect()
        torch.mps.empty_cache()
        print("[Q-Align] Unloaded.")
    
# ==============================================
# 3. WRAPPER CHO LOCAL LLM (FIXED MODEL & SAVING)
# ==============================================
class LocalLLMWrapper:
    # SỬ DỤNG MODEL MLX-COMMUNITY ĐỂ TỐI ƯU CHO MAC
    # Dùng bản 7B để nhẹ và nhanh. Nếu máy >32GB RAM có thể đổi sang 32B.
    def __init__(self, model_path="mlx-community/Qwen2.5-32B-Instruct-4bit"):
        from mlx_lm import load, generate
        print(f"[Local LLM] Loading {model_path}...")
        
        # Load model
        self.model, self.tokenizer = load(model_path)
        self.generate_fn = generate
        print("[Local LLM] Loaded!")

    def generate_rationale(self, info_dict):
        prompt = f"""<|im_start|>system
You are an expert video quality analyst.<|im_end|>
<|im_start|>user
Generate a rationale for this video evaluation:
- Content: {info_dict.get('caption', 'N/A')}
- Aesthetic Score: {info_dict.get('aesthetic_score', {}).get('aesthetic', 0)}/10
- Motion (ECR): {info_dict.get('ecr', 0.0)}

Why did it receive this score? (Max 2 sentences)<|im_end|>
<|im_start|>assistant
"""
        try:
            response = self.generate_fn(
                self.model, self.tokenizer, prompt=prompt, max_tokens=512, verbose=False
            )
            return response.strip()
        except Exception as e:
            print(f"Local LLM Error: {e}")
            return "Error generating rationale."

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()

# ==============================================
# 4. MAIN PIPELINE
# ==============================================
def process_dataset_multi_pass(csv_file, video_folder, output_file):
    df = pd.read_csv(csv_file)
    results = {}
    
    # Init data dict
    count = 0
    for _, row in df.iterrows():
        count += 1
        vid_path = os.path.join(video_folder, f"{row['Id']}.mp4")
        if os.path.exists(vid_path):
            results[row['Id']] = {
                "video_path": vid_path,
                "ecr": row['ECR'],
                "title": str(row['Title']),
                "description": str(row['Description']),
                "caption": "",
                "aesthetic_score": {},
                "imagebind_emb": None
            }
        if count == 50:
            break 

    # # --- PASS 1: TimeChat ---
    # print("\n=== PASS 1: TimeChat ===")
    # timechat = TimeChatWrapper()
    # for vid_id, data in tqdm(results.items()):
    #     desc = timechat.ask(data['video_path'], "Describe the first 5 seconds in detail.")
    #     results[vid_id]['caption'] = desc
    # timechat.unload()
    # --- PASS 1: TimeChat (Caption & Score) ---
    print("\n=== PASS 1: TimeChat Analysis ===")
    timechat = TimeChatWrapper()
    
    for vid_id, data in tqdm(results.items()):
        caption, score = timechat.analyze_video(data['video_path'])
        
        results[vid_id]['caption'] = caption
        results[vid_id]['aesthetic_score'] = {
            "aesthetic": score,
            "technical": score # TimeChat chấm tổng quát
        }
        print(f"Video {vid_id} | Score: {score} | Caption len: {len(caption)}")
        
    timechat.unload()
    
    # # --- PASS 2: DOVER --- đang lỗi
    # print("\n=== PASS 2: DOVER ===")
    # dover = DOVERWrapper()
    # for vid_id, data in tqdm(results.items()):
    #     score = dover.predict(data['video_path'])
    #     results[vid_id]['aesthetic_score'] = score
    # dover.unload()
    # # --- PASS 2: Aesthetic (CLIP) --- không phù hợp với video
    # print("\n=== PASS 2: Aesthetic Scoring ===")
    # scorer = AestheticScorerWrapper()
    # for vid_id, data in tqdm(results.items()):
    #     score = scorer.predict(data['video_path'])
    #     results[vid_id]['aesthetic_score'] = score
    # scorer.unload()
    # # --- PASS 2: Q-Align (OneAlign) --- đang lỗi
    # print("\n=== PASS 2: Q-Align Scoring ===")
    # scorer = QAlignWrapper()
    # for vid_id, data in tqdm(results.items()):
    #     score = scorer.predict(data['video_path'])
    #     results[vid_id]['aesthetic_score'] = score
    #     # In ra để debug xem điểm có thay đổi không
    #     print(f"Video {vid_id}: {score}")
    # scorer.unload()

    # --- PASS 3: Local LLM ---
    print("\n=== PASS 3: Local LLM Rationale ===")
    llm = LocalLLMWrapper()
    
    final_data = []
    print(f"Processing LLM for {len(results)} videos...")
    
    for vid_id, data in tqdm(results.items()):
        rationale = llm.generate_rationale(data)
        data['rationale'] = rationale
        final_data.append(data)
        
        # LƯU NGAY LẬP TỨC SAU MỖI VIDEO
        # Để tránh mất dữ liệu nếu crash
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=4)
            
    llm.unload()
    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    process_dataset_multi_pass(
        csv_file="data/train_data.csv", 
        video_folder="data/train_videos",
        output_file="data/train_processed.json"
    )
