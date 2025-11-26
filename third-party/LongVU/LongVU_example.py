# Cài đặt dependencies
# git clone https://github.com/Vision-CAIR/LongVU
# pip install torch numpy decord transformers

import os
# Allow MPS to fallback to CPU for unimplemented ops (temporary, slower).
# Must be set before importing torch so PyTorch picks it up.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

import longvu
import inspect
import importlib

# Diagnostics: print which files are being imported at runtime to ensure
# we're using the edited workspace modules rather than an installed package.
try:
    print('longvu package file:', longvu.__file__)
    md = importlib.import_module('longvu.mm_datautils')
    print('longvu.mm_datautils file:', md.__file__)
    try:
        print('\nprocess_images source (first 20 lines):')
        src = '\n'.join(inspect.getsource(md.process_images).splitlines()[:20])
        print(src)
    except Exception as e:
        print('Could not get source for process_images:', e)
    cq = importlib.import_module('longvu.language_model.cambrian_qwen')
    print('cambrian_qwen file:', cq.__file__)
    try:
        print('\nCambrianQwenModel methods present:', [m for m in dir(cq.CambrianQwenModel) if not m.startswith('__')][:40])
    except Exception as e:
        print('Could not introspect CambrianQwenModel:', e)
except Exception as e:
    print('Diagnostics import failed:', e)

# 1. Load model và tokenizer
from longvu.utils import get_torch_device

# Resolve preferred device: prefer MPS (on mac), then CUDA, then CPU.
device = get_torch_device()

tokenizer, model, image_processor, context_len = load_pretrained_model(
    "./checkpoints/LongVU_Qwen2_7B",  # Đường dẫn đến model checkpoint
    None,
    "cambrian_qwen",
    device=device,
)
model.eval()

# 2. Chuẩn bị video input
video_path = "/Users/macco/Downloads/khoaluanvjp/taolam/0ada4a54b0f74f86561075531f71ffaa.mp4"
qs = "Describe this video in detail"  # Câu hỏi về video

# 3. Đọc và xử lý video
vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
fps = float(vr.get_avg_fps())

# Sample frames: lấy 1 frame mỗi giây
frame_indices = np.array([i for i in range(0, len(vr), round(fps))])

video = []
for frame_index in frame_indices:
    img = vr[frame_index].asnumpy()
    video.append(img)

video = np.stack(video)
image_sizes = [video[0].shape[:2]]

# 4. Xử lý frames với image processor
video = process_images(video, image_processor, model.config)
video = [item.unsqueeze(0).to(device) for item in video]

# 5. Chuẩn bị prompt
qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
conv = conv_templates["qwen"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# 6. Tokenize input
input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)

# 7. Setup stopping criteria
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

# 8. Generate response
with torch.inference_mode():
    try:
        # Prepare multimodal embeddings and attention mask using the model
        # helper. Then run a small greedy loop in the example (keeps edits
        # local to this file) to avoid internal generate path issues on
        # some backends where attention shapes change during generation.
        (
            inputs_prepared,
            position_ids_prepared,
            attention_mask_prepared,
            _,
            inputs_embeds,
            _,
            vision_tower_aux_feature_list,
            vision_tower_aux_attention_masks_list,
            final_vision_feature_size,
            global_context_feature,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            video,
            image_aux_attention_masks_list=None,
            image_sizes=image_sizes,
        )

        # Ensure tensors are on the correct device/dtype
        inputs_embeds = inputs_embeds.to(device)
        if attention_mask_prepared is None:
            attention_mask_prepared = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                dtype=torch.long,
                device=device,
            )
        else:
            attention_mask_prepared = attention_mask_prepared.to(device=device)

        # Greedy generation loop (append argmax token each step)
        max_new_tokens = 128
        generated = []
        cur_embeds = inputs_embeds
        cur_attn = attention_mask_prepared
        bsz = cur_embeds.shape[0]

        for step in range(max_new_tokens):
            pos_ids = torch.arange(0, cur_embeds.size(1), device=device).unsqueeze(0)
            out = model(
                input_ids=None,
                inputs_embeds=cur_embeds,
                attention_mask=cur_attn,
                position_ids=pos_ids,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits if hasattr(out, 'logits') else out[0]
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(int(next_token[0].cpu().item()))

            next_emb = model.get_model().embed_tokens(next_token.unsqueeze(0))
            next_emb = next_emb.to(dtype=cur_embeds.dtype, device=device)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones((bsz, 1), dtype=torch.long, device=device)], dim=1)

            # --- SỬA LỖI TẠI ĐÂY ---
            gen_tensor = torch.tensor([generated], dtype=torch.long)
            prefix = input_ids.to('cpu')
            all_ids = torch.cat([prefix, gen_tensor], dim=1)
            
            # Tạo bản sao sạch để decode (thay thế token âm bằng 0)
            clean_ids = all_ids.clone()
            clean_ids[clean_ids < 0] = 0 
            
            text = tokenizer.batch_decode(clean_ids, skip_special_tokens=True)[0]
            if stop_str in text:
                break
            # -----------------------

        output_ids = torch.tensor([generated], dtype=torch.long)
        output_ids = torch.cat([input_ids.to('cpu'), output_ids], dim=1)
    except Exception as e:
        print('Generation failed:', e)
        print('input_ids shape:', getattr(input_ids, 'shape', None))
        print('inputs_embeds shape:', getattr(locals().get('inputs_embeds', None), 'shape', None))
        print('attention_mask prepared shape:', getattr(locals().get('attention_mask_prepared', None), 'shape', None))
        raise

# 9. Decode output
clean_output_ids = output_ids.clone()
clean_output_ids[clean_output_ids < 0] = 0 # Thay thế token ảnh (-200) bằng 0 để tránh crash
pred = tokenizer.batch_decode(clean_output_ids, skip_special_tokens=True)[0].strip()
print(f"Model Response: {pred}")
