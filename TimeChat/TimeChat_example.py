# Cài đặt từ GitHub
# git clone https://github.com/RenShuhuai-Andy/TimeChat
# conda env create -f environment.yml

import torch
from timechat.models import TimeChat
from timechat.processors import Blip2ImageEvalProcessor
from timechat.conversation import Chat
import numpy as np
from decord import VideoReader, cpu

# 1. Load TimeChat model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeChat.from_pretrained(
    "ShuhuaiRen/TimeChat-7b",  # Hugging Face model ID
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# 2. Initialize processors
image_processor = Blip2ImageEvalProcessor()
chat = Chat(model, image_processor, device=device)

# 3. Load video với timestamp awareness
video_path = "examples/cooking_video.mp4"
vr = VideoReader(video_path, ctx=cpu(0))
fps = vr.get_avg_fps()
duration = len(vr) / fps

# Sample frames với timestamp binding
num_frames = 100  # TimeChat sử dụng nhiều frames hơn
frame_indices = np.linspace(0, len(vr)-1, num_frames, dtype=int)
frames = vr.get_batch(frame_indices).asnumpy()

# Tạo timestamp cho mỗi frame
timestamps = [i / fps for i in frame_indices]

# 4. Task 1: Dense Video Captioning
print("=== Dense Video Captioning ===")
prompt_dvc = "Generate dense captions for this video with timestamps."
response_dvc = chat.generate(
    frames=frames,
    timestamps=timestamps,
    prompt=prompt_dvc,
    task_type="dense_caption"
)
print(response_dvc)

# 5. Task 2: Temporal Grounding
print("\n=== Temporal Grounding ===")
query = "Find when the person adds salt to the pan"
response_tg = chat.generate(
    frames=frames,
    timestamps=timestamps,
    prompt=query,
    task_type="temporal_grounding"
)
print(f"Event occurs at: {response_tg}")

# 6. Task 3: Highlight Detection
print("\n=== Highlight Detection ===")
response_hd = chat.generate(
    frames=frames,
    timestamps=timestamps,
    prompt="Detect the most important moments in this video",
    task_type="highlight_detection"
)
print(f"Highlights: {response_hd}")
