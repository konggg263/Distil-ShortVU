import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor
from student_model import ViralStudentModel
import json
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model

# --- 1. Dataset Class ---
class SnapUGCDataset(Dataset):
    def __init__(self, data_file, tokenizer, processor):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.processor = processor # SigLIP processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load Image (Lấy frame đầu tiên hoặc giữa làm đại diện)
        # Trong thực tế nên dùng VideoReader lấy nhiều frame
        # Ở đây demo dùng PIL mở ảnh thumbnail
        image = Image.new('RGB', (224, 224)) # Placeholder
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        
        # Prepare Text Input & Label
        # Input: "Predict the engagement of this video."
        # Output: item['rationale']
        prompt = f"<|user|>\nAnalyze the engagement of this video.<|end|>\n<|assistant|>\n{item['rationale']}<|end|>"
        
        encodings = self.tokenizer(
            prompt, 
            max_length=512, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings.input_ids.squeeze(),
            "attention_mask": encodings.attention_mask.squeeze(),
            "pixel_values": pixel_values,
            "ecr_target": torch.tensor(item['ecr'], dtype=torch.float)
        }

# --- 2. Training Loop ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Init Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    
    model = ViralStudentModel()
    
    # Áp dụng LoRA cho LLM để train nhẹ hơn
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"], 
        lora_dropout=0.05, bias="none"
    )
    # Chỉ apply LoRA vào phần LLM của model
    model.llm = get_peft_model(model.llm, peft_config)
    
    model.to(device)
    model.train()
    
    # DataLoader
    dataset = SnapUGCDataset("./data/train_processed.json", tokenizer, processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print("Start Training...")
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            ecr_targets = batch["ecr_target"].to(device)
            
            # Forward pass
            # Lưu ý: Cần chỉnh sửa hàm forward trong student_model để tính loss text generation
            # Ở đây gọi tượng trưng
            outputs = model(input_ids, attention_mask, pixel_values, labels=input_ids, ecr_targets=ecr_targets)
            
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
