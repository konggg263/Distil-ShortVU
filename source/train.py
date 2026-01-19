import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPProcessor
from student_model import ViralStudentModel
from PIL import Image
import numpy as np

try:
    from peft import LoraConfig, get_peft_model
    _has_peft = True
except Exception:
    _has_peft = False


# --- 1. Dataset Class ---
class SnapUGCDataset(Dataset):
    def __init__(self, data_file, tokenizer, processor, max_text_len=128):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        # If JSON is map of id->obj, convert to list
        if isinstance(self.data, dict):
            # try to get list from values
            vals = list(self.data.values())
            if all(isinstance(x, dict) for x in vals):
                self.data = vals
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Try to use video_path midpoint frame if available
        pixel_values = None
        try:
            video_path = item.get('video_path') or item.get('video')
            if video_path and os.path.exists(video_path):
                from decord import VideoReader, cpu
                vr = VideoReader(video_path, ctx=cpu(0))
                frame = vr[len(vr)//2].asnumpy()
                image = Image.fromarray(frame)
            else:
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values.squeeze(0)

        # Text prompt / target
        rationale = item.get('rationale', '')
        prompt = f"Analyze the engagement of this video. {rationale}"

        enc = self.tokenizer(
            prompt,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'pixel_values': pixel_values,
            'ecr_target': torch.tensor(item.get('ecr', 0.0), dtype=torch.float)
        }


# --- 2. Training Loop ---
def train(batch_size=4, epochs=2, lr=2e-5, data_file='./data/train_processed.json', use_hf_llm=False, llm_name='gpt2', device_name=None):
    device = torch.device(device_name) if device_name is not None else torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Tokenizer and processor
    if use_hf_llm:
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Ensure tokenizer has a pad token (some models like GPT2 do not by default)
    if tokenizer.pad_token is None:
        # Use eos_token as pad token to allow padding operations
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # As a fallback, add a new pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # Build model with small vocab or HF LM integration
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.get_vocab())
    model = ViralStudentModel(vocab_size=vocab_size, d_model=512, use_hf_llm=use_hf_llm, llm_name=llm_name)

    # Apply LoRA only if peft is available and student_model exposes a HF model
    if use_hf_llm and _has_peft and hasattr(model, 'llm'):
        try:
            peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj'], lora_dropout=0.05, bias='none')
            model.llm = get_peft_model(model.llm, peft_config)
            print('Applied LoRA to model.llm')
        except Exception as e:
            print('PEFT/LoRA skipped:', e)

    model.to(device)
    model.train()

    dataset = SnapUGCDataset(data_file, tokenizer, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs('checkpoints', exist_ok=True)

    print('Start Training...')
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            ecr_targets = batch['ecr_target'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=input_ids, ecr_targets=ecr_targets)

            loss = outputs.get('loss', torch.tensor(0.0, device=device))
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        # Save checkpoint each epoch
        # ckpt_path = os.path.join('checkpoints', f'student_epoch{epoch+1}.pth')
        # torch.save(model.state_dict(), ckpt_path)
        # print('Saved checkpoint:', ckpt_path)
    # Save checkpoint
    ckpt_path = os.path.join('checkpoints', f'student_epoch{epochs}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print('Saved checkpoint:', ckpt_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/train_processed.json')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--use-hf-llm', action='store_true', dest='use_hf_llm', help='Use a HF causal LM inside the student model')
    parser.add_argument('--llm-name', default='gpt2')
    parser.add_argument('--device', default=None, help='Torch device string, e.g. cpu,cuda,mps')

    args = parser.parse_args()

    train(batch_size=args.batch, epochs=args.epochs, lr=args.lr, data_file=args.data, use_hf_llm=args.use_hf_llm, llm_name=args.llm_name, device_name=args.device)
