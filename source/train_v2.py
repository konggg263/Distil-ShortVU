"""
train_v2.py - Training script for Student Model V2

Uses precomputed data from pipeline_v2:
- ImageBind embeddings (1024-dim)
- Aesthetic/Technical scores
- ECR targets
- Optional captions for LM distillation

Usage:
    # Train MLP (fastest)
    python source/train_v2.py --data data/train_processed_v2.json --model mlp --epochs 50
    
    # Train Transformer
    python source/train_v2.py --data data/train_processed_v2.json --model transformer --epochs 100
    
    # Train full model with LLM
    python source/train_v2.py --data data/train_processed_v2.json --model full --use-llm --epochs 50

Author: Pipeline V2 Training
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add source to path
sys.path.insert(0, os.path.dirname(__file__))

from student_model_v2 import StudentMLP, StudentTransformer, ViralStudentV2, create_student_model


# ==============================================
# DATASET
# ==============================================
class PipelineV2Dataset(Dataset):
    """
    Dataset for pipeline v2 output.
    
    Uses precomputed ImageBind embeddings for fast training.
    """
    
    def __init__(
        self, 
        json_file: str,
        use_captions: bool = False,
        tokenizer=None,
        max_caption_len: int = 128,
        normalize_scores: bool = True,
    ):
        print(f"Loading dataset from {json_file}...")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(raw_data, dict):
            data_list = list(raw_data.values())
        else:
            data_list = raw_data
        
        # Filter out entries without ImageBind embeddings
        self.data = []
        skipped = 0
        for item in data_list:
            if item.get('imagebind_emb') is not None:
                self.data.append(item)
            else:
                skipped += 1
        
        print(f"Loaded {len(self.data)} samples (skipped {skipped} without embeddings)")
        
        self.use_captions = use_captions
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len
        self.normalize_scores = normalize_scores
        
        # Compute normalization stats for scores
        if normalize_scores:
            ecrs = [d['ecr'] for d in self.data]
            aes = [d['aesthetic_score']['aesthetic'] for d in self.data]
            tech = [d['aesthetic_score']['technical'] for d in self.data]
            
            self.ecr_mean = np.mean(ecrs)
            self.ecr_std = np.std(ecrs) + 1e-6
            self.aes_mean = np.mean(aes)
            self.aes_std = np.std(aes) + 1e-6
            self.tech_mean = np.mean(tech)
            self.tech_std = np.std(tech) + 1e-6
            
            print(f"Score stats: ECR={self.ecr_mean:.4f}±{self.ecr_std:.4f}, "
                  f"Aes={self.aes_mean:.2f}±{self.aes_std:.2f}, "
                  f"Tech={self.tech_mean:.2f}±{self.tech_std:.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # ImageBind embedding
        emb = torch.tensor(item['imagebind_emb'], dtype=torch.float32)
        
        # ECR target (already 0-1 in most cases, but let's normalize)
        ecr = item['ecr']
        
        # Aesthetic/Technical scores (0-10 scale, normalize to 0-1)
        aes = item['aesthetic_score']['aesthetic'] / 10.0
        tech = item['aesthetic_score']['technical'] / 10.0
        
        result = {
            'imagebind_emb': emb,
            'ecr': torch.tensor(ecr, dtype=torch.float32),
            'aesthetic': torch.tensor(aes, dtype=torch.float32),
            'technical': torch.tensor(tech, dtype=torch.float32),
        }
        
        # Optional: tokenize caption
        if self.use_captions and self.tokenizer and item.get('caption'):
            caption = item['caption']
            encoded = self.tokenizer(
                caption,
                max_length=self.max_caption_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['input_ids'] = encoded['input_ids'].squeeze(0)
            result['attention_mask'] = encoded['attention_mask'].squeeze(0)
            # Labels for LM loss (same as input_ids)
            result['labels'] = encoded['input_ids'].squeeze(0).clone()
        
        return result


# ==============================================
# TRAINING LOOP
# ==============================================
def train_epoch(model, dataloader, optimizer, device, epoch, use_text=False):
    model.train()
    total_loss = 0
    total_ecr_loss = 0
    total_aes_loss = 0
    total_tech_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Move to device
        imagebind_emb = batch['imagebind_emb'].to(device)
        ecr_targets = batch['ecr'].to(device)
        aesthetic_targets = batch['aesthetic'].to(device)
        technical_targets = batch['technical'].to(device)
        
        # Optional text inputs
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        if input_ids is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        
        if use_text and input_ids is not None:
            outputs = model(
                imagebind_emb=imagebind_emb,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                ecr_targets=ecr_targets,
                aesthetic_targets=aesthetic_targets,
                technical_targets=technical_targets,
            )
        else:
            outputs = model(
                imagebind_emb=imagebind_emb,
                ecr_targets=ecr_targets,
                aesthetic_targets=aesthetic_targets,
                technical_targets=technical_targets,
            )
        
        loss = outputs['loss']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        losses = outputs['losses']
        if 'ecr_loss' in losses:
            total_ecr_loss += losses['ecr_loss'].item()
        if 'aesthetic_loss' in losses:
            total_aes_loss += losses['aesthetic_loss'].item()
        if 'technical_loss' in losses:
            total_tech_loss += losses['technical_loss'].item()
        
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ecr': f'{losses.get("ecr_loss", torch.tensor(0)).item():.4f}',
        })
    
    return {
        'loss': total_loss / num_batches,
        'ecr_loss': total_ecr_loss / num_batches,
        'aesthetic_loss': total_aes_loss / num_batches,
        'technical_loss': total_tech_loss / num_batches,
    }


def evaluate(model, dataloader, device, use_text=False):
    model.eval()
    total_loss = 0
    all_ecr_pred = []
    all_ecr_true = []
    all_aes_pred = []
    all_aes_true = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            imagebind_emb = batch['imagebind_emb'].to(device)
            ecr_targets = batch['ecr'].to(device)
            aesthetic_targets = batch['aesthetic'].to(device)
            technical_targets = batch['technical'].to(device)
            
            outputs = model(
                imagebind_emb=imagebind_emb,
                ecr_targets=ecr_targets,
                aesthetic_targets=aesthetic_targets,
                technical_targets=technical_targets,
            )
            
            total_loss += outputs['loss'].item()
            
            all_ecr_pred.extend(outputs['predicted_ecr'].cpu().numpy())
            all_ecr_true.extend(ecr_targets.cpu().numpy())
            
            if 'predicted_aesthetic' in outputs:
                all_aes_pred.extend(outputs['predicted_aesthetic'].cpu().numpy())
                all_aes_true.extend(aesthetic_targets.cpu().numpy())
            
            num_batches += 1
    
    # Compute metrics
    ecr_pred = np.array(all_ecr_pred)
    ecr_true = np.array(all_ecr_true)
    ecr_mse = np.mean((ecr_pred - ecr_true) ** 2)
    ecr_corr = np.corrcoef(ecr_pred, ecr_true)[0, 1] if len(ecr_pred) > 1 else 0
    
    metrics = {
        'loss': total_loss / num_batches,
        'ecr_mse': ecr_mse,
        'ecr_corr': ecr_corr,
    }
    
    if all_aes_pred:
        aes_pred = np.array(all_aes_pred)
        aes_true = np.array(all_aes_true)
        metrics['aes_mse'] = np.mean((aes_pred - aes_true) ** 2)
        metrics['aes_corr'] = np.corrcoef(aes_pred, aes_true)[0, 1] if len(aes_pred) > 1 else 0
    
    return metrics


# ==============================================
# MAIN
# ==============================================
def main():
    parser = argparse.ArgumentParser(description="Train Student Model V2")
    parser.add_argument('--data', type=str, required=True, help='Path to pipeline v2 JSON output')
    parser.add_argument('--val-data', type=str, default=None, help='Validation data (optional)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'transformer', 'full'])
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use-llm', action='store_true', help='Use HuggingFace LLM (for full model)')
    parser.add_argument('--llm-name', type=str, default='gpt2')
    parser.add_argument('--use-captions', action='store_true', help='Use captions for LM distillation')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')
    parser.add_argument('--save-dir', type=str, default='checkpoints_v2')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"Training Student Model V2")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    
    # Tokenizer (for caption-based models)
    tokenizer = None
    if args.use_captions or (args.model == 'full' and args.use_llm):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset
    train_dataset = PipelineV2Dataset(
        args.data,
        use_captions=args.use_captions,
        tokenizer=tokenizer,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multiprocessing well
        pin_memory=False,
    )
    
    val_loader = None
    if args.val_data:
        val_dataset = PipelineV2Dataset(args.val_data, use_captions=args.use_captions, tokenizer=tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Create model
    if args.model == 'mlp':
        model = StudentMLP(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    elif args.model == 'transformer':
        model = StudentTransformer(
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            use_text=args.use_captions,
        )
    else:  # full
        model = ViralStudentV2(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            use_hf_llm=args.use_llm,
            llm_name=args.llm_name,
        )
    
    model = model.to(device)
    
    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} (trainable: {num_trainable:,})")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    use_text = args.use_captions and tokenizer is not None
    
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, use_text)
        scheduler.step()
        
        # Evaluate
        if val_loader:
            val_metrics = evaluate(model, val_loader, device, use_text)
            print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_ecr_corr={val_metrics['ecr_corr']:.4f}")
        else:
            print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                  f"ecr_loss={train_metrics['ecr_loss']:.4f}, "
                  f"aes_loss={train_metrics['aesthetic_loss']:.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, f'student_v2_{args.model}_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        current_loss = val_metrics['loss'] if val_loader else train_metrics['loss']
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = os.path.join(args.save_dir, f'student_v2_{args.model}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_metrics': train_metrics,
                'args': vars(args),
            }, best_path)
            print(f"  New best model saved: {best_path}")
    
    print("\n" + "=" * 60)
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
