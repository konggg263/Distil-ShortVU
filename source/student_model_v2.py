"""
student_model_v2.py - Student Model for Pipeline V2

Key improvements:
1. Uses precomputed ImageBind embeddings (1024-dim) - no need to recompute
2. Multi-task learning: ECR + Aesthetic + Technical score prediction
3. Optional caption generation via distillation
4. Multiple architectures: MLP-only, Transformer, or full LLM

Input features from pipeline_v2:
- imagebind_emb: 1024-dim multimodal embedding (vision + audio)
- caption: text description from BLIP
- aesthetic_score: {aesthetic: float, technical: float}
- ecr: engagement rate (target)

Author: Pipeline V2 Student Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


# ==============================================
# 1. LIGHTWEIGHT MLP MODEL (Fastest)
# ==============================================
class StudentMLP(nn.Module):
    """
    Simple MLP that takes precomputed ImageBind embeddings
    and predicts ECR + aesthetic + technical scores.
    
    Very fast inference - just 3 linear layers!
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # ImageBind embedding dimension
        hidden_dim: int = 512,
        dropout: float = 0.1,
        predict_aesthetic: bool = True,
        predict_technical: bool = True,
    ):
        super().__init__()
        
        self.predict_aesthetic = predict_aesthetic
        self.predict_technical = predict_technical
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # ECR is 0-1
        )
        
        if predict_aesthetic:
            self.aesthetic_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        if predict_technical:
            self.technical_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
    
    def forward(
        self,
        imagebind_emb: torch.Tensor,  # (B, 1024)
        ecr_targets: Optional[torch.Tensor] = None,
        aesthetic_targets: Optional[torch.Tensor] = None,
        technical_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        
        # Encode
        hidden = self.encoder(imagebind_emb)
        
        # Predictions
        predicted_ecr = self.ecr_head(hidden).squeeze(-1)
        
        outputs = {
            'predicted_ecr': predicted_ecr,
            'hidden': hidden,
        }
        
        if self.predict_aesthetic:
            outputs['predicted_aesthetic'] = self.aesthetic_head(hidden).squeeze(-1)
        
        if self.predict_technical:
            outputs['predicted_technical'] = self.technical_head(hidden).squeeze(-1)
        
        # Compute losses if targets provided
        loss = 0.0
        losses = {}
        
        if ecr_targets is not None:
            ecr_loss = F.mse_loss(predicted_ecr, ecr_targets)
            losses['ecr_loss'] = ecr_loss
            loss = loss + ecr_loss
        
        if aesthetic_targets is not None and self.predict_aesthetic:
            aes_loss = F.mse_loss(outputs['predicted_aesthetic'], aesthetic_targets)
            losses['aesthetic_loss'] = aes_loss
            loss = loss + 0.5 * aes_loss  # Lower weight for auxiliary task
        
        if technical_targets is not None and self.predict_technical:
            tech_loss = F.mse_loss(outputs['predicted_technical'], technical_targets)
            losses['technical_loss'] = tech_loss
            loss = loss + 0.5 * tech_loss
        
        outputs['loss'] = loss
        outputs['losses'] = losses
        
        return outputs


# ==============================================
# 2. TRANSFORMER MODEL (Balanced)
# ==============================================
class StudentTransformer(nn.Module):
    """
    Transformer-based student model.
    
    Can operate in two modes:
    1. Embedding-only: Uses precomputed ImageBind embeddings
    2. Full: Processes raw video frames with vision encoder
    
    Supports optional text input (caption) for multimodal learning.
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        vocab_size: int = 50257,  # GPT-2 vocab
        max_text_len: int = 128,
        use_text: bool = False,
    ):
        super().__init__()
        
        self.use_text = use_text
        self.hidden_dim = hidden_dim
        
        # Project ImageBind embedding to hidden_dim
        self.emb_projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Text embedding (optional)
        if use_text:
            self.token_emb = nn.Embedding(vocab_size, hidden_dim)
            self.pos_emb = nn.Embedding(max_text_len, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Task heads
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.aesthetic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.technical_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Optional: LM head for caption generation
        if use_text:
            self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        imagebind_emb: torch.Tensor,  # (B, 1024)
        input_ids: Optional[torch.Tensor] = None,  # (B, seq_len)
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # For LM loss
        ecr_targets: Optional[torch.Tensor] = None,
        aesthetic_targets: Optional[torch.Tensor] = None,
        technical_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        
        device = imagebind_emb.device
        batch_size = imagebind_emb.size(0)
        
        # Project embedding to sequence token
        emb_token = self.emb_projector(imagebind_emb).unsqueeze(1)  # (B, 1, hidden)
        
        # Build sequence
        if self.use_text and input_ids is not None:
            seq_len = input_ids.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            text_emb = self.token_emb(input_ids) + self.pos_emb(positions)
            
            # Concat: [emb_token, text_tokens]
            sequence = torch.cat([emb_token, text_emb], dim=1)
            
            # Build attention mask
            emb_mask = torch.ones((batch_size, 1), device=device)
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), device=device)
            full_mask = torch.cat([emb_mask, attention_mask], dim=1)
            
            # Convert to transformer format (1 = attend, 0 = mask)
            src_key_padding_mask = (full_mask == 0)
        else:
            sequence = emb_token
            src_key_padding_mask = None
            seq_len = 0
        
        # Transformer forward
        hidden = self.transformer(sequence, src_key_padding_mask=src_key_padding_mask)
        
        # Pool the embedding token for regression tasks
        pooled = hidden[:, 0, :]  # First token is the ImageBind embedding
        
        # Predictions
        predicted_ecr = self.ecr_head(pooled).squeeze(-1)
        predicted_aesthetic = self.aesthetic_head(pooled).squeeze(-1)
        predicted_technical = self.technical_head(pooled).squeeze(-1)
        
        outputs = {
            'predicted_ecr': predicted_ecr,
            'predicted_aesthetic': predicted_aesthetic,
            'predicted_technical': predicted_technical,
            'hidden': hidden,
            'pooled': pooled,
        }
        
        # LM logits (for caption distillation)
        if self.use_text and input_ids is not None:
            text_hidden = hidden[:, 1:, :]  # Skip embedding token
            logits = self.lm_head(text_hidden)
            outputs['logits'] = logits
        
        # Compute losses
        loss = 0.0
        losses = {}
        
        if ecr_targets is not None:
            ecr_loss = F.mse_loss(predicted_ecr, ecr_targets)
            losses['ecr_loss'] = ecr_loss
            loss = loss + ecr_loss
        
        if aesthetic_targets is not None:
            aes_loss = F.mse_loss(predicted_aesthetic, aesthetic_targets)
            losses['aesthetic_loss'] = aes_loss
            loss = loss + 0.3 * aes_loss
        
        if technical_targets is not None:
            tech_loss = F.mse_loss(predicted_technical, technical_targets)
            losses['technical_loss'] = tech_loss
            loss = loss + 0.3 * tech_loss
        
        if labels is not None and self.use_text and 'logits' in outputs:
            # Shift for autoregressive loss
            shift_logits = outputs['logits'][:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            losses['lm_loss'] = lm_loss
            loss = loss + 0.5 * lm_loss
        
        outputs['loss'] = loss
        outputs['losses'] = losses
        
        return outputs


# ==============================================
# 3. FULL STUDENT MODEL (Most Capable)
# ==============================================
class ViralStudentV2(nn.Module):
    """
    Full student model that can:
    1. Use precomputed ImageBind embeddings (fast inference)
    2. OR compute embeddings from raw video (for end-to-end training)
    3. Optionally use a HuggingFace LLM for caption generation
    
    Multi-task outputs:
    - ECR prediction (main task)
    - Aesthetic score prediction
    - Technical score prediction
    - Caption generation (optional)
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        dropout: float = 0.1,
        use_vision_encoder: bool = False,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        use_hf_llm: bool = False,
        llm_name: str = "gpt2",
    ):
        super().__init__()
        
        self.use_vision_encoder = use_vision_encoder
        self.use_hf_llm = use_hf_llm
        self.hidden_dim = hidden_dim
        
        # Optional: CLIP vision encoder for raw frames
        if use_vision_encoder:
            from transformers import CLIPModel
            print(f"Loading CLIP vision encoder: {vision_model_name}")
            clip = CLIPModel.from_pretrained(vision_model_name)
            self.vision_encoder = clip.vision_model
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            self.vision_dim = self.vision_encoder.config.hidden_size
            self.vision_projector = nn.Linear(self.vision_dim, embedding_dim)
        
        # Embedding projector
        self.emb_projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Core model: Transformer or HF LLM
        if use_hf_llm:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading HF LLM: {llm_name}")
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.hidden_dim = self.llm.config.hidden_size
            # Re-create projector with correct dim
            self.emb_projector = nn.Sequential(
                nn.Linear(embedding_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            # Lightweight transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Multi-task heads
        self.ecr_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.aesthetic_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 1),
        )
        
        self.technical_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 1),
        )
    
    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode raw video frames to embedding"""
        if not self.use_vision_encoder:
            raise ValueError("Vision encoder not loaded. Use imagebind_emb instead.")
        
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values)
        
        # Pool over spatial tokens
        pooled = vision_out.last_hidden_state.mean(dim=1)
        return self.vision_projector(pooled)
    
    def forward(
        self,
        imagebind_emb: Optional[torch.Tensor] = None,  # (B, 1024)
        pixel_values: Optional[torch.Tensor] = None,    # (B, C, H, W)
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ecr_targets: Optional[torch.Tensor] = None,
        aesthetic_targets: Optional[torch.Tensor] = None,
        technical_targets: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'ecr': 1.0,
                'aesthetic': 0.3,
                'technical': 0.3,
                'lm': 0.5,
            }
        
        # Get embedding
        if imagebind_emb is not None:
            emb = imagebind_emb
        elif pixel_values is not None:
            emb = self.encode_video(pixel_values)
        else:
            raise ValueError("Either imagebind_emb or pixel_values required")
        
        device = emb.device
        batch_size = emb.size(0)
        
        # Project to hidden dim
        emb_hidden = self.emb_projector(emb).unsqueeze(1)  # (B, 1, hidden)
        
        if self.use_hf_llm and input_ids is not None:
            # HF LLM path
            embed_layer = self.llm.get_input_embeddings()
            text_emb = embed_layer(input_ids)
            
            # Concat embedding with text
            sequence = torch.cat([emb_hidden, text_emb], dim=1)
            
            # Build masks
            emb_mask = torch.ones((batch_size, 1), device=device, dtype=torch.long)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            full_mask = torch.cat([emb_mask, attention_mask], dim=1)
            
            # Build labels (mask embedding position)
            labels_full = None
            if labels is not None:
                labels_full = torch.full((batch_size, 1 + labels.size(1)), -100, device=device, dtype=torch.long)
                labels_full[:, 1:] = labels
            
            # Forward LLM
            outputs = self.llm(
                inputs_embeds=sequence,
                attention_mask=full_mask,
                labels=labels_full,
                output_hidden_states=True,
            )
            
            hidden = outputs.hidden_states[-1]
            pooled = hidden[:, 0, :]  # Embedding token
            logits = outputs.logits[:, 1:, :] if outputs.logits is not None else None
            lm_loss = outputs.loss
        else:
            # Lightweight transformer path
            if hasattr(self, 'transformer'):
                hidden = self.transformer(emb_hidden)
            else:
                hidden = emb_hidden
            pooled = hidden[:, 0, :]
            logits = None
            lm_loss = None
        
        # Multi-task predictions
        predicted_ecr = self.ecr_head(pooled).squeeze(-1)
        predicted_aesthetic = self.aesthetic_head(pooled).squeeze(-1)
        predicted_technical = self.technical_head(pooled).squeeze(-1)
        
        outputs = {
            'predicted_ecr': predicted_ecr,
            'predicted_aesthetic': predicted_aesthetic,
            'predicted_technical': predicted_technical,
            'hidden': hidden,
            'pooled': pooled,
            'logits': logits,
        }
        
        # Compute losses
        loss = 0.0
        losses = {}
        
        if ecr_targets is not None:
            ecr_loss = F.mse_loss(predicted_ecr, ecr_targets)
            losses['ecr_loss'] = ecr_loss
            loss = loss + loss_weights['ecr'] * ecr_loss
        
        if aesthetic_targets is not None:
            aes_loss = F.mse_loss(predicted_aesthetic, aesthetic_targets)
            losses['aesthetic_loss'] = aes_loss
            loss = loss + loss_weights['aesthetic'] * aes_loss
        
        if technical_targets is not None:
            tech_loss = F.mse_loss(predicted_technical, technical_targets)
            losses['technical_loss'] = tech_loss
            loss = loss + loss_weights['technical'] * tech_loss
        
        if lm_loss is not None:
            losses['lm_loss'] = lm_loss
            loss = loss + loss_weights['lm'] * lm_loss
        
        outputs['loss'] = loss
        outputs['losses'] = losses
        
        return outputs


# ==============================================
# FACTORY FUNCTION
# ==============================================
def create_student_model(
    model_type: str = 'mlp',
    **kwargs
) -> nn.Module:
    """
    Factory function to create student models.
    
    Args:
        model_type: 'mlp', 'transformer', or 'full'
        **kwargs: Model-specific arguments
    
    Returns:
        Student model instance
    """
    if model_type == 'mlp':
        return StudentMLP(**kwargs)
    elif model_type == 'transformer':
        return StudentTransformer(**kwargs)
    elif model_type == 'full':
        return ViralStudentV2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ==============================================
# TEST
# ==============================================
if __name__ == "__main__":
    print("Testing Student Models V2...\n")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Test data
    batch_size = 4
    imagebind_emb = torch.randn(batch_size, 1024).to(device)
    ecr_targets = torch.rand(batch_size).to(device)
    aesthetic_targets = torch.rand(batch_size).to(device) * 10  # 0-10 scale
    technical_targets = torch.rand(batch_size).to(device) * 10
    
    # Test 1: MLP
    print("=" * 50)
    print("Test 1: StudentMLP")
    print("=" * 50)
    mlp = StudentMLP().to(device)
    out = mlp(imagebind_emb, ecr_targets, aesthetic_targets, technical_targets)
    print(f"  Parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  ECR pred shape: {out['predicted_ecr'].shape}")
    print(f"  Aesthetic pred: {out['predicted_aesthetic'][:2]}")
    
    # Test 2: Transformer
    print("\n" + "=" * 50)
    print("Test 2: StudentTransformer")
    print("=" * 50)
    transformer = StudentTransformer().to(device)
    out = transformer(imagebind_emb, ecr_targets=ecr_targets)
    print(f"  Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  ECR pred shape: {out['predicted_ecr'].shape}")
    
    # Test 3: Full model (lightweight mode)
    print("\n" + "=" * 50)
    print("Test 3: ViralStudentV2 (lightweight)")
    print("=" * 50)
    full = ViralStudentV2(use_vision_encoder=False, use_hf_llm=False).to(device)
    out = full(imagebind_emb=imagebind_emb, ecr_targets=ecr_targets)
    print(f"  Parameters: {sum(p.numel() for p in full.parameters()):,}")
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  ECR pred shape: {out['predicted_ecr'].shape}")
    
    print("\n✅ All tests passed!")
