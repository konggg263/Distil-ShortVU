"""
models.py - Teacher and Student models for Knowledge Distillation

Architecture comparison:
┌─────────────────────────────────────────────────────────────┐
│ Teacher Model (~5M params)  │  Student Model (~1.5M params) │
├─────────────────────────────┼───────────────────────────────┤
│ Input: visual(1024)         │  Input: visual(1024)          │
│      + text(384)            │       + text(384)             │
│      + quality(2)           │  (NO quality scores)          │
│                             │                               │
│ Cross-attention fusion      │  Gated multimodal fusion      │
│ 4 residual blocks (512-d)   │  2 residual blocks (256-d)    │
│                             │                               │
│ Output: ECR                 │  Output: ECR                  │
│                             │        + aesthetic (aux/KD)    │
│                             │        + technical (aux/KD)    │
│                             │        + KD projection         │
└─────────────────────────────┴───────────────────────────────┘

KD Loss for Student:
  L = L_ecr_hard + α*L_ecr_soft + β*L_repr + γ*L_aesthetic + δ*L_technical
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


# ================================================================
# TEACHER MODEL
# ================================================================
class TeacherModel(nn.Module):
    """
    Large teacher model using ALL features for best ECR prediction.
    Serves as the upper bound in KD comparison.

    Inputs: visual_emb(1024) + text_emb(384) + quality_scores(2)
    """

    def __init__(self, visual_dim=1024, text_dim=384, hidden_dim=512,
                 n_blocks=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.quality_proj = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual_emb, text_emb, quality_scores, ecr_targets=None):
        v = self.visual_proj(visual_emb)
        t = self.text_proj(text_emb)
        q = self.quality_proj(quality_scores)

        tokens = torch.stack([v, t, q], dim=1)  # (B, 3, hidden)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)
        hidden = tokens.mean(dim=1)  # (B, hidden)

        hidden = self.blocks(hidden)
        predicted_ecr = self.ecr_head(hidden).squeeze(-1)

        outputs = {'predicted_ecr': predicted_ecr, 'hidden': hidden}

        if ecr_targets is not None:
            outputs['loss'] = F.mse_loss(predicted_ecr, ecr_targets)

        return outputs


# ================================================================
# STUDENT MODEL (with KD support)
# ================================================================
class StudentModel(nn.Module):
    """
    Lightweight student with gated fusion + KD components.
    Uses only visual + text (no quality scores as input).

    Auxiliary heads learn to predict quality scores from the teacher.
    KD projector aligns student's hidden space with teacher's.
    """

    def __init__(self, visual_dim=1024, text_dim=384, hidden_dim=256,
                 teacher_hidden_dim=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )
        self.aesthetic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.technical_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.kd_projector = nn.Sequential(
            nn.Linear(hidden_dim, teacher_hidden_dim),
            nn.LayerNorm(teacher_hidden_dim),
        )

    def forward(self, visual_emb, text_emb, ecr_targets=None,
                aesthetic_targets=None, technical_targets=None,
                teacher_ecr=None, teacher_hidden=None,
                loss_weights=None):
        if loss_weights is None:
            loss_weights = {
                'ecr_hard': 1.0, 'ecr_soft': 0.5, 'kd_repr': 0.3,
                'aesthetic': 0.2, 'technical': 0.2,
            }

        v = self.visual_encoder(visual_emb)
        t = self.text_encoder(text_emb)

        concat = torch.cat([v, t], dim=-1)
        g = self.gate(concat)
        fused = self.fusion(concat) * g

        predicted_ecr = self.ecr_head(fused).squeeze(-1)
        predicted_aesthetic = self.aesthetic_head(fused).squeeze(-1)
        predicted_technical = self.technical_head(fused).squeeze(-1)

        outputs = {
            'predicted_ecr': predicted_ecr,
            'predicted_aesthetic': predicted_aesthetic,
            'predicted_technical': predicted_technical,
            'fused_hidden': fused, 'gate_weights': g,
        }

        loss = torch.tensor(0.0, device=visual_emb.device)
        losses = {}

        if ecr_targets is not None:
            l = F.mse_loss(predicted_ecr, ecr_targets)
            losses['ecr_hard'] = l
            loss = loss + loss_weights['ecr_hard'] * l

        if teacher_ecr is not None:
            l = F.mse_loss(predicted_ecr, teacher_ecr)
            losses['ecr_soft'] = l
            loss = loss + loss_weights.get('ecr_soft', 0) * l

        if teacher_hidden is not None:
            proj = self.kd_projector(fused)
            l = 1.0 - F.cosine_similarity(proj, teacher_hidden, dim=-1).mean()
            losses['kd_repr'] = l
            loss = loss + loss_weights.get('kd_repr', 0) * l

        if aesthetic_targets is not None:
            l = F.mse_loss(predicted_aesthetic, aesthetic_targets)
            losses['aesthetic'] = l
            loss = loss + loss_weights.get('aesthetic', 0) * l

        if technical_targets is not None:
            l = F.mse_loss(predicted_technical, technical_targets)
            losses['technical'] = l
            loss = loss + loss_weights.get('technical', 0) * l

        outputs['loss'] = loss
        outputs['losses'] = losses
        return outputs

    def get_feature_importance(self, visual_emb, text_emb):
        """Ablation-based modality importance."""
        self.eval()
        with torch.no_grad():
            full = self(visual_emb, text_emb)['predicted_ecr'].item()
            vis_only = self(visual_emb, torch.zeros_like(text_emb))['predicted_ecr'].item()
            txt_only = self(torch.zeros_like(visual_emb), text_emb)['predicted_ecr'].item()
        return {
            'full_ecr': full,
            'visual_importance': abs(full - txt_only),
            'text_importance': abs(full - vis_only),
        }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
