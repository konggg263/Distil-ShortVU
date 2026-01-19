import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModelForCausalLM


class ViralStudentModel(nn.Module):
    """A lightweight student model for quick experiments.

    Design:
    - Vision encoder: CLIP image encoder (frozen)
    - Simple token embedding + Transformer encoder as a lightweight 'LLM'
    - Project visual tokens into the transformer hidden dim and concatenate with text embeddings
    - Regression head to predict ECR (0-1)

    This keeps dependencies small and avoids requiring a full HF causal LM during dev.
    """
    def __init__(self, vocab_size=50257, d_model=512, n_heads=8, n_layers=4, vision_model_name="openai/clip-vit-base-patch32", use_hf_llm=False, llm_name="gpt2"):
        super().__init__()

        # Vision encoder (frozen)
        print("Loading CLIP vision encoder...")
        self.vision = CLIPModel.from_pretrained(vision_model_name).vision_model
        for p in self.vision.parameters():
            p.requires_grad = False

        # Dimensions
        self.vision_dim = self.vision.config.hidden_size
        self.use_hf_llm = use_hf_llm

        # If using HF causal LM, load it and align dims
        if self.use_hf_llm:
            print(f"Loading HF causal LM: {llm_name}")
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
            self.d_model = self.llm.config.hidden_size
            # We'll use the HF LM's input embeddings
            self.token_emb = None
            self.vocab_size = self.llm.config.vocab_size
        else:
            self.d_model = d_model
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.vocab_size = vocab_size

        # Projector from vision_dim -> d_model
        self.vision_projector = nn.Linear(self.vision_dim, self.d_model)

        # If not using HF LM, use a small transformer encoder as the 'LLM'
        if not self.use_hf_llm:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=n_heads)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            # LM head (to produce logits when training language loss)
            self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        # Regression head for ECR prediction (works in both modes)
        self.score_head = nn.Sequential(
            nn.Linear(self.d_model, max(8, self.d_model // 2)),
            nn.ReLU(True),
            nn.Linear(max(8, self.d_model // 2), 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, ecr_targets=None):
        device = next(self.parameters()).device

        # 1) Visual features
        if pixel_values is None:
            raise ValueError("pixel_values is required")

        # CLIP vision_model expects pixel_values shape (B, C, H, W)
        vision_outputs = self.vision(pixel_values)
        # last_hidden_state shape: (B, seq_len, vision_dim)
        img_feats = vision_outputs.last_hidden_state
        img_proj = self.vision_projector(img_feats)  # (B, img_seq, d_model)

        # 2) Text embeddings and two modes: HF causal LM or lightweight transformer
        batch_size = img_proj.size(0)
        img_len = img_proj.size(1)

        if self.use_hf_llm:
            # Use HF LM embeddings
            if input_ids is None:
                text_emb = torch.zeros((batch_size, 0, self.d_model), device=device)
                text_len = 0
            else:
                embed_layer = self.llm.get_input_embeddings()
                text_emb = embed_layer(input_ids)
                text_len = text_emb.size(1)

            # Concatenate and build attention/labels for HF LM
            concat = torch.cat([img_proj, text_emb], dim=1)

            # Build attention mask
            if attention_mask is None:
                text_mask = torch.ones((batch_size, text_len), device=device, dtype=torch.long)
            else:
                text_mask = attention_mask
            img_mask = torch.ones((batch_size, img_len), device=device, dtype=torch.long)
            full_attention_mask = torch.cat([img_mask, text_mask], dim=1)

            # Prepare labels for HF LM: pad image token positions with -100
            labels_full = None
            logits = None
            if labels is not None:
                labels_full = torch.full((batch_size, img_len + text_len), -100, device=device, dtype=torch.long)
                labels_full[:, img_len:] = labels[:, :text_len]

            # Forward HF LM with inputs_embeds
            outputs = self.llm(
                inputs_embeds=concat,
                attention_mask=full_attention_mask,
                labels=labels_full,
                output_hidden_states=True
            )

            # logits correspond to text portion (after image tokens)
            if hasattr(outputs, 'logits') and outputs.logits is not None and text_len > 0:
                logits = outputs.logits[:, img_len:, :]

            # Regression: pool image token hidden states
            hidden = outputs.hidden_states[-1]
            pooled_img = hidden[:, :img_len, :].mean(dim=1)
            predicted_ecr = self.score_head(pooled_img).squeeze(-1)

            loss = 0.0
            losses = {}
            # HF LM may already include lm loss in outputs.loss
            if outputs.loss is not None:
                losses['ce_loss'] = outputs.loss
                loss = loss + outputs.loss

            if ecr_targets is not None:
                reg_loss = nn.MSELoss()(predicted_ecr, ecr_targets)
                losses['reg_loss'] = reg_loss
                loss = loss + reg_loss

            return {
                'loss': loss,
                'losses': losses,
                'logits': logits,
                'predicted_ecr': predicted_ecr
            }

        else:
            # Lightweight transformer path (previous implementation)
            if input_ids is None:
                text_emb = torch.zeros((batch_size, 0, self.d_model), device=device)
                text_len = 0
            else:
                text_emb = self.token_emb(input_ids)
                text_len = text_emb.size(1)

            concat = torch.cat([img_proj, text_emb], dim=1)
            concat_t = concat.permute(1, 0, 2)

            # Build src_key_padding_mask
            if input_ids is None:
                src_key_pad = None
            else:
                text_pad = (attention_mask == 0) if attention_mask is not None else torch.zeros((batch_size, text_len), dtype=torch.bool, device=device)
                img_pad = torch.zeros((batch_size, img_len), dtype=torch.bool, device=device)
                src_key_pad = torch.cat([img_pad, text_pad], dim=1)

            if src_key_pad is None:
                transformed = self.transformer(concat_t)
            else:
                transformed = self.transformer(concat_t, src_key_padding_mask=src_key_pad)

            transformed = transformed.permute(1, 0, 2)

            logits = None
            if text_len > 0:
                text_out = transformed[:, img_len:, :]
                logits = self.lm_head(text_out)

            pooled_img = transformed[:, :img_len, :].mean(dim=1)
            predicted_ecr = self.score_head(pooled_img).squeeze(-1)

            loss = 0.0
            losses = {}
            if ecr_targets is not None:
                reg_loss = nn.MSELoss()(predicted_ecr, ecr_targets)
                losses['reg_loss'] = reg_loss
                loss = loss + reg_loss

            if labels is not None and logits is not None:
                shift_logits = logits.view(-1, logits.size(-1))
                shift_labels = labels[:, :text_len].contiguous().view(-1)
                ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)
                losses['ce_loss'] = ce_loss
                loss = loss + ce_loss

            return {
                'loss': loss,
                'losses': losses,
                'logits': logits,
                'predicted_ecr': predicted_ecr
            }
