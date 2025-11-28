import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

class ViralStudentModel(nn.Module):
    def __init__(self, llm_path="microsoft/Phi-3-mini-4k-instruct", vision_path="google/siglip-so400m-patch14-384"):
        super().__init__()
        
        # 1. Vision Encoder (SigLIP - Nhẹ, Tốt)
        print("Loading Vision Encoder...")
        self.vision_tower = AutoModel.from_pretrained(vision_path)
        self.vision_tower.requires_grad_(False) # Freeze vision tower
        
        # 2. LLM Backbone (Phi-3)
        print("Loading LLM...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        # Freeze LLM (chúng ta sẽ dùng LoRA trong train.py, nhưng ở đây cứ để raw)
        
        # 3. Projector (Nối Vision sang LLM)
        # SigLIP output dim = 1152, Phi-3 hidden dim = 3072
        self.mm_projector = nn.Sequential(
            nn.Linear(1152, 3072),
            nn.GELU(),
            nn.Linear(3072, 3072)
        )
        
        # 4. Regression Head (Để dự đoán ECR score cụ thể)
        # Đầu vào là hidden state cuối cùng của LLM
        self.score_head = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid() # ECR nằm trong khoảng 0-1
        )

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, ecr_targets=None):
        # 1. Extract Visual Features
        with torch.no_grad():
            vision_outputs = self.vision_tower.vision_model(pixel_values)
            image_embeds = vision_outputs.last_hidden_state # [Batch, Num_Patches, Dim]
            
        # 2. Project to LLM Space
        image_embeds = self.mm_projector(image_embeds)
        
        # 3. Embed Text Inputs
        inputs_embeds = self.llm.model.embed_tokens(input_ids)
        
        # 4. Concat: [Image_Embeds, Text_Embeds]
        # Lưu ý: Trong thực tế cần xử lý padding cẩn thận, ở đây làm đơn giản nối vào đầu
        inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
        
        # Mở rộng attention mask cho phần hình ảnh
        batch_size = input_ids.shape[0]
        num_images_tokens = image_embeds.shape[1]
        image_mask = torch.ones((batch_size, num_images_tokens), device=input_ids.device)
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # 5. Pass through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        loss = 0
        # 6. Calculate Language Modeling Loss (Text Generation)
        if labels is not None:
            # Cần shift labels để khớp với độ dài mới sau khi nối ảnh
            # (Phần này code demo, thực tế cần collator xử lý kỹ)
            pass 
            # loss += outputs.loss # Giả sử dùng hàm loss có sẵn của Phi-3

        # 7. Calculate Regression Loss (ECR Prediction)
        # Lấy hidden state của token cuối cùng để dự đoán điểm
        last_token_state = outputs.hidden_states[-1][:, -1, :]
        predicted_ecr = self.score_head(last_token_state).squeeze()
        
        if ecr_targets is not None:
            reg_loss = nn.MSELoss()(predicted_ecr, ecr_targets)
            loss += reg_loss # Cộng loss hồi quy vào loss tổng
            
        return {
            "loss": loss,
            "logits": outputs.logits,
            "predicted_ecr": predicted_ecr
        }
