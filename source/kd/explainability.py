"""
explainability.py - Ablation-based Explainability Engine for StudentModel

Three analysis methods:
  1. Modality Ablation  - zero-out visual/text to measure contribution
  2. Quality Diagnosis  - read aesthetic/technical scores from internal heads
  3. LLM Prompt Builder - format findings into a strict, hallucination-free prompt

Usage (standalone):
    python source/kd/explainability.py \
        --model results_kd/student_kd_best.pth \
        --data source/kaggle_kd/results/500_videos/features_500.json \
        --out results_kd/explanations.json

Usage (in code):
    from source.kd.explainability import ExplainabilityEngine
    engine = ExplainabilityEngine(student_model, device)
    exp = engine.explain(visual_emb, text_emb, video_meta)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from models import StudentModel, count_params


# ================================================================
# EXPLAINABILITY ENGINE
# ================================================================
class ExplainabilityEngine:
    """
    Ablation-based explainability for StudentModel.
    All numbers come directly from model forward passes — no hallucination.
    """

    def __init__(self, model: StudentModel, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def explain(self, visual_emb: torch.Tensor, text_emb: torch.Tensor,
                video_meta: dict = None) -> dict:
        """
        Full explanation for one video sample.

        Args:
            visual_emb: shape [D_v] or [1, D_v]
            text_emb:   shape [D_t] or [1, D_t]
            video_meta: optional dict with keys 'video_id', 'title', 'caption'

        Returns:
            explanation dict ready for JSON serialisation and LLM prompt
        """
        v = self._to_batch(visual_emb)
        t = self._to_batch(text_emb)

        # ── 1. Baseline (full input) ──────────────────────────────────────
        base_out = self.model(v, t)
        base_ecr = float(base_out['predicted_ecr'].item())
        base_aes = float(base_out['predicted_aesthetic'].item())
        base_tech = float(base_out['predicted_technical'].item())
        gate_w = base_out['gate_weights'].squeeze(0).cpu().numpy()

        # ── 2. Visual-only ablation (zero text) ───────────────────────────
        t_zero = torch.zeros_like(t)
        vis_out = self.model(v, t_zero)
        ecr_visual_only = float(vis_out['predicted_ecr'].item())
        text_drop = base_ecr - ecr_visual_only  # how much ECR drops without text

        # ── 3. Text-only ablation (zero visual) ───────────────────────────
        v_zero = torch.zeros_like(v)
        txt_out = self.model(v_zero, t)
        ecr_text_only = float(txt_out['predicted_ecr'].item())
        visual_drop = base_ecr - ecr_text_only  # how much ECR drops without visual

        # ── 4. Normalised importance (%) ──────────────────────────────────
        total = abs(visual_drop) + abs(text_drop) + 1e-9
        visual_pct = abs(visual_drop) / total * 100
        text_pct = abs(text_drop) / total * 100

        # ── 5. ECR interpretation ─────────────────────────────────────────
        ecr_level = (
            "very high (>0.08)" if base_ecr > 0.08 else
            "high (0.05–0.08)" if base_ecr > 0.05 else
            "moderate (0.03–0.05)" if base_ecr > 0.03 else
            "low (<0.03)"
        )

        # ── 6. Quality interpretation ─────────────────────────────────────
        # aesthetic/technical heads output raw logits; normalise to 0-10 scale
        aes_10 = round(float(np.clip(base_aes * 10, 0, 10)), 1)
        tech_10 = round(float(np.clip(base_tech * 10, 0, 10)), 1)

        # ── 7. Gate analysis (which hidden dims were opened) ──────────────
        gate_mean = float(gate_w.mean())
        gate_above_half = int((gate_w > 0.5).sum())
        gate_total = int(gate_w.shape[0])

        explanation = {
            "video_id": (video_meta or {}).get("video_id", "unknown"),
            "predicted_ecr": round(base_ecr, 6),
            "ecr_level": ecr_level,
            "modality_ablation": {
                "ecr_full": round(base_ecr, 6),
                "ecr_visual_only": round(ecr_visual_only, 6),
                "ecr_text_only": round(ecr_text_only, 6),
                "visual_drop": round(visual_drop, 6),
                "text_drop": round(text_drop, 6),
            },
            "relative_importance": {
                "visual_pct": round(visual_pct, 1),
                "text_pct": round(text_pct, 1),
                "dominant": "visual" if visual_pct >= text_pct else "text",
            },
            "internal_quality": {
                "aesthetic_score_10": aes_10,
                "technical_score_10": tech_10,
                "quality_verdict": _quality_verdict(aes_10, tech_10),
            },
            "gate_analysis": {
                "gate_mean_activation": round(gate_mean, 4),
                "active_dims": f"{gate_above_half}/{gate_total}",
            },
            "metadata": {
                "title": (video_meta or {}).get("title", ""),
                "caption": (video_meta or {}).get("caption", ""),
            },
        }
        return explanation

    def explain_batch(self, data_list: list, max_samples: int = None) -> list:
        """Run explain() on a list of feature dicts."""
        samples = data_list[:max_samples] if max_samples else data_list
        results = []
        for item in samples:
            v = torch.tensor(item["visual_emb"], dtype=torch.float32)
            t = torch.tensor(item.get("text_emb") or [0.0] * 384, dtype=torch.float32)
            meta = {k: item.get(k, "") for k in ("video_id", "title", "caption")}
            exp = self.explain(v, t, meta)
            exp["true_ecr"] = float(item.get("ecr", 0))
            results.append(exp)
        return results

    def generate_llm_prompt(self, explanation: dict) -> str:
        """
        Build a strict, grounded prompt for ChatGPT / Claude / Gemini.
        The prompt forbids the LLM from adding reasons not in the JSON data.
        """
        m = explanation
        ri = m["relative_importance"]
        iq = m["internal_quality"]
        title = m["metadata"].get("title") or "N/A"
        caption = m["metadata"].get("caption") or "N/A"

        dominant = ri["dominant"]
        dominant_vi = "hình ảnh/video" if dominant == "visual" else "văn bản/tiêu đề"
        minor_vi = "văn bản/tiêu đề" if dominant == "visual" else "hình ảnh/video"
        dominant_pct = ri["visual_pct"] if dominant == "visual" else ri["text_pct"]
        minor_pct = ri["text_pct"] if dominant == "visual" else ri["visual_pct"]

        prompt = f"""Bạn là một chuyên gia phân tích nội dung mạng xã hội.
Hãy dựa VÀO CHÍNH XÁC các số liệu Toán học dưới đây để viết nhận xét cho người tạo nội dung.
KHÔNG ĐƯỢC tự bịa thêm bất kỳ lý do nào ngoài các số liệu được cung cấp.

[THÔNG TIN VIDEO]
- Tiêu đề: "{title}"
- Caption nhận diện trong video: "{caption}"

[KẾT QUẢ TỪ HỆ THỐNG DEEP LEARNING]
- Điểm thu hút dự đoán (ECR): {m['predicted_ecr']:.4f} — mức: {m['ecr_level']}
- Đóng góp vào sức hút: {dominant_vi} chiếm {dominant_pct:.1f}%, {minor_vi} chiếm {minor_pct:.1f}%
  (Khi xóa hình ảnh: ECR giảm {abs(m['modality_ablation']['visual_drop']):.4f}; khi xóa văn bản: ECR giảm {abs(m['modality_ablation']['text_drop']):.4f})
- Điểm Thẩm mỹ (Aesthetic): {iq['aesthetic_score_10']}/10
- Điểm Kỹ thuật (Technical): {iq['technical_score_10']}/10
- Chất lượng tổng thể: {iq['quality_verdict']}

Hãy viết đúng 3 đoạn:
1. 📈 Phân tích sức hút (dựa trên ECR và tỷ lệ đóng góp)
2. 💡 Khuyến nghị cải thiện (dựa trên điểm kỹ thuật và thẩm mỹ)
3. 🌟 Tổng kết (1 câu ngắn)

Viết bằng tiếng Việt, lịch sự và chuyên nghiệp."""
        return prompt

    def print_summary(self, explanation: dict):
        """Pretty-print explanation to console."""
        m = explanation
        ri = m["relative_importance"]
        iq = m["internal_quality"]
        print(f"\n{'─'*55}")
        print(f"  VIDEO: {m['video_id']}")
        print(f"  Predicted ECR : {m['predicted_ecr']:.6f}  [{m['ecr_level']}]")
        print(f"  True ECR      : {m.get('true_ecr', 'N/A')}")
        print(f"  ── Modality Ablation ──")
        print(f"    Full         : {m['modality_ablation']['ecr_full']:.6f}")
        print(f"    Visual only  : {m['modality_ablation']['ecr_visual_only']:.6f}  (text zeroed out)")
        print(f"    Text only    : {m['modality_ablation']['ecr_text_only']:.6f}  (visual zeroed out)")
        print(f"    → Visual importance : {ri['visual_pct']:.1f}%  |  Text: {ri['text_pct']:.1f}%")
        print(f"    → Dominant channel  : {ri['dominant'].upper()}")
        print(f"  ── Internal Quality ──")
        print(f"    Aesthetic    : {iq['aesthetic_score_10']}/10")
        print(f"    Technical    : {iq['technical_score_10']}/10")
        print(f"    Verdict      : {iq['quality_verdict']}")
        print(f"  ── Gate Analysis ──")
        print(f"    Active dims  : {m['gate_analysis']['active_dims']}")
        print(f"    Mean gate    : {m['gate_analysis']['gate_mean_activation']:.4f}")
        if m["metadata"].get("caption"):
            print(f"  Caption: \"{m['metadata']['caption'][:80]}\"")
        print(f"{'─'*55}")

    # ── helpers ──────────────────────────────────────────────────────
    def _to_batch(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(self.device)


def _quality_verdict(aes: float, tech: float) -> str:
    if aes >= 7 and tech >= 7:
        return "Tốt (cả thẩm mỹ lẫn kỹ thuật đều cao)"
    if aes >= 7 and tech < 5:
        return "Nội dung đẹp nhưng chất lượng quay kém (mờ/rung)"
    if aes < 5 and tech >= 7:
        return "Chất lượng quay tốt nhưng nội dung kém hấp dẫn"
    if aes < 5 and tech < 5:
        return "Cả thẩm mỹ và kỹ thuật đều cần cải thiện"
    return "Trung bình (có thể cải thiện thêm)"


# ================================================================
# DATASET-LEVEL SUMMARY
# ================================================================
def summarise_explanations(explanations: list) -> dict:
    """Compute aggregate statistics over a list of explanations."""
    visual_pcts = [e["relative_importance"]["visual_pct"] for e in explanations]
    text_pcts = [e["relative_importance"]["text_pct"] for e in explanations]
    ecrs = [e["predicted_ecr"] for e in explanations]
    aes = [e["internal_quality"]["aesthetic_score_10"] for e in explanations]
    tech = [e["internal_quality"]["technical_score_10"] for e in explanations]

    dominant_visual = sum(1 for e in explanations
                          if e["relative_importance"]["dominant"] == "visual")
    return {
        "n_samples": len(explanations),
        "visual_dominant_pct": round(dominant_visual / len(explanations) * 100, 1),
        "avg_visual_importance": round(float(np.mean(visual_pcts)), 1),
        "avg_text_importance": round(float(np.mean(text_pcts)), 1),
        "avg_predicted_ecr": round(float(np.mean(ecrs)), 6),
        "avg_aesthetic_10": round(float(np.mean(aes)), 2),
        "avg_technical_10": round(float(np.mean(tech)), 2),
    }


# ================================================================
# CLI ENTRY POINT
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Explainability Engine for StudentModel")
    parser.add_argument("--model", required=True, help="Path to student_kd_best.pth")
    parser.add_argument("--data", required=True, help="Path to features JSON")
    parser.add_argument("--out", default="results_kd/explanations.json")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to explain")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", action="store_true", help="Print LLM prompt for first sample")
    parser.add_argument("--student-hidden", type=int, default=256)
    parser.add_argument("--teacher-hidden", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.model} ...")
    model = StudentModel(hidden_dim=args.student_hidden,
                         teacher_hidden_dim=args.teacher_hidden).to(device)
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    total, _ = count_params(model)
    print(f"  Student params: {total:,}")

    print(f"Loading data from {args.data} ...")
    with open(args.data) as f:
        raw = json.load(f)
    data = list(raw.values()) if isinstance(raw, dict) else raw
    valid = [d for d in data if d.get("visual_emb") and d.get("ecr") is not None]
    print(f"  {len(valid)} samples available — explaining {min(args.n, len(valid))}")

    engine = ExplainabilityEngine(model, device)
    explanations = engine.explain_batch(valid, max_samples=args.n)

    for exp in explanations:
        engine.print_summary(exp)

    summary = summarise_explanations(explanations)
    print(f"\n{'='*55}")
    print("  DATASET SUMMARY")
    print(f"{'='*55}")
    for k, v in summary.items():
        print(f"  {k:<35}: {v}")

    if args.prompt and explanations:
        print(f"\n{'='*55}")
        print("  LLM PROMPT (first sample)")
        print(f"{'='*55}")
        print(engine.generate_llm_prompt(explanations[0]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    output = {"summary": summary, "explanations": explanations}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
