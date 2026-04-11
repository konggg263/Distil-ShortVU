"""
ablation_study.py - Systematic Ablation Study for KD Thesis

Compares 5 configurations required by the thesis:
  1. Student + KD (full, gated fusion)          ← already trained
  2. Student Baseline (gated, no KD)            ← already trained
  3. Visual-Only Student (no text branch)
  4. Text-Only Student (no visual branch)
  5. Concat-Fusion Student (no gate, same params area)

Also measures inference time (ms/video) for trade-off analysis.

Usage (standalone):
    python source/kd/ablation_study.py \
        --data source/kaggle_kd/results/500_videos/features_500.json \
        --kd-model results_kd/student_kd_best.pth \
        --baseline-model results_kd/student_baseline_best.pth \
        --teacher-model results_kd/teacher_best.pth \
        --out results_kd/ablation_report.json

Usage (from run_experiment.py):
    Imported and called automatically when --ablation flag is set.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr, spearmanr, kendalltau

sys.path.insert(0, os.path.dirname(__file__))
from models import StudentModel, TeacherModel, ResidualBlock, count_params


# ================================================================
# ABLATION MODEL VARIANTS
# ================================================================
class StudentVisualOnly(nn.Module):
    """Student that uses ONLY visual embeddings (text branch removed)."""

    def __init__(self, visual_dim=1024, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, visual_emb, text_emb=None, ecr_targets=None, **_):
        h = self.visual_encoder(visual_emb)
        h = self.blocks(h)
        predicted_ecr = self.ecr_head(h).squeeze(-1)
        out = {"predicted_ecr": predicted_ecr}
        if ecr_targets is not None:
            out["loss"] = F.mse_loss(predicted_ecr, ecr_targets)
        return out


class StudentTextOnly(nn.Module):
    """Student that uses ONLY text embeddings (visual branch removed)."""

    def __init__(self, text_dim=384, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, visual_emb=None, text_emb=None, ecr_targets=None, **_):
        h = self.text_encoder(text_emb)
        h = self.blocks(h)
        predicted_ecr = self.ecr_head(h).squeeze(-1)
        out = {"predicted_ecr": predicted_ecr}
        if ecr_targets is not None:
            out["loss"] = F.mse_loss(predicted_ecr, ecr_targets)
        return out


class StudentConcatFusion(nn.Module):
    """
    Student with simple concatenation fusion (no gating).
    Baseline to show gated fusion outperforms naive concat.
    Architecture deliberately mirrors StudentModel for fair comparison.
    """

    def __init__(self, visual_dim=1024, text_dim=384, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # No gate — straight concat → project
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )
        self.ecr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, visual_emb, text_emb, ecr_targets=None, **_):
        v = self.visual_encoder(visual_emb)
        t = self.text_encoder(text_emb)
        fused = self.fusion(torch.cat([v, t], dim=-1))
        fused = self.blocks(fused)
        predicted_ecr = self.ecr_head(fused).squeeze(-1)
        out = {"predicted_ecr": predicted_ecr}
        if ecr_targets is not None:
            out["loss"] = F.mse_loss(predicted_ecr, ecr_targets)
        return out


# ================================================================
# SHARED UTILITIES
# ================================================================
def _evaluate(model, loader, device, model_type="generic"):
    """Evaluate any model variant. Returns metric dict."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            visual = batch["visual_emb"].to(device)
            text = batch["text_emb"].to(device)
            ecr = batch["ecr"]

            if model_type == "teacher":
                quality = batch["quality_scores"].to(device)
                out = model(visual, text, quality)
            elif model_type == "visual_only":
                out = model(visual_emb=visual)
            elif model_type == "text_only":
                out = model(text_emb=text)
            else:
                out = model(visual, text)

            all_pred.extend(out["predicted_ecr"].cpu().numpy())
            all_true.extend(ecr.numpy())

    pred = np.array(all_pred)
    true = np.array(all_true)
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    plcc = float(pearsonr(pred, true)[0]) if len(pred) > 2 else 0.0
    srcc = float(spearmanr(pred, true).correlation) if len(pred) > 2 else 0.0
    ktau = float(kendalltau(pred, true).correlation) if len(pred) > 2 else 0.0
    for v in (plcc, srcc, ktau):
        if np.isnan(v):
            v = 0.0
    return {"plcc": plcc, "srcc": srcc, "ktau": ktau, "mse": mse, "mae": mae}


def _train(model, train_loader, val_loader, epochs, lr, device,
           model_type="generic", verbose=True):
    """Generic training loop for ablation variants."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_mse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            visual = batch["visual_emb"].to(device)
            text = batch["text_emb"].to(device)
            ecr = batch["ecr"].to(device)

            optimizer.zero_grad()
            if model_type == "visual_only":
                out = model(visual_emb=visual, ecr_targets=ecr)
            elif model_type == "text_only":
                out = model(text_emb=text, ecr_targets=ecr)
            else:
                out = model(visual, text, ecr_targets=ecr)

            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out["loss"].item()
            n += 1
        scheduler.step()

        if (epoch % max(1, epochs // 5) == 0 or epoch == epochs) and verbose:
            val = _evaluate(model, val_loader, device, model_type)
            print(f"    ep {epoch:3d}: loss={total_loss/n:.5f} | "
                  f"PLCC={val['plcc']:.4f} MSE={val['mse']:.5f}")
            if val["mse"] < best_mse:
                best_mse = val["mse"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch == epochs and not verbose:
            val = _evaluate(model, val_loader, device, model_type)
            if val["mse"] < best_mse:
                best_mse = val["mse"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ================================================================
# INFERENCE TIME MEASUREMENT
# ================================================================
def measure_inference_time(model, sample_visual, sample_text, device,
                           model_type="generic", sample_quality=None, n_runs=200):
    """
    Measure average inference time in ms for a single sample.
    Warms up 20 runs then averages over n_runs.
    """
    model.eval()
    v = sample_visual.unsqueeze(0).to(device)
    t = sample_text.unsqueeze(0).to(device)
    q = sample_quality.unsqueeze(0).to(device) if sample_quality is not None else None

    with torch.no_grad():
        for _ in range(20):
            if model_type == "visual_only":
                model(visual_emb=v)
            elif model_type == "text_only":
                model(text_emb=t)
            elif model_type == "teacher":
                model(v, t, q)
            else:
                model(v, t)

        start = time.perf_counter()
        for _ in range(n_runs):
            if model_type == "visual_only":
                model(visual_emb=v)
            elif model_type == "text_only":
                model(text_emb=t)
            elif model_type == "teacher":
                model(v, t, q)
            else:
                model(v, t)
        elapsed = time.perf_counter() - start

    return round(elapsed / n_runs * 1000, 3)  # ms


# ================================================================
# MAIN ABLATION RUNNER
# ================================================================
def run_ablation_study(train_data, val_data, epochs, lr, batch_size, device,
                       pretrained_kd=None, pretrained_baseline=None,
                       pretrained_teacher=None, save_dir=None,
                       hidden_dim=256, teacher_hidden=512):
    """
    Train and evaluate all ablation variants. Returns full report dict.

    Args:
        train_data / val_data: lists of feature dicts
        pretrained_kd / pretrained_baseline / pretrained_teacher: state dicts (optional)
    """
    from run_experiment import KDDataset  # reuse dataset class

    train_loader = DataLoader(KDDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KDDataset(val_data), batch_size=batch_size, shuffle=False)

    # pick a fixed sample for inference time measurement
    sample = val_data[0]
    s_vis = torch.tensor(sample["visual_emb"], dtype=torch.float32)
    s_txt = torch.tensor(sample.get("text_emb") or [0.0]*384, dtype=torch.float32)
    s_qual = torch.tensor([
        sample.get("quality_scores", {}).get("aesthetic", 5.0) / 10.0,
        sample.get("quality_scores", {}).get("technical", 5.0) / 10.0
    ], dtype=torch.float32)

    results = {}

    # ── A. Teacher (pre-trained, just evaluate) ─────────────────────────
    if pretrained_teacher is not None:
        print("\n  [Teacher] evaluating ...")
        teacher = TeacherModel(hidden_dim=teacher_hidden).to(device)
        teacher.load_state_dict(pretrained_teacher)
        metrics = _evaluate(teacher, val_loader, device, "teacher")
        infer_ms = measure_inference_time(teacher,
            s_vis, s_txt, device, "teacher", sample_quality=s_qual)
        total_p, _ = count_params(teacher)
        results["teacher"] = {**metrics, "params": total_p, "inference_ms": infer_ms}
        print(f"    PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} "
              f"MSE={metrics['mse']:.5f} | {infer_ms:.2f} ms/video")

    # ── B. Student + KD (pre-trained) ───────────────────────────────────
    if pretrained_kd is not None:
        print("\n  [Student + KD] evaluating ...")
        stu_kd = StudentModel(hidden_dim=hidden_dim,
                              teacher_hidden_dim=teacher_hidden).to(device)
        stu_kd.load_state_dict(pretrained_kd)
        metrics = _evaluate(stu_kd, val_loader, device)
        infer_ms = measure_inference_time(stu_kd, s_vis, s_txt, device)
        total_p, _ = count_params(stu_kd)
        results["student_kd"] = {**metrics, "params": total_p, "inference_ms": infer_ms}
        print(f"    PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} "
              f"MSE={metrics['mse']:.5f} | {infer_ms:.2f} ms/video")

    # ── C. Student Baseline (pre-trained) ───────────────────────────────
    if pretrained_baseline is not None:
        print("\n  [Student Baseline] evaluating ...")
        stu_base = StudentModel(hidden_dim=hidden_dim,
                                teacher_hidden_dim=teacher_hidden).to(device)
        stu_base.load_state_dict(pretrained_baseline)
        metrics = _evaluate(stu_base, val_loader, device)
        infer_ms = measure_inference_time(stu_base, s_vis, s_txt, device)
        total_p, _ = count_params(stu_base)
        results["student_baseline"] = {**metrics, "params": total_p,
                                        "inference_ms": infer_ms}
        print(f"    PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} "
              f"MSE={metrics['mse']:.5f} | {infer_ms:.2f} ms/video")

    # ── D. Visual-Only ───────────────────────────────────────────────────
    print("\n  [Visual-Only] training ...")
    vis_model = StudentVisualOnly(hidden_dim=hidden_dim).to(device)
    vis_model = _train(vis_model, train_loader, val_loader, epochs, lr, device,
                       model_type="visual_only")
    metrics = _evaluate(vis_model, val_loader, device, "visual_only")
    infer_ms = measure_inference_time(vis_model, s_vis, s_txt, device, "visual_only")
    total_p, _ = count_params(vis_model)
    results["visual_only"] = {**metrics, "params": total_p, "inference_ms": infer_ms}
    print(f"    PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} "
          f"MSE={metrics['mse']:.5f} | {infer_ms:.2f} ms/video")
    if save_dir:
        torch.save(vis_model.state_dict(),
                   os.path.join(save_dir, "ablation_visual_only.pth"))

    # ── E. Text-Only ─────────────────────────────────────────────────────
    print("\n  [Text-Only] training ...")
    txt_model = StudentTextOnly(hidden_dim=hidden_dim).to(device)
    txt_model = _train(txt_model, train_loader, val_loader, epochs, lr, device,
                       model_type="text_only")
    metrics = _evaluate(txt_model, val_loader, device, "text_only")
    infer_ms = measure_inference_time(txt_model, s_vis, s_txt, device, "text_only")
    total_p, _ = count_params(txt_model)
    results["text_only"] = {**metrics, "params": total_p, "inference_ms": infer_ms}
    print(f"    PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} "
          f"MSE={metrics['mse']:.5f} | {infer_ms:.2f} ms/video")
    if save_dir:
        torch.save(txt_model.state_dict(),
                   os.path.join(save_dir, "ablation_text_only.pth"))

    # ── F. Concat Fusion (no gate) ───────────────────────────────────────
    print("\n  [Concat Fusion] training ...")
    concat_model = StudentConcatFusion(hidden_dim=hidden_dim).to(device)
    concat_model = _train(concat_model, train_loader, val_loader, epochs, lr, device)
    metrics = _evaluate(concat_model, val_loader, device)
    infer_ms = measure_inference_time(concat_model, s_vis, s_txt, device)
    total_p, _ = count_params(concat_model)
    results["concat_fusion"] = {**metrics, "params": total_p, "inference_ms": infer_ms}
    print(f"    PLCC={metrics['plcc']:.4f} SRCC={metrics['srcc']:.4f} "
          f"MSE={metrics['mse']:.5f} | {infer_ms:.2f} ms/video")
    if save_dir:
        torch.save(concat_model.state_dict(),
                   os.path.join(save_dir, "ablation_concat_fusion.pth"))

    # ── Print comparison table ───────────────────────────────────────────
    _print_ablation_table(results)
    return results


def _print_ablation_table(results: dict):
    label_map = {
        "teacher":           "Teacher (upper bound)",
        "student_kd":        "Student + KD (gated)",
        "student_baseline":  "Student Baseline (gated)",
        "concat_fusion":     "Student + Concat Fusion",
        "visual_only":       "Student Visual-Only",
        "text_only":         "Student Text-Only",
    }
    order = ["teacher", "student_kd", "student_baseline",
             "concat_fusion", "visual_only", "text_only"]

    print(f"\n{'='*85}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*85}")
    header = f"  {'Model':<30} {'Params':>8} {'PLCC':>7} {'SRCC':>7} {'KTAU':>7} {'MSE':>9} {'Infer(ms)':>10}"
    print(header)
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*10}")

    for key in order:
        if key not in results:
            continue
        r = results[key]
        label = label_map.get(key, key)
        print(f"  {label:<30} {r.get('params',0):>8,} "
              f"{r.get('plcc',0):>7.4f} {r.get('srcc',0):>7.4f} "
              f"{r.get('ktau',0):>7.4f} {r.get('mse',0):>9.6f} "
              f"{r.get('inference_ms',0):>10.2f}")

    # KD gain
    if "student_kd" in results and "student_baseline" in results:
        gain = results["student_kd"]["plcc"] - results["student_baseline"]["plcc"]
        print(f"\n  KD gain (PLCC): Student+KD vs Baseline = {gain:+.4f}")
    # Gated vs Concat
    if "student_kd" in results and "concat_fusion" in results:
        gain = results["student_kd"]["plcc"] - results["concat_fusion"]["plcc"]
        print(f"  Gated vs Concat (PLCC): {gain:+.4f}")
    # Multimodal vs single modal
    if "student_kd" in results and "visual_only" in results:
        gain = results["student_kd"]["plcc"] - results["visual_only"]["plcc"]
        print(f"  Multimodal vs Visual-Only (PLCC): {gain:+.4f}")
    if "student_kd" in results and "text_only" in results:
        gain = results["student_kd"]["plcc"] - results["text_only"]["plcc"]
        print(f"  Multimodal vs Text-Only (PLCC): {gain:+.4f}")
    print(f"{'='*85}\n")


# ================================================================
# CLI ENTRY POINT
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="features JSON")
    parser.add_argument("--kd-model", default=None)
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--out", default="results_kd/ablation_report.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--student-hidden", type=int, default=256)
    parser.add_argument("--teacher-hidden", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)

    with open(args.data) as f:
        raw = json.load(f)
    data = list(raw.values()) if isinstance(raw, dict) else raw
    valid = [d for d in data if d.get("visual_emb") and d.get("ecr") is not None]

    np.random.seed(42)
    idx = np.random.permutation(len(valid))
    split = int(len(valid) * 0.8)
    train_data = [valid[i] for i in idx[:split]]
    val_data = [valid[i] for i in idx[split:]]
    print(f"Data: {len(train_data)} train, {len(val_data)} val")

    def _load(path, model_class, **kwargs):
        if path is None:
            return None
        m = model_class(**kwargs).to(device)
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        return m.state_dict()

    kd_state = _load(args.kd_model, StudentModel,
                     hidden_dim=args.student_hidden,
                     teacher_hidden_dim=args.teacher_hidden)
    base_state = _load(args.baseline_model, StudentModel,
                       hidden_dim=args.student_hidden,
                       teacher_hidden_dim=args.teacher_hidden)
    teacher_state = _load(args.teacher_model, TeacherModel,
                          hidden_dim=args.teacher_hidden)

    results = run_ablation_study(
        train_data, val_data,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch, device=device,
        pretrained_kd=kd_state, pretrained_baseline=base_state,
        pretrained_teacher=teacher_state,
        save_dir=os.path.dirname(args.out) or "results_kd",
        hidden_dim=args.student_hidden, teacher_hidden=args.teacher_hidden,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
