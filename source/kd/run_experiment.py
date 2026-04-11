"""
run_experiment.py - Complete Knowledge Distillation experiment

Pipeline:
  1. Load extracted features (JSON)
  2. Split into train/val (80/20)
  3. Train Teacher Model (large, ALL features) → upper bound
  4. Generate teacher soft targets for all data
  5. Train Student Baseline (no KD, only hard ECR labels)
  6. Train Student + KD (hard labels + soft targets + repr distillation + aux tasks)
  7. Compare all three models → print report

Usage:
    # With extracted features JSON
    python source/kd/run_experiment.py --data source/kaggle_kd/results/500_videos/features_500.json

    # With custom settings
    python source/kd/run_experiment.py --data source/kaggle_kd/results/500_videos/features_500.json \\
        --teacher-epochs 60 --student-epochs 80 --batch 32

    # Quick test with fewer epochs
    python source/kd/run_experiment.py --data source/kaggle_kd/results/500_videos/features_500.json --quick
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr, pearsonr, kendalltau

sys.path.insert(0, os.path.dirname(__file__))
from models import TeacherModel, StudentModel, count_params
from explainability import ExplainabilityEngine, summarise_explanations
from ablation_study import run_ablation_study


# ================================================================
# DATASET
# ================================================================
class KDDataset(Dataset):
    """Dataset for KD experiment. Handles both notebook and pipeline_fast JSON formats."""

    def __init__(self, data_list, text_dim=384):
        self.data = data_list
        self.text_dim = text_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        visual = item.get('visual_emb') or item.get('imagebind_emb')
        visual_emb = torch.tensor(visual, dtype=torch.float32)

        text = item.get('text_emb')
        if text is not None:
            text_emb = torch.tensor(text, dtype=torch.float32)
        else:
            text_emb = torch.zeros(self.text_dim, dtype=torch.float32)

        quality = item.get('quality_scores') or item.get('aesthetic_score', {})
        aesthetic = quality.get('aesthetic', 5.0) / 10.0
        technical = quality.get('technical', 5.0) / 10.0

        ecr = item.get('ecr', 0.0) or 0.0

        result = {
            'visual_emb': visual_emb,
            'text_emb': text_emb,
            'quality_scores': torch.tensor([aesthetic, technical], dtype=torch.float32),
            'ecr': torch.tensor(ecr, dtype=torch.float32),
            'aesthetic': torch.tensor(aesthetic, dtype=torch.float32),
            'technical': torch.tensor(technical, dtype=torch.float32),
        }

        if 'teacher_ecr' in item:
            result['teacher_ecr'] = torch.tensor(item['teacher_ecr'], dtype=torch.float32)
        if 'teacher_hidden' in item:
            result['teacher_hidden'] = torch.tensor(item['teacher_hidden'], dtype=torch.float32)

        return result


def load_data(json_path, max_samples=None):
    """Load and validate extracted features."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        raw = json.load(f)

    data = list(raw.values()) if isinstance(raw, dict) else raw

    valid = []
    for item in data:
        emb = item.get('visual_emb') or item.get('imagebind_emb')
        if emb is not None and item.get('ecr') is not None:
            valid.append(item)

    if max_samples and len(valid) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(valid), max_samples, replace=False)
        valid = [valid[i] for i in sorted(indices)]

    ecrs = [d['ecr'] for d in valid]
    print(f"  Loaded {len(valid)} samples with ECR")
    print(f"  ECR: mean={np.mean(ecrs):.4f}, std={np.std(ecrs):.4f}, "
          f"min={np.min(ecrs):.4f}, max={np.max(ecrs):.4f}")

    has_text = sum(1 for d in valid if d.get('text_emb') is not None)
    print(f"  With text embeddings: {has_text}/{len(valid)}")

    return valid


def split_data(data, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(data)
    indices = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return [data[i] for i in train_idx], [data[i] for i in val_idx]


# ================================================================
# TRAINING FUNCTIONS
# ================================================================
def train_teacher_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        visual = batch['visual_emb'].to(device)
        text = batch['text_emb'].to(device)
        quality = batch['quality_scores'].to(device)
        ecr = batch['ecr'].to(device)

        optimizer.zero_grad()
        out = model(visual, text, quality, ecr_targets=ecr)
        out['loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += out['loss'].item()
        n += 1
    return total_loss / n


def train_student_epoch(model, loader, optimizer, device, use_kd=False, loss_weights=None):
    model.train()
    total_loss = 0
    loss_accum = {}
    n = 0
    for batch in loader:
        visual = batch['visual_emb'].to(device)
        text = batch['text_emb'].to(device)
        ecr = batch['ecr'].to(device)
        aesthetic = batch['aesthetic'].to(device)
        technical = batch['technical'].to(device)

        kwargs = {
            'visual_emb': visual, 'text_emb': text,
            'ecr_targets': ecr,
            'loss_weights': loss_weights,
        }

        if use_kd:
            kwargs['aesthetic_targets'] = aesthetic
            kwargs['technical_targets'] = technical
            if 'teacher_ecr' in batch:
                kwargs['teacher_ecr'] = batch['teacher_ecr'].to(device)
            if 'teacher_hidden' in batch:
                kwargs['teacher_hidden'] = batch['teacher_hidden'].to(device)

        optimizer.zero_grad()
        out = model(**kwargs)
        out['loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += out['loss'].item()
        for k, v in out['losses'].items():
            loss_accum[k] = loss_accum.get(k, 0) + v.item()
        n += 1

    metrics = {'loss': total_loss / n}
    for k, v in loss_accum.items():
        metrics[k] = v / n
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, device, model_type='teacher'):
    """Evaluate any model. Returns ECR predictions + metrics."""
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        visual = batch['visual_emb'].to(device)
        text = batch['text_emb'].to(device)
        ecr = batch['ecr']

        if model_type == 'teacher':
            quality = batch['quality_scores'].to(device)
            out = model(visual, text, quality)
        else:
            out = model(visual, text)

        all_pred.extend(out['predicted_ecr'].cpu().numpy())
        all_true.extend(ecr.numpy())

    pred = np.array(all_pred)
    true = np.array(all_true)

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    plcc = pearsonr(pred, true)[0] if len(pred) > 2 else 0
    srcc = spearmanr(pred, true).correlation if len(pred) > 2 else 0
    ktau = kendalltau(pred, true).correlation if len(pred) > 2 else 0

    return {
        'mse': mse, 'mae': mae,
        'plcc': plcc, 'srcc': srcc, 'ktau': ktau,
        'pred_mean': pred.mean(), 'pred_std': pred.std(),
    }


@torch.no_grad()
def generate_teacher_targets(teacher, data_list, device, batch_size=64):
    """Run teacher on all data → add teacher_ecr and teacher_hidden to each item."""
    teacher.eval()
    dataset = KDDataset(data_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_ecr, all_hidden = [], []
    for batch in loader:
        visual = batch['visual_emb'].to(device)
        text = batch['text_emb'].to(device)
        quality = batch['quality_scores'].to(device)
        out = teacher(visual, text, quality)
        all_ecr.append(out['predicted_ecr'].cpu().numpy())
        all_hidden.append(out['hidden'].cpu().numpy())

    ecrs = np.concatenate(all_ecr)
    hiddens = np.concatenate(all_hidden)

    for i, item in enumerate(data_list):
        item['teacher_ecr'] = float(ecrs[i])
        item['teacher_hidden'] = hiddens[i].tolist()

    print(f"  Generated teacher targets for {len(data_list)} samples")
    print(f"  Teacher ECR: mean={ecrs.mean():.4f}, std={ecrs.std():.4f}")
    return data_list


# ================================================================
# MAIN EXPERIMENT
# ================================================================
def run_experiment(args):
    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"  KNOWLEDGE DISTILLATION EXPERIMENT")
    print(f"  Device: {device} | Data: {args.data}")
    print(f"{'='*70}\n")

    # --- Load & split data ---
    data = load_data(args.data, max_samples=args.max_samples)
    train_data, val_data = split_data(data, val_ratio=0.2)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}\n")

    # ==========================================================
    # PHASE 1: Train Teacher Model
    # ==========================================================
    print(f"{'='*70}")
    print("  PHASE 1: Training Teacher Model")
    print(f"{'='*70}")

    teacher = TeacherModel(
        hidden_dim=args.teacher_hidden, n_blocks=args.teacher_blocks,
        dropout=args.dropout,
    ).to(device)
    t_total, t_train = count_params(teacher)
    print(f"  Teacher params: {t_total:,} ({t_train:,} trainable)")

    train_loader = DataLoader(KDDataset(train_data), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(KDDataset(val_data), batch_size=args.batch, shuffle=False)

    optimizer = AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.teacher_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(1, args.teacher_epochs + 1):
        train_loss = train_teacher_epoch(teacher, train_loader, optimizer, device)
        scheduler.step()

        if epoch % max(1, args.teacher_epochs // 10) == 0 or epoch == args.teacher_epochs:
            val_metrics = evaluate_model(teacher, val_loader, device, 'teacher')
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.5f} | "
                  f"val PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
                  f"MSE={val_metrics['mse']:.5f}")

            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                torch.save(teacher.state_dict(), os.path.join(args.save_dir, 'teacher_best.pth'))

    teacher_time = time.time() - t_start
    teacher.load_state_dict(torch.load(os.path.join(args.save_dir, 'teacher_best.pth'),
                                       map_location=device, weights_only=True))
    teacher_val = evaluate_model(teacher, val_loader, device, 'teacher')
    print(f"\n  Teacher BEST: PLCC={teacher_val['plcc']:.4f} SRCC={teacher_val['srcc']:.4f} "
          f"MSE={teacher_val['mse']:.5f} MAE={teacher_val['mae']:.4f} ({teacher_time:.0f}s)")

    # ==========================================================
    # PHASE 2: Generate Teacher Soft Targets
    # ==========================================================
    print(f"\n{'='*70}")
    print("  PHASE 2: Generating Teacher Soft Targets")
    print(f"{'='*70}")
    train_data = generate_teacher_targets(teacher, train_data, device, args.batch)
    val_data = generate_teacher_targets(teacher, val_data, device, args.batch)

    # ==========================================================
    # PHASE 3: Train Student Baseline (NO KD)
    # ==========================================================
    print(f"\n{'='*70}")
    print("  PHASE 3: Training Student Baseline (NO KD)")
    print(f"{'='*70}")

    baseline = StudentModel(
        hidden_dim=args.student_hidden,
        teacher_hidden_dim=args.teacher_hidden,
        dropout=args.dropout,
    ).to(device)
    s_total, s_train = count_params(baseline)
    print(f"  Student params: {s_total:,} ({s_train:,} trainable)")
    print(f"  Compression ratio: {t_total/s_total:.1f}x smaller than teacher")

    baseline_weights = {
        'ecr_hard': 1.0, 'ecr_soft': 0.0, 'kd_repr': 0.0,
        'aesthetic': 0.0, 'technical': 0.0,
    }

    train_loader = DataLoader(KDDataset(train_data), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(KDDataset(val_data), batch_size=args.batch, shuffle=False)

    optimizer = AdamW(baseline.parameters(), lr=args.student_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.student_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(1, args.student_epochs + 1):
        train_metrics = train_student_epoch(
            baseline, train_loader, optimizer, device,
            use_kd=False, loss_weights=baseline_weights
        )
        scheduler.step()

        if epoch % max(1, args.student_epochs // 10) == 0 or epoch == args.student_epochs:
            val_metrics = evaluate_model(baseline, val_loader, device, 'student')
            print(f"  Epoch {epoch:3d}: train_loss={train_metrics['loss']:.5f} | "
                  f"val PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
                  f"MSE={val_metrics['mse']:.5f}")

            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                torch.save(baseline.state_dict(), os.path.join(args.save_dir, 'student_baseline_best.pth'))

    baseline_time = time.time() - t_start
    baseline.load_state_dict(torch.load(os.path.join(args.save_dir, 'student_baseline_best.pth'),
                                        map_location=device, weights_only=True))
    baseline_val = evaluate_model(baseline, val_loader, device, 'student')
    print(f"\n  Baseline BEST: PLCC={baseline_val['plcc']:.4f} SRCC={baseline_val['srcc']:.4f} "
          f"MSE={baseline_val['mse']:.5f} MAE={baseline_val['mae']:.4f} ({baseline_time:.0f}s)")

    # ==========================================================
    # PHASE 4: Train Student + KD
    # ==========================================================
    print(f"\n{'='*70}")
    print("  PHASE 4: Training Student + Knowledge Distillation")
    print(f"{'='*70}")

    student_kd = StudentModel(
        hidden_dim=args.student_hidden,
        teacher_hidden_dim=args.teacher_hidden,
        dropout=args.dropout,
    ).to(device)

    kd_weights = {
        'ecr_hard': 1.0, 'ecr_soft': args.alpha, 'kd_repr': args.beta,
        'aesthetic': args.gamma, 'technical': args.delta,
    }
    print(f"  KD weights: α(soft)={args.alpha}, β(repr)={args.beta}, "
          f"γ(aes)={args.gamma}, δ(tech)={args.delta}")

    optimizer = AdamW(student_kd.parameters(), lr=args.student_lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.student_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(1, args.student_epochs + 1):
        train_metrics = train_student_epoch(
            student_kd, train_loader, optimizer, device,
            use_kd=True, loss_weights=kd_weights
        )
        scheduler.step()

        if epoch % max(1, args.student_epochs // 10) == 0 or epoch == args.student_epochs:
            val_metrics = evaluate_model(student_kd, val_loader, device, 'student')
            kd_losses = {k: f"{v:.4f}" for k, v in train_metrics.items() if k != 'loss'}
            print(f"  Epoch {epoch:3d}: loss={train_metrics['loss']:.5f} ({kd_losses}) | "
                  f"val PLCC={val_metrics['plcc']:.4f} SRCC={val_metrics['srcc']:.4f} "
                  f"MSE={val_metrics['mse']:.5f}")

            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                torch.save(student_kd.state_dict(), os.path.join(args.save_dir, 'student_kd_best.pth'))

    kd_time = time.time() - t_start
    student_kd.load_state_dict(torch.load(os.path.join(args.save_dir, 'student_kd_best.pth'),
                                          map_location=device, weights_only=True))
    kd_val = evaluate_model(student_kd, val_loader, device, 'student')
    print(f"\n  KD BEST: PLCC={kd_val['plcc']:.4f} SRCC={kd_val['srcc']:.4f} "
          f"MSE={kd_val['mse']:.5f} MAE={kd_val['mae']:.4f} ({kd_time:.0f}s)")

    # ==========================================================
    # PHASE 5: Comparison Report
    # ==========================================================
    print(f"\n{'='*70}")
    print("  COMPARISON REPORT")
    print(f"{'='*70}")

    print(f"\n  {'Model':<25} {'Params':>10} {'PLCC':>8} {'SRCC':>8} {'KTAU':>8} {'MSE':>10} {'MAE':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    rows = [
        ('Teacher (upper bound)', t_total, teacher_val),
        ('Student + KD', s_total, kd_val),
        ('Student baseline', s_total, baseline_val),
    ]
    for name, params, m in rows:
        print(f"  {name:<25} {params:>10,} {m['plcc']:>8.4f} {m['srcc']:>8.4f} "
              f"{m['ktau']:>8.4f} {m['mse']:>10.6f} {m['mae']:>8.4f}")

    print(f"\n  Compression: Teacher({t_total:,}) → Student({s_total:,}) = "
          f"{t_total/s_total:.1f}x reduction")

    kd_gain = kd_val['plcc'] - baseline_val['plcc']
    gap_to_teacher = teacher_val['plcc'] - kd_val['plcc']
    print(f"\n  KD Improvement (PLCC): Student+KD vs Baseline = {kd_gain:+.4f}")
    print(f"  Gap to Teacher (PLCC): {gap_to_teacher:+.4f}")

    if kd_gain > 0:
        print(f"\n  ✓ KD HELPS: Student+KD outperforms baseline by {kd_gain:.4f} PLCC")
        recovery = kd_gain / (teacher_val['plcc'] - baseline_val['plcc'] + 1e-8) * 100
        print(f"  ✓ Recovery rate: {recovery:.1f}% of teacher-baseline gap")
    else:
        print(f"\n  ✗ KD did not help in this configuration. Try adjusting weights or epochs.")

    report = {
        'data_path': args.data,
        'n_train': len(train_data), 'n_val': len(val_data),
        'teacher': {'params': t_total, **teacher_val},
        'student_kd': {'params': s_total, **kd_val, 'kd_weights': kd_weights},
        'student_baseline': {'params': s_total, **baseline_val},
        'kd_gain_plcc': kd_gain,
    }
    report_path = os.path.join(args.save_dir, 'experiment_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Report saved: {report_path}")
    print(f"  Models saved: {args.save_dir}/")
    print(f"{'='*70}\n")

    # ==========================================================
    # PHASE 6: Ablation Study
    # ==========================================================
    if args.ablation:
        print(f"\n{'='*70}")
        print("  PHASE 6: Ablation Study")
        print(f"{'='*70}")
        ablation_epochs = args.student_epochs // 2  # use half epochs for ablation variants
        ablation_results = run_ablation_study(
            train_data, val_data,
            epochs=ablation_epochs, lr=args.student_lr,
            batch_size=args.batch, device=device,
            pretrained_kd=torch.load(
                os.path.join(args.save_dir, 'student_kd_best.pth'),
                map_location=device, weights_only=True),
            pretrained_baseline=torch.load(
                os.path.join(args.save_dir, 'student_baseline_best.pth'),
                map_location=device, weights_only=True),
            pretrained_teacher=torch.load(
                os.path.join(args.save_dir, 'teacher_best.pth'),
                map_location=device, weights_only=True),
            save_dir=args.save_dir,
            hidden_dim=args.student_hidden,
            teacher_hidden=args.teacher_hidden,
        )
        report['ablation'] = ablation_results
        ablation_path = os.path.join(args.save_dir, 'ablation_report.json')
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2, default=float)
        print(f"  Ablation report saved: {ablation_path}")

    # ==========================================================
    # PHASE 7: Explainability
    # ==========================================================
    if args.explain:
        print(f"\n{'='*70}")
        print("  PHASE 7: Explainability Analysis")
        print(f"{'='*70}")
        engine = ExplainabilityEngine(student_kd, device)
        n_explain = min(args.explain_n, len(val_data))
        print(f"  Explaining {n_explain} validation samples ...")
        explanations = engine.explain_batch(val_data, max_samples=n_explain)

        for exp in explanations[:3]:  # print first 3
            engine.print_summary(exp)

        summary = summarise_explanations(explanations)
        print(f"\n  Dataset summary:")
        for k, v in summary.items():
            print(f"    {k:<35}: {v}")

        # Generate LLM prompt for the first sample
        print(f"\n  Sample LLM prompt (first video):")
        print(f"  {'─'*50}")
        print(engine.generate_llm_prompt(explanations[0]))

        explain_path = os.path.join(args.save_dir, 'explanations.json')
        with open(explain_path, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'explanations': explanations},
                      f, indent=2, ensure_ascii=False)
        report['explainability_summary'] = summary
        print(f"\n  Explanations saved: {explain_path}")

    # Re-save report (may include ablation + explainability)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)

    return report


def main():
    parser = argparse.ArgumentParser(description="KD Experiment for Distil-ShortVU")
    parser.add_argument('--data', required=True, help='Path to extracted features JSON')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or mps')
    parser.add_argument('--save-dir', default='results_kd', help='Output directory')

    parser.add_argument('--teacher-hidden', type=int, default=512)
    parser.add_argument('--teacher-blocks', type=int, default=4)
    parser.add_argument('--teacher-epochs', type=int, default=60)
    parser.add_argument('--teacher-lr', type=float, default=3e-4)

    parser.add_argument('--student-hidden', type=int, default=256)
    parser.add_argument('--student-epochs', type=int, default=80)
    parser.add_argument('--student-lr', type=float, default=5e-4)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=32)

    parser.add_argument('--alpha', type=float, default=0.5, help='Soft ECR loss weight')
    parser.add_argument('--beta', type=float, default=0.3, help='Repr distillation weight')
    parser.add_argument('--gamma', type=float, default=0.2, help='Aesthetic aux weight')
    parser.add_argument('--delta', type=float, default=0.2, help='Technical aux weight')

    parser.add_argument('--quick', action='store_true', help='Quick test (fewer epochs)')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study after main experiment')
    parser.add_argument('--explain', action='store_true',
                        help='Run explainability analysis after KD training')
    parser.add_argument('--explain-n', type=int, default=20,
                        help='Number of samples to explain (default: 20)')

    args = parser.parse_args()

    if args.quick:
        args.teacher_epochs = 15
        args.student_epochs = 20

    os.makedirs(args.save_dir, exist_ok=True)
    run_experiment(args)


if __name__ == '__main__':
    main()
