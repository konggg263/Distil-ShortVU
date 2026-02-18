"""
run_student_v2.py - Inference script for Student Model V2

Usage:
    # From precomputed embeddings (fast)
    python source/run_student_v2.py --checkpoint checkpoints_v2/student_v2_mlp_best.pth --data data/test.json
    
    # From raw video (requires ImageBind)
    python source/run_student_v2.py --checkpoint checkpoints_v2/student_v2_mlp_best.pth --video path/to/video.mp4

Author: Pipeline V2 Inference
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from student_model_v2 import StudentMLP, StudentTransformer, ViralStudentV2


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    model_type = args.get('model', 'mlp')
    
    print(f"Model type: {model_type}")
    
    # Reconstruct model
    if model_type == 'mlp':
        model = StudentMLP(
            hidden_dim=args.get('hidden_dim', 512),
            dropout=args.get('dropout', 0.1),
        )
    elif model_type == 'transformer':
        model = StudentTransformer(
            hidden_dim=args.get('hidden_dim', 512),
            n_heads=args.get('n_heads', 8),
            n_layers=args.get('n_layers', 4),
            dropout=args.get('dropout', 0.1),
        )
    else:
        model = ViralStudentV2(
            hidden_dim=args.get('hidden_dim', 768),
            n_layers=args.get('n_layers', 6),
            n_heads=args.get('n_heads', 12),
            dropout=args.get('dropout', 0.1),
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded from epoch {epoch}")
    
    return model, args


def get_embedding_from_video(video_path, device):
    """Extract ImageBind embedding from video"""
    from pipeline_fast import FastImageBind
    
    imagebind = FastImageBind()
    emb = imagebind.embed(video_path, num_frames=3)
    imagebind.unload()
    
    if emb is not None:
        return torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
    return None


def predict_single(model, imagebind_emb):
    """Run prediction for a single sample"""
    with torch.no_grad():
        outputs = model(imagebind_emb=imagebind_emb)
    
    return {
        'predicted_ecr': outputs['predicted_ecr'].item(),
        'predicted_aesthetic': outputs.get('predicted_aesthetic', torch.tensor(0)).item() * 10,  # Scale back to 0-10
        'predicted_technical': outputs.get('predicted_technical', torch.tensor(0)).item() * 10,
    }


def predict_batch(model, data_path, device, output_path=None):
    """Run predictions on a JSON dataset"""
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data_list = list(data.values())
    else:
        data_list = data
    
    results = []
    
    for item in tqdm(data_list, desc="Predicting"):
        if item.get('imagebind_emb') is None:
            continue
        
        emb = torch.tensor(item['imagebind_emb'], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(imagebind_emb=emb)
        
        result = {
            'video_path': item.get('video_path', ''),
            'true_ecr': item.get('ecr'),
            'true_aesthetic': item.get('aesthetic_score', {}).get('aesthetic', 0),
            'true_technical': item.get('aesthetic_score', {}).get('technical', 0),
            'predicted_ecr': outputs['predicted_ecr'].item(),
            'predicted_aesthetic': outputs.get('predicted_aesthetic', torch.tensor(0)).item() * 10,
            'predicted_technical': outputs.get('predicted_technical', torch.tensor(0)).item() * 10,
        }
        results.append(result)
    
    # Compute metrics (only when true ECR is available)
    if results:
        has_ecr = [r for r in results if r['true_ecr'] is not None]
        
        if has_ecr:
            true_ecr = np.array([r['true_ecr'] for r in has_ecr])
            pred_ecr = np.array([r['predicted_ecr'] for r in has_ecr])
            
            ecr_mse = np.mean((pred_ecr - true_ecr) ** 2)
            ecr_mae = np.mean(np.abs(pred_ecr - true_ecr))
            ecr_corr = np.corrcoef(pred_ecr, true_ecr)[0, 1] if len(pred_ecr) > 1 else 0
            
            print(f"\n{'='*50}")
            print("Evaluation Metrics")
            print(f"{'='*50}")
            print(f"Samples: {len(has_ecr)}")
            print(f"ECR MSE: {ecr_mse:.6f}")
            print(f"ECR MAE: {ecr_mae:.6f}")
            print(f"ECR Correlation: {ecr_corr:.4f}")
            
            # Aesthetic metrics
            true_aes = np.array([r['true_aesthetic'] for r in has_ecr])
            pred_aes = np.array([r['predicted_aesthetic'] for r in has_ecr])
            if true_aes.std() > 0:
                aes_corr = np.corrcoef(pred_aes, true_aes)[0, 1]
                print(f"Aesthetic Correlation: {aes_corr:.4f}")
        else:
            pred_ecr = np.array([r['predicted_ecr'] for r in results])
            print(f"\n{'='*50}")
            print("Prediction Summary (no ground truth ECR)")
            print(f"{'='*50}")
            print(f"Samples: {len(results)}")
            print(f"Predicted ECR: mean={pred_ecr.mean():.4f}, std={pred_ecr.std():.4f}, "
                  f"min={pred_ecr.min():.4f}, max={pred_ecr.max():.4f}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Student Model V2 Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--video', type=str, default=None, help='Single video path')
    parser.add_argument('--data', type=str, default=None, help='JSON data file with embeddings')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--device', type=str, default='auto')
    
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
    
    print(f"Device: {device}")
    
    # Load model
    model, model_args = load_model(args.checkpoint, device)
    
    # Run inference
    if args.video:
        print(f"\nProcessing video: {args.video}")
        emb = get_embedding_from_video(args.video, device)
        if emb is not None:
            result = predict_single(model, emb)
            print(f"\nPredictions:")
            print(f"  ECR: {result['predicted_ecr']:.4f}")
            print(f"  Aesthetic: {result['predicted_aesthetic']:.2f}/10")
            print(f"  Technical: {result['predicted_technical']:.2f}/10")
        else:
            print("Failed to extract embedding from video")
    
    elif args.data:
        results = predict_batch(model, args.data, device, args.output)
    
    else:
        print("Error: Provide either --video or --data")
        return


if __name__ == "__main__":
    main()
