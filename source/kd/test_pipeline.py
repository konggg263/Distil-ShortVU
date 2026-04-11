"""Quick test: generate synthetic data and run the KD experiment to verify pipeline works."""

import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def generate_synthetic_data(n=200, output_path='data/test_synthetic.json'):
    """Generate synthetic features that mimic real extracted data."""
    np.random.seed(42)
    data = []
    for i in range(n):
        aesthetic = np.random.uniform(2, 9)
        technical = np.random.uniform(2, 9)
        visual_signal = np.random.randn(1024).astype(float)
        text_signal = np.random.randn(384).astype(float)

        ecr = 0.3 * (aesthetic / 10) + 0.2 * (technical / 10) + \
              0.3 * float(np.tanh(visual_signal[:5].mean())) + \
              0.2 * float(np.tanh(text_signal[:3].mean()))
        ecr = max(0.0, min(1.0, ecr + np.random.normal(0, 0.05)))

        data.append({
            'video_id': f'synthetic_{i:04d}',
            'ecr': round(ecr, 4),
            'visual_emb': (visual_signal / (np.linalg.norm(visual_signal) + 1e-8)).tolist(),
            'text_emb': (text_signal / (np.linalg.norm(text_signal) + 1e-8)).tolist(),
            'quality_scores': {
                'aesthetic': round(aesthetic, 2),
                'technical': round(technical, 2),
            },
            'caption': f'synthetic caption {i}',
            'title': f'title {i}',
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Generated {n} synthetic samples → {output_path}")
    return output_path


if __name__ == '__main__':
    path = generate_synthetic_data(200, 'data/test_synthetic.json')
    print(f"\nNow run:")
    print(f"  python source/kd/run_experiment.py --data {path} --quick --save-dir results_kd_test")
