"""
extract_features.py - Extract multimodal features for KD experiment

Extracts from teacher ensemble:
  - visual_emb (1024-d) from ImageBind
  - text_emb (384-d) from MiniLM (title + description + BLIP caption)
  - quality_scores (aesthetic, technical) from MUSIQ + TOPIQ

Usage:
    # Extract features for 500 videos (repo layout)
    python source/kd/extract_features.py \\
        --csv data/train_data.csv \\
        --videos data/train_videos \\
        --out source/kaggle_kd/results/500_videos/features_500.json \\
        --max 500

    # Skip text embedding (if MiniLM not available)
    python source/kd/extract_features.py \\
        --csv data/train_data.csv \\
        --videos data/train_videos \\
        --out source/kaggle_kd/results/500_videos/features_500.json \\
        --max 500 --skip-text

    # Download videos first, then extract
    python source/kd/extract_features.py \\
        --csv data/train_data.csv \\
        --videos data/train_videos \\
        --out source/kaggle_kd/results/500_videos/features_500.json \\
        --max 500 --download
"""

import os
import sys
import json
import argparse
import time
import gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third-party", "ImageBind"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source"))

DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

CPU_DEVICE = "cpu"


class QualityScorer:
    """MUSIQ (aesthetic) + TOPIQ (technical) scorer."""

    def __init__(self):
        import pyiqa
        self.device = torch.device(CPU_DEVICE if DEVICE == "mps" else DEVICE)
        self.musiq = pyiqa.create_metric('musiq', device=self.device)
        self.topiq = pyiqa.create_metric('topiq_nr', device=self.device)
        print(f"[QualityScorer] loaded on {self.device}")

    def score(self, video_path, num_frames=4):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            if total == 0:
                return {"aesthetic": 5.0, "technical": 5.0}

            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = vr.get_batch(indices)
            frames = frames.asnumpy() if hasattr(frames, 'asnumpy') else frames.numpy()

            aes_scores, tech_scores = [], []
            for f in frames:
                t = torch.from_numpy(f).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                t = t.to(self.device)
                with torch.no_grad():
                    aes_scores.append(self.musiq(t).item())
                    tech_scores.append(self.topiq(t).item())

            aes = np.mean(aes_scores) / 10.0
            tech = np.mean(tech_scores) * 10.0
            return {
                "aesthetic": round(max(0, min(10, aes)), 2),
                "technical": round(max(0, min(10, tech)), 2),
            }
        except Exception as e:
            print(f"  [Scorer] Error: {e}")
            return {"aesthetic": 5.0, "technical": 5.0}

    def unload(self):
        del self.musiq, self.topiq
        gc.collect()


class VisualEncoder:
    """ImageBind visual embedding (1024-d)."""

    def __init__(self):
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        self.ModalityType = ModalityType
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.device = CPU_DEVICE if DEVICE == "mps" else DEVICE
        self.model = self.model.to(self.device).eval()
        print(f"[VisualEncoder] ImageBind loaded on {self.device}")

    def embed(self, video_path, num_frames=4):
        try:
            from decord import VideoReader, cpu
            from torchvision import transforms

            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            if total == 0:
                return None

            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = vr.get_batch(indices)
            frames = frames.asnumpy() if hasattr(frames, 'asnumpy') else frames.numpy()

            preprocess = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])

            tensors = [preprocess(Image.fromarray(f)) for f in frames]
            video_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embs = self.model({self.ModalityType.VISION: video_tensor})

            emb = embs[self.ModalityType.VISION].cpu().numpy().flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb.tolist()
        except Exception as e:
            print(f"  [VisualEncoder] Error: {e}")
            return None

    def unload(self):
        del self.model
        gc.collect()


class CaptionGenerator:
    """BLIP caption from center frame."""

    def __init__(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)
        self.dev = CPU_DEVICE if DEVICE == "mps" else DEVICE
        self.model = self.model.to(self.dev).eval()
        print(f"[CaptionGenerator] BLIP loaded on {self.dev}")

    def caption(self, video_path):
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            if total == 0:
                return ""
            frame = vr[total // 2].asnumpy()
            image = Image.fromarray(frame)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.dev) for k, v in inputs.items()}
            with torch.no_grad():
                ids = self.model.generate(**inputs, max_new_tokens=40)
            return self.processor.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  [Caption] Error: {e}")
            return ""

    def unload(self):
        del self.model, self.processor
        gc.collect()


class TextEncoder:
    """MiniLM sentence embedding (384-d) from title + description + caption."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[TextEncoder] MiniLM loaded")

    def encode(self, title="", description="", caption=""):
        parts = [p for p in [title, description, caption] if p.strip()]
        if not parts:
            return [0.0] * 384
        text = " | ".join(parts)
        emb = self.model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def unload(self):
        del self.model
        gc.collect()


def download_videos(csv_path, video_dir, max_videos=500):
    """Download videos from CSV URLs."""
    import urllib.request

    df = pd.read_csv(csv_path)
    os.makedirs(video_dir, exist_ok=True)

    has_ecr = 'ECR' in df.columns
    if has_ecr:
        df = df[df['ECR'].notna()]

    work = []
    for _, row in df.iterrows():
        vid_id = str(row['Id'])
        vid_path = os.path.join(video_dir, f"{vid_id}.mp4")
        if not os.path.exists(vid_path) and pd.notna(row.get('Download_link', '')):
            work.append((vid_id, str(row['Download_link']), vid_path))

    work = work[:max_videos]
    print(f"Downloading {len(work)} videos to {video_dir}...")

    downloaded, errors = 0, 0
    for vid_id, url, path in tqdm(work, desc="Downloading"):
        try:
            urllib.request.urlretrieve(url, path)
            downloaded += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                tqdm.write(f"  Error {vid_id}: {e}")

    print(f"  Downloaded: {downloaded}, Errors: {errors}")
    return downloaded


def extract_features(csv_path, video_dir, output_path, max_videos=500,
                     skip_text=False, save_every=50):
    """Extract features for all videos."""
    print(f"\n{'='*60}")
    print(f"Feature Extraction | Device: {DEVICE}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    has_ecr = 'ECR' in df.columns

    existing = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for item in json.load(f):
                vid = item.get('video_id', '')
                if vid and item.get('visual_emb') is not None:
                    existing[vid] = item
        print(f"Resuming: {len(existing)} already processed")

    work = []
    for _, row in df.iterrows():
        vid_id = str(row['Id'])
        vid_path = os.path.join(video_dir, f"{vid_id}.mp4")
        if vid_id not in existing and os.path.exists(vid_path):
            ecr = float(row['ECR']) if has_ecr and pd.notna(row.get('ECR')) else None
            if ecr is not None:
                work.append({
                    'id': vid_id, 'path': vid_path, 'ecr': ecr,
                    'title': str(row.get('Title', '')) if pd.notna(row.get('Title', '')) else '',
                    'description': str(row.get('Description', '')) if pd.notna(row.get('Description', '')) else '',
                })

    work = work[:max_videos]
    print(f"To process: {len(work)} videos")

    if not work and not existing:
        print("No videos found! Run with --download first.")
        return

    print("\nLoading models (this takes a minute)...")
    scorer = QualityScorer()
    captioner = CaptionGenerator()
    visual_enc = VisualEncoder()
    text_enc = TextEncoder() if not skip_text else None

    results = list(existing.values())
    new_count, errors = 0, 0
    t_start = time.time()

    pbar = tqdm(work, desc="Extracting", unit="video")
    for item in pbar:
        try:
            quality = scorer.score(item['path'])
            caption = captioner.caption(item['path'])
            visual_emb = visual_enc.embed(item['path'])

            text_emb = None
            if text_enc:
                text_emb = text_enc.encode(item['title'], item['description'], caption)

            result = {
                'video_id': item['id'], 'video_path': item['path'],
                'ecr': item['ecr'],
                'title': item['title'], 'description': item['description'],
                'caption': caption,
                'visual_emb': visual_emb, 'text_emb': text_emb,
                'quality_scores': quality,
            }
            results.append(result)
            new_count += 1

            elapsed = time.time() - t_start
            speed = new_count / elapsed if elapsed > 0 else 0
            eta = (len(work) - new_count) / speed / 3600 if speed > 0 else 0
            pbar.set_postfix({'new': new_count, 'spd': f'{speed:.1f}v/s', 'eta': f'{eta:.1f}h'})

            if new_count % save_every == 0:
                _save(results, output_path)
        except Exception as e:
            errors += 1
            tqdm.write(f"  Error {item['id']}: {e}")

    _save(results, output_path)

    scorer.unload()
    captioner.unload()
    visual_enc.unload()
    if text_enc:
        text_enc.unload()

    elapsed = time.time() - t_start
    print(f"\nDone! {new_count} videos in {elapsed/60:.1f} min")
    print(f"Total samples: {len(results)} | Errors: {errors}")
    print(f"Output: {output_path}")


def _save(results, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description="Extract features for KD experiment")
    parser.add_argument('--csv', default='data/train_data.csv')
    parser.add_argument('--videos', default='data/train_videos')
    parser.add_argument(
        '--out',
        default='source/kaggle_kd/results/500_videos/features_500.json',
        help='Output JSON path (parent dirs are created)',
    )
    parser.add_argument('--max', type=int, default=500)
    parser.add_argument('--download', action='store_true', help='Download videos first')
    parser.add_argument('--skip-text', action='store_true', help='Skip MiniLM text encoding')
    parser.add_argument('--save-every', type=int, default=50)
    args = parser.parse_args()

    if args.download:
        download_videos(args.csv, args.videos, max_videos=args.max)

    extract_features(
        args.csv, args.videos, args.out,
        max_videos=args.max, skip_text=args.skip_text,
        save_every=args.save_every,
    )


if __name__ == '__main__':
    main()
