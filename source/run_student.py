import os
import argparse
import torch
from PIL import Image

def load_frame_from_video(video_path, idx=None, ctx=None):
    try:
        from decord import VideoReader, cpu
    except Exception as e:
        raise RuntimeError(f"Decord is required to read videos: {e}")

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError("Video has no frames")
    if idx is None:
        idx = total // 2
    frame = vr[idx].asnumpy()
    img = Image.fromarray(frame)
    return img


def main():
    parser = argparse.ArgumentParser(description='Run student model on a single video and output predictions')
    parser.add_argument('--video', required=True, help='Path to input video (mp4)')
    parser.add_argument('--checkpoint', default='checkpoints/student_epoch1.pth', help='Path to model checkpoint')
    parser.add_argument('--use-hf-llm', action='store_true', dest='use_hf_llm', help='If the student was trained with an HF LM inside, enable this')
    parser.add_argument('--llm-name', default='gpt2', help='LM name used during training (when --use-hf-llm)')
    parser.add_argument('--device', default=None, help='Torch device string (e.g. cpu,cuda,mps)')
    parser.add_argument('--save-json', default=None, help='Optional path to save a small JSON with predictions')

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        return

    # Lazy import of model and processors
    try:
        from student_model import ViralStudentModel
        from transformers import CLIPProcessor, AutoTokenizer
    except Exception as e:
        print(f"Error importing model or transformers: {e}")
        return

    # Prepare tokenizer/processor
    if args.use_hf_llm:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.get_vocab())
    else:
        # small dummy tokenizer for embedding size; we only need processor here
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.get_vocab())

    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # Instantiate model with same flags as training
    model = ViralStudentModel(vocab_size=vocab_size, d_model=512, use_hf_llm=args.use_hf_llm, llm_name=args.llm_name)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    # Load checkpoint
    try:
        state = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    # Read a representative frame
    try:
        img = load_frame_from_video(args.video)
    except Exception as e:
        print(f"Failed to read video frame: {e}")
        return

    # Preprocess
    inputs = processor(images=img, return_tensors='pt')
    pixel_values = inputs.pixel_values.to(device)

    # Run model (no text input) - produces predicted_ecr
    with torch.no_grad():
        out = model(input_ids=None, attention_mask=None, pixel_values=pixel_values)

    predicted = out.get('predicted_ecr')
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy().tolist()

    print('Predicted ECR:', predicted)

    # Optionally save to JSON
    if args.save_json:
        import json
        result = {'video': args.video, 'predicted_ecr': predicted}
        with open(args.save_json, 'w') as f:
            json.dump(result, f, indent=2)
        print('Saved predictions to', args.save_json)


if __name__ == '__main__':
    main()
