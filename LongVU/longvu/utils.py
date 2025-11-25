# pyre-unsafe
import torch
from transformers import AutoConfig


def get_torch_device():
    """Return the best available torch.device: cuda > mps > cpu."""
    # Prefer MPS on macOS (if available), then CUDA, else CPU.
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
