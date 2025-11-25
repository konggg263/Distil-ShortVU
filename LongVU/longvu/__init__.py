# Ensure MPS -> CPU fallback for unimplemented ops is enabled by default when
# importing the longvu package. This sets the env var before torch is imported
# elsewhere. Users can still override this externally if desired.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# pyre-unsafe
from .language_model.cambrian_qwen import CambrianQwenModel
from .language_model.cambrian_llama import CambrianLlamaModel
