"""Install explicit study plugins in vLLM's spawned worker processes too."""
import os
if os.environ.get("AUF_CAPTURE") == "1":
    from experiments.auf_vllm.capture_runtime import install
    install()
