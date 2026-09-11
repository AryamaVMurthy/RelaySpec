# Inference-only: expose this directory on PYTHONPATH for vLLM child workers.
import os
if os.environ.get('AUF_SAFE_RUNTIME')=='1':
 import lora_draft_sharing
 lora_draft_sharing.install()
