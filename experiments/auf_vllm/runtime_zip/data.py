"""Read-only weight lookup used by the archived runtime attachment checks."""
import json
from safetensors import safe_open
from paths import model


def tensor(root, size, key):
    path = model(size)
    index = path/"model.safetensors.index.json"
    file = path/json.loads(index.read_text())["weight_map"][key] if index.exists() else path/"model.safetensors"
    with safe_open(file, framework="pt", device="cpu") as reader:
        return reader.get_tensor(key)
