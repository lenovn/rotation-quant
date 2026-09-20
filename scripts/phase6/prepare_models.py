"""Download only missing official model files, recording the resolved revision."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
MODELS = [('Qwen/Qwen3-0.6B', 'qwen3-0.6b'), ('meta-llama/Llama-3.2-3B-Instruct', 'llama-3.2-3b-instruct')]

def fetch(pair):
    repo, name = pair
    directory = ROOT / 'cache/models' / name
    if (directory / 'phase6_revision.json').exists():
        return
    revision = HfApi().model_info(repo).sha
    print(repo, revision, flush=True)
    snapshot_download(repo, revision=revision, local_dir=directory,
                      allow_patterns=['*.json', '*.safetensors', '*.model', 'merges.txt', 'vocab.json', 'LICENSE*', 'README.md'], max_workers=2)
    (directory / 'phase6_revision.json').write_text(json.dumps(dict(repo=repo, revision=revision), indent=2)+'\n')
    print('completed', directory, flush=True)

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, MODELS))
