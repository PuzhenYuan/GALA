"""Resolve portable local or Hugging Face inference checkpoints."""
import json
import os
from pathlib import Path


def resolve_checkpoint(source: str) -> str:
    path = Path(source).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if source.startswith('hf://'):
        source = source[5:]
    if 'REPLACE_WITH' in source:
        raise ValueError('Set MODEL_PATH to a local checkpoint or ypz21/GALA_robocasa_gr1.')
    from huggingface_hub import snapshot_download
    return snapshot_download(source, revision=os.environ.get('GALA_MODEL_REVISION'))


def load_config(model_path: str):
    from gala.model.base import GALAConfig
    root = Path(model_path)
    with (root / 'config.json').open() as handle:
        data = json.load(handle)
    if data.get('model_type') != 'gala' or 'bridge_cfg' not in data:
        raise ValueError('Expected a GALA inference checkpoint with model_type=gala and bridge_cfg.')
    backbone = os.environ.get('GALA_BACKBONE_PATH', data['backbone_cfg']['eagle_path'])
    candidate = root / backbone
    if candidate.is_dir():
        backbone = str(candidate.resolve())
    data['backbone_cfg']['eagle_path'] = backbone
    return GALAConfig(**data)
