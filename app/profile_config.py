"""Shared PROFILE preflight checks; no model imports or writes."""
import json
from pathlib import Path

def resolve_profile_config(cfg):
    value=cfg.get('profile_config','').strip()
    if not value:
        raise ValueError('PROFILE model is not configured. Select a matching 2D JSON config in Settings > Models and environments, or select PROFILE and click Detect to choose one.')
    config_path=Path(value).expanduser()
    if not config_path.is_file():
        raise ValueError(f'PROFILE config does not exist:\n{config_path}')
    try:
        c=json.loads(config_path.read_text(encoding='utf-8-sig'))
    except (ValueError,OSError) as e:
        raise ValueError(f'Cannot read PROFILE JSON config:\n{config_path}\n{e}') from e
    if not isinstance(c,dict) or c.get('ndim')!=2:
        raise ValueError('Select a PROFILE 2D model config (ndim = 2).')
    root=Path(cfg.get('profile_root',''))
    if not (root/'profile_inference/inference.py').is_file():
        raise ValueError(f'PROFILE repository is missing profile_inference/inference.py:\n{root}')
    checkpoint=cfg.get('profile_checkpoint','').strip() or c.get('checkpoint','')
    if not checkpoint:
        raise ValueError('PROFILE config has no checkpoint. Select its matching checkpoint in Settings.')
    checkpoint=Path(checkpoint).expanduser()
    if not checkpoint.is_absolute(): checkpoint=root/checkpoint
    if not checkpoint.is_file():
        raise ValueError(f'PROFILE checkpoint does not exist:\n{checkpoint}\nConfig: {config_path.name}\nSelect the matching checkpoint in Settings. Do not substitute a similarly named model.')
    python=Path(cfg.get('profile_python',''))
    if not python.is_file():
        raise ValueError(f'PROFILE Python executable does not exist:\n{python}')
    c['checkpoint']=str(checkpoint.resolve())
    return c
