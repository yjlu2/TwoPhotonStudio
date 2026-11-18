"""Branch a combined import without modifying the source analysis."""
import json
import os
from pathlib import Path
import numpy as np
import backend as B


def create_extended_session(parent_session,config,dirty,paths,destination,log=print):
    original=Path(parent_session['root'])
    source=json.loads((original/'source.json').read_text(encoding='utf-8'))
    old=source['files']
    normal=lambda p:os.path.normcase(str(Path(p).resolve()))
    if [normal(p) for p in paths[:len(old)]]!=[normal(p) for p in old]:
        raise ValueError('Keep existing source stacks in their original order; append new stacks at the end.')
    if len(paths)<=len(old): raise ValueError('Select at least one new TIFF stack to append.')
    if len({normal(p) for p in paths})!=len(paths): raise ValueError('A TIFF file is already in this session.')
    for record in source.get('records',[]):
        info=Path(record['path']).stat()
        if info.st_size!=record['bytes'] or info.st_mtime_ns!=record['mtime_ns']:
            raise ValueError('Source TIFF changed since import: '+record['path'])
    root=Path(destination)
    log('Creating an extended session. The parent analysis will not be modified.')
    meta=B.import_tiffs(paths,root,log,preserve_order=True)
    if list(meta['shape'][1:])!=list(source['shape'][1:]):
        raise ValueError('FOV dimensions differ from the original session.')
    meta['parent_session']=str(original)
    meta['appended_files']=meta['files'][len(old):]
    B.write_json(root/'source.json',meta)
    np.save(root/'parent_session.npy',dict(session=parent_session,cfg=config,dirty=dirty))
    session=dict(root=str(root),ops_file='',stat=[],result='',frame_times=None,sync=None)
    np.save(root/'session.npy',dict(session=session,cfg=config,dirty=True))
    B.write_json(root/'extension.json',dict(
        parent_root=str(original),original_frames=source['shape'][0],total_frames=meta['shape'][0],
        appended_files=meta['appended_files'],parent_snapshot='parent_session.npy',
        requires=['registration','ROI detection','trace extraction','synchronization import']))
    log(f'Extended import ready: {len(meta["files"])} stacks / {meta["shape"][0]} frames.')
    log('Original ROI edits, traces and synchronization are preserved in parent_session.npy.')
    log('The extended session needs new registration, ROI detection, traces and synchronization.')
    return meta
