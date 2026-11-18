"""Transactional copy of an application session to another output folder."""
import hashlib
import json
import ntpath
import os
from pathlib import Path
import shutil
import stat
import uuid
from datetime import datetime
import numpy as np


def _inside(path,root):
    try:
        return os.path.commonpath([str(path),str(root)])==str(root)
    except ValueError:
        return False


def _is_link(path):
    info=path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)


def _path_rebaser(original,destination):
    prefixes=[str(Path(original).absolute()),str(Path(original).resolve())]
    pairs=[(ntpath.normcase(ntpath.normpath(p)),p) for p in prefixes]
    def remap(value):
        if isinstance(value,Path):
            changed=remap(str(value))
            return Path(changed)
        if isinstance(value,str):
            normal=ntpath.normcase(ntpath.normpath(value))
            for prefix,_ in pairs:
                if normal==prefix: return str(destination)
                if normal.startswith(prefix+'\\'):
                    # Normalization only establishes path boundaries. Preserve case of the suffix.
                    suffix=ntpath.normpath(value)[len(prefix)+1:]
                    return str(Path(destination)/Path(suffix))
            return value
        if isinstance(value,bytes):
            try: text=value.decode('utf-8')
            except UnicodeDecodeError: return value
            mapped=remap(text)
            return mapped.encode('utf-8') if mapped!=text else value
        if isinstance(value,dict): return {key:remap(item) for key,item in value.items()}
        if isinstance(value,list): return [remap(item) for item in value]
        if isinstance(value,tuple): return tuple(remap(item) for item in value)
        if isinstance(value,np.ndarray) and value.dtype.kind=='O':
            result=np.empty(value.shape,dtype=object)
            for index in np.ndindex(value.shape): result[index]=remap(value[index])
            return result
        if isinstance(value,np.ndarray) and value.dtype.kind in ('U','S'):
            if not value.size: return value
            return np.asarray([remap(item.item()) for item in value.flat]).reshape(value.shape)
        return value
    return remap


def cleanup_owned_copy(path,token):
    """Only remove an unfinished directory created and marked by this operation."""
    path=Path(path)
    if path.name!='.twophoton-output-'+token or len(token)!=32:
        raise ValueError('Unrecognized output-copy staging directory.')
    if not path.exists(): return
    if _is_link(path): raise ValueError('Output-copy staging path is a link.')
    marker=path/'.copy-owner'
    if not marker.is_file() or marker.read_text(encoding='ascii')!=token:
        raise ValueError('Output-copy staging ownership check failed.')
    shutil.rmtree(path)


def _metadata_files(folder,remap):
    for path in folder.rglob('*'):
        if not path.is_file(): continue
        if path.suffix.lower()=='.json':
            with path.open(encoding='utf-8-sig') as handle: data=json.load(handle)
            data=remap(data)
            with path.open('w',encoding='utf-8') as handle: json.dump(data,handle,ensure_ascii=False,indent=2)
        elif path.suffix.lower()=='.npy':
            # Inspect only the header for large numeric movies/trace arrays.
            with path.open('rb') as handle:
                version=np.lib.format.read_magic(handle)
                _,_,dtype=np.lib.format._read_array_header(handle,version)
            if dtype.hasobject or dtype.kind in ('U','S'):
                data=np.load(path,allow_pickle=True)
                if dtype.kind not in ('O','U','S'):
                    raise ValueError('Structured object-array metadata is not supported for output relocation.')
                updated=remap(data)
                with path.open('wb') as handle: np.save(handle,updated)


def _copy_verified(source,target,log):
    before=source.stat(); digest=hashlib.sha256()
    with source.open('rb') as reader,target.open('xb') as writer:
        for block in iter(lambda:reader.read(8*1024*1024),b''):
            writer.write(block); digest.update(block)
    after=source.stat()
    if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
        raise ValueError('Source TIFF changed or an analysis file changed while copying: '+str(source))
    verification=hashlib.sha256()
    with target.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): verification.update(block)
    if verification.digest()!=digest.digest():
        raise IOError('Output-copy checksum verification failed: '+str(source))
    shutil.copystat(source,target)
    log('Copied and verified: '+str(source.name))


def relocate_output(snapshot,destination,log=print):
    """Create a verified copy and rebase application metadata; keep source untouched."""
    original=snapshot['session']['root']
    source=Path(original).resolve()
    target=Path(destination).resolve()
    if not (source/'source.json').is_file(): raise FileNotFoundError('Session source.json not found: '+str(source))
    if (source/'.copy-owner').exists():
        raise ValueError('Output relocation cannot use an unfinished output copy as its source.')
    if _inside(target,source) or _inside(source,target):
        raise ValueError('Output folder must be outside the current session and must not be its parent.')
    if target.exists(): raise FileExistsError('Output destination already exists: '+str(target))
    if not target.parent.is_dir(): raise FileNotFoundError('Output parent folder not found: '+str(target.parent))
    files=[]; directories=[]
    for parent,dirs,names in os.walk(source,followlinks=False):
        base=Path(parent)
        for name in dirs+names:
            path=base/name
            if _is_link(path): raise ValueError('Output relocation cannot copy linked entries: '+str(path))
        directories.extend((base/name).relative_to(source) for name in dirs)
        files.extend(base/name for name in names)
    required=sum(path.stat().st_size for path in files)
    if shutil.disk_usage(target.parent).free<required+max(1024*1024,int(required*.03)):
        raise OSError(28,'Not enough space for the output copy on '+str(target.parent))
    token=uuid.uuid4().hex
    staging=target.parent/('.twophoton-output-'+token)
    staging.mkdir()
    (staging/'.copy-owner').write_text(token,encoding='ascii')
    if hasattr(log,'temporary_tree'): log.temporary_tree(staging,token)
    committed=False
    try:
        log(f'Copying {len(files)} files ({required/1024**3:.2f} GiB) to {target}')
        for relative in directories: (staging/relative).mkdir(parents=True,exist_ok=True)
        for index,path in enumerate(files,1):
            _copy_verified(path,staging/path.relative_to(source),log)
            if index%20==0: log(f'Output copy: {index}/{len(files)} files')
        remap=_path_rebaser(original,target)
        _metadata_files(staging,remap)
        updated=remap(snapshot)
        updated['session']['root']=str(target)
        with (staging/'session.npy').open('wb') as handle: np.save(handle,updated)
        for key in ('ops_file','result'):
            value=updated['session'].get(key)
            if not value: continue
            location=Path(value)
            check=staging/location.relative_to(target) if _inside(location,target) else location
            if not check.exists(): raise FileNotFoundError(f'Copied session {key} path is unavailable: {value}')
        with (staging/'output_relocation.json').open('w',encoding='utf-8') as handle:
            json.dump(dict(original_output=str(source),output=str(target),date=datetime.now().isoformat(),
                           original_retained=True,exports_copied_as_is=True,copied_bytes=required),
                      handle,indent=2,ensure_ascii=False)
        # Atomic commit on the destination disk; only completed copies become sessions.
        staging.rename(target)
        committed=True
        try: (target/'.copy-owner').unlink(missing_ok=True)
        except OSError: log('Output copied; temporary ownership marker could not be removed.')
        log('Output folder changed: '+str(target))
        log('Original output retained: '+str(source))
        return updated
    finally:
        if not committed: cleanup_owned_copy(staging,token)
