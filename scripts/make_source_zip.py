"""Build a source-only archive; never includes installed environments or analysis results."""
from pathlib import Path
import os
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OMIT={'.git','.local','__pycache__','dist','build','.pytest_cache'}

def main():
    destination=ROOT/'dist/TwoPhotonStudio-0.0.1-source.zip'
    destination.parent.mkdir(exist_ok=True)
    count=0
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for folder,dirs,files in os.walk(ROOT):
            dirs[:]=sorted(d for d in dirs if d not in OMIT and not d.endswith('.egg-info'))
            for name in sorted(files):
                path=Path(folder)/name; rel=path.relative_to(ROOT)
                if path.suffix in ('.pyc','.pyo','.lnk','.tif','.tiff','.mat','.npy','.npz','.bin'): continue
                if path.name in ('preferences.ini','runtime.json'): continue
                if path.stat().st_size>95*1024*1024: raise RuntimeError('Unexpected large source file: '+str(rel))
                archive.write(path,'TwoPhotonStudio/'+rel.as_posix()); count+=1
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(digest+'  '+destination.name+'\n')
    print(str(destination)); print(str(count)+' files, '+str(destination.stat().st_size)+' bytes'); print(digest)

if __name__=='__main__': main()
