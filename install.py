"""Conda bootstrap. Only the Python standard library is needed in the calling environment."""
import argparse
from datetime import datetime
import hashlib
import json
import os
import queue
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'.local'
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8',errors='replace')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def run(args, env=None):
    args=[str(x) for x in args]
    if args[1:4]==['-m','pip','install']:
        args[4:4]=['--no-compile','--no-warn-script-location']
    print('\n> '+subprocess.list2cmdline(args),flush=True)
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'install.log').open('a',encoding='utf-8') as log:
        log.write('\n'+datetime.now().isoformat()+' > '+subprocess.list2cmdline(args)+'\n'); log.flush()
        process=subprocess.Popen(args,env=env,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        lines=queue.Queue()
        def read_output():
            try:
                for line in process.stdout: lines.put(line)
            finally: lines.put(None)
        threading.Thread(target=read_output,daemon=True).start()
        started=time.monotonic()
        try:
            while True:
                try: line=lines.get(timeout=30)
                except queue.Empty:
                    elapsed=int(time.monotonic()-started)
                    line='[Installer] Step still running (%d min %02d s); please wait.\n'%(elapsed//60,elapsed%60)
                if line is None: break
                print(line,end='',flush=True); log.write(line); log.flush()
            code=process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
    if code: raise subprocess.CalledProcessError(code,args)

def conda_executable():
    candidates=[os.environ.get('CONDA_EXE'),shutil.which('conda.exe')]
    prefix=os.environ.get('CONDA_PREFIX')
    if prefix: candidates.append(str(Path(prefix)/'Scripts/conda.exe'))
    candidates.append(str(Path(sys.executable).parent/'Scripts/conda.exe'))
    for value in candidates:
        if value and Path(value).is_file(): return value
    raise RuntimeError('Conda was not found. Run this command in Anaconda Prompt or Miniforge Prompt.')

def download_model(destination, local=None):
    item=json.loads((ROOT/'models.json').read_text())['cellpose']
    destination.parent.mkdir(parents=True,exist_ok=True)
    def valid(p):
        return p.is_file() and p.stat().st_size==item['bytes'] and sha256(p)==item['sha256']
    if valid(destination):
        print('Cellpose-SAM checkpoint: SHA256 verified.',flush=True); return
    temp=destination.with_suffix('.partial')
    if local:
        source=Path(local).resolve()
        if not valid(source): raise ValueError('The supplied Cellpose-SAM weights do not match models.json.')
        shutil.copyfile(source,temp)
    else:
        for attempt in range(3):
            try:
                request=urllib.request.Request(item['url'],headers={'User-Agent':'TwoPhotonStudio/0.0.1'})
                with urllib.request.urlopen(request,timeout=120) as src,temp.open('wb') as dst:
                    received=0; last=0
                    while True:
                        chunk=src.read(8*1024*1024)
                        if not chunk: break
                        dst.write(chunk); received+=len(chunk)
                        if time.monotonic()-last>5:
                            print('Cellpose-SAM download: %.1f / %.1f MB'%(received/1e6,item['bytes']/1e6),flush=True)
                            last=time.monotonic()
                break
            except Exception:
                if attempt==2: raise
                print('Download interrupted; retrying from the beginning.',flush=True)
                time.sleep(2)
    if not valid(temp): raise RuntimeError('Cellpose-SAM checksum mismatch. No model was activated; rerun the installer.')
    os.replace(str(temp),str(destination))

def verify_bundled():
    for record in json.loads((ROOT/'bundled-assets.json').read_text()):
        path=ROOT/record['path']
        if not path.is_file() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise RuntimeError('Missing or damaged bundled asset: '+record['path'])

def make_env(conda,prefix,python_version,env):
    marker=prefix.parent/('.'+prefix.name+'-twophoton-owned.json')
    expected={'repository':str(ROOT),'python':python_version}
    if prefix.exists() and not marker.is_file():
        raise RuntimeError('Refusing to modify an unmanaged environment: '+str(prefix))
    if marker.exists() and json.loads(marker.read_text())!=expected:
        raise RuntimeError('Environment belongs to another checkout/version: '+str(prefix))
    prefix.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(expected),encoding='utf-8')
    python=prefix/'python.exe'
    if not python.is_file():
        run([conda,'create','--yes','--prefix',prefix,'--override-channels','--channel','conda-forge',
             'python='+python_version,'pip'],env)
    actual=subprocess.check_output([str(python),'-c','import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))'],env=env,text=True).strip()
    if actual!=python_version: raise RuntimeError('Unexpected Python version in '+str(prefix))
    return python

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='Default: cuda (NVIDIA). Use cpu on machines without a supported GPU.')
    parser.add_argument('--env-root',type=Path,help='Default: .local/envs in this checkout. May be on another drive.')
    parser.add_argument('--matlab',help='Optional absolute path to the installed MATLAB executable.')
    parser.add_argument('--cellpose-weights',type=Path,help='Use a previously downloaded, checksum-matching cpsam instead of downloading it.')
    parser.add_argument('--check',action='store_true',help='Check the completed installation without reinstalling.')
    args=parser.parse_args()
    if os.name!='nt' or sys.maxsize<2**32: raise RuntimeError('This release supports Windows x64 only.')
    STATE.mkdir(parents=True,exist_ok=True)
    if args.check:
        if not (STATE/'runtime.json').is_file():
            raise RuntimeError('Installation is incomplete. Run python install.py --device cuda (or cpu) first.')
        config=json.loads((STATE/'runtime.json').read_text())
        run([config['core_python'],ROOT/'scripts/doctor.py']); return
    sys.path.insert(0,str(ROOT/'app'))
    from runtime_config import find_matlab
    matlab=find_matlab(args.matlab)
    # Prevent concurrent installers from mutating the same environments.
    import msvcrt
    lock=(STATE/'install.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise RuntimeError('Another installer is running for this checkout.')
    verify_bundled()
    temporary=STATE/'tmp'
    temporary.mkdir(exist_ok=True)
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME'): env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_WARN_SCRIPT_LOCATION='1',
               TWOPHOTON_STATE_DIR=str(STATE),PIP_CACHE_DIR=str(STATE/'pip-cache'),
               TEMP=str(temporary),TMP=str(temporary),
               CONDA_PKGS_DIRS=str(STATE/'conda-pkgs'))
    conda=conda_executable()
    envroot=(args.env_root or STATE/'envs').expanduser().resolve()
    if any(c in str(ROOT)+str(envroot) for c in '%\r\n"'):
        raise ValueError('Choose an installation path without percent signs or quotes.')
    core=make_env(conda,envroot/'gui','3.10',env)
    modern=make_env(conda,envroot/'models','3.11',env)
    run([core,'-m','pip','install','pip==24.3.1','setuptools==75.3.2','wheel==0.45.1'],env)
    run([modern,'-m','pip','install','pip==25.1.1','setuptools==75.3.2','wheel==0.45.1'],env)
    # Resolve all non-Torch dependencies once, with the version constraints below.
    run([core,'-m','pip','install','--no-deps','torch==2.4.1+cpu','--index-url','https://download.pytorch.org/whl/cpu'],env)
    flavor='cu128' if args.device=='cuda' else 'cpu'
    run([modern,'-m','pip','install','--no-deps','torch==2.7.0+'+flavor,'torchvision==0.22.0+'+flavor,
         '--index-url','https://download.pytorch.org/whl/'+flavor],env)
    for python,name in ((core,'core'),(modern,'models')):
        run([python,'-m','pip','install','-r',ROOT/'requirements'/(name+'.txt')],env)
        run([python,'-m','pip','check'],env)
        frozen=subprocess.check_output([str(python),'-m','pip','freeze'],env=env,text=True)
        (STATE/(name+'-installed.txt')).write_text(frozen,encoding='utf-8')
    run([core,'-m','pip','install','--no-deps','--no-build-isolation','-e',ROOT],env)
    # Use modern Python's TLS stack (and the same certificate/proxy environment) for downloads.
    command=[modern,ROOT/'scripts/download_models.py']
    if args.cellpose_weights: command += ['--local',args.cellpose_weights.resolve()]
    run(command,env)
    config=dict(core_python=str(core),modern_python=str(modern),device=args.device,
                matlab=matlab,installed_at=datetime.now().isoformat(timespec='seconds'),
                repository=str(ROOT))
    candidate=STATE/'runtime.pending.json'
    candidate.write_text(json.dumps(config,indent=2),encoding='utf-8')
    run([core,ROOT/'scripts/doctor.py','--config',candidate],env)
    os.replace(str(candidate),str(STATE/'runtime.json'))
    # Launch without activation. Conda DLL directories are added inside cli.py.
    (STATE/'launch.cmd').write_text('@echo off\nchcp 65001 >nul\nset "TWOPHOTON_STATE_DIR='+str(STATE)+'"\n"'+str(core)+'" -m twophoton_studio\n',encoding='utf-8')
    print('\nINSTALLATION COMPLETE. Double-click Launch.cmd or activate '+str(envroot/'gui')+' and run twophoton-studio.')
    print('Settings, caches, installation records and logs: '+str(STATE))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('\nINSTALLATION FAILED: '+str(exc),file=sys.stderr)
        print('Fix the reported problem and rerun the same command. Existing owned environments and verified weights are reused.',file=sys.stderr)
        sys.exit(1)

