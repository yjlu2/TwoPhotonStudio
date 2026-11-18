"""Machine-specific paths, created by install.py; no developer drive assumptions."""
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME_KEYS = ('suite2p_gpu_python','cellpose_python','profile_python','support_python',
                'profile_root','support_root','support_checkpoint','cellpose_model','reg_code')

def state_dir():
    path = Path(os.environ.get('TWOPHOTON_STATE_DIR', str(ROOT/'.local'))).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def find_matlab(explicit=None):
    value = explicit or os.environ.get('MATLAB_EXECUTABLE')
    if value:
        path=Path(value).expanduser()
        if not path.is_file(): raise ValueError('MATLAB executable does not exist: '+str(path))
        return str(path.resolve())
    found=shutil.which('matlab')
    if found: return found
    candidates=[]
    if os.name=='nt':
        import winreg
        for view in (winreg.KEY_WOW64_64KEY,winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\MathWorks\MATLAB',0,winreg.KEY_READ|view) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        with winreg.OpenKey(key,winreg.EnumKey(key,i)) as child:
                            try: candidates.append(Path(winreg.QueryValueEx(child,'MATLABROOT')[0])/'bin/matlab.exe')
                            except OSError: pass
            except OSError: pass
        for base in (os.environ.get('ProgramFiles','C:/Program Files'),os.environ.get('ProgramW6432','C:/Program Files')):
            candidates.extend(Path(base).glob('MATLAB/R*/bin/matlab.exe'))
    return str(sorted((p for p in candidates if p.is_file()),reverse=True)[0]) if any(p.is_file() for p in candidates) else ''

def installation():
    path=state_dir()/'runtime.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}

def runtime_defaults():
    data=installation()
    modern=Path(data.get('modern_python', str(state_dir()/'envs/models/python.exe')))
    return dict(suite2p_gpu_python=str(modern),cellpose_python=str(modern),
                profile_python=str(modern),support_python=str(modern),
                profile_root=str(HERE/'vendor/PROFILE'),support_root=str(HERE/'vendor/SUPPORT'),
                support_checkpoint=str(HERE/'vendor/SUPPORT/src/GUI/trained_models/bs1.pth'),
                cellpose_model=str(state_dir()/'models/cellpose/cpsam'),
                reg_code=str(HERE/'matlab'),matlab=data.get('matlab') or find_matlab(),
                suite2p_backend='gpu' if data.get('device')=='cuda' else 'cpu',
                device=data.get('device','cpu'))

def rebind_runtime(cfg):
    result=dict(cfg); current=runtime_defaults()
    # Retain user overrides which exist on this machine; rebind stale exported-session paths.
    for key in RUNTIME_KEYS:
        if not result.get(key) or not Path(result[key]).exists():
            result[key]=current[key]
    if not result.get('matlab') or not Path(result['matlab']).is_file(): result['matlab']=current['matlab']
    name=Path(result.get('profile_config','')).name
    if not result.get('profile_config') or not Path(result['profile_config']).is_file():
        candidates=(HERE/'profile_configs'/name,HERE/'vendor/PROFILE/configs'/name) if name else (HERE/'profile_configs/profile_gcamp8m_1p038um_m124_balanced.json',)
        for p in candidates:
            if p.is_file(): result['profile_config']=str(p); break
    if result.get('profile_checkpoint') and not Path(result['profile_checkpoint']).is_file():
        p=HERE/'vendor/PROFILE/models'/Path(result['profile_checkpoint']).name
        if p.is_file(): result['profile_checkpoint']=str(p)
    # A CPU installation must not inherit an unavailable CUDA backend from an old session.
    if installation().get('device')=='cpu':
        result['device']='cpu'; result['suite2p_backend']='cpu'
    return result

