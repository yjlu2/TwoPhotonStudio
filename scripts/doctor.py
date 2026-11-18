"""Check dependencies, real model forwards, and GUI imports using the configured interpreters."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8',errors='replace')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=ROOT/'.local/runtime.json')
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',
               QT_QPA_PLATFORM='offscreen',TWOPHOTON_STATE_DIR=str(ROOT/'.local'),
               NUMBA_CACHE_DIR=str(ROOT/'.local/runtime_cache'))
    env.pop('PYTHONPATH',None)
    results=[]
    commands=[
        ('GUI / legacy Suite2p',[cfg['core_python'],ROOT/'scripts/probe.py','core',str(a.config.resolve())]),
        ('Models / GPU',[cfg['modern_python'],ROOT/'scripts/probe.py','models',str(a.config.resolve())]),
    ]
    for name,command in commands:
        print('\nChecking '+name,flush=True)
        process=subprocess.Popen([str(x) for x in command],env=env,text=True,encoding='utf-8',errors='replace',
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output=[]
        for line in process.stdout:
            output.append(line); print(line,end='',flush=True)
        results.append(dict(check=name,ok=process.wait()==0,output=''.join(output)))
    record=dict(results=results,matlab=cfg.get('matlab',''),ok=all(r['ok'] for r in results))
    report=ROOT/'.local/doctor-report.json'
    report.write_text(json.dumps(record,indent=2),encoding='utf-8')
    if not record['ok']: raise SystemExit('Installation check failed. See '+str(report))
    print('ALL REQUIRED CHECKS PASSED. MATLAB: '+(cfg.get('matlab') or 'not found automatically; set its executable in Settings.'))
    print('These are synthetic software checks, not biological segmentation validation.')

if __name__=='__main__': main()

