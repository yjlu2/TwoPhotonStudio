"""Isolated Suite2p 1.x CUDA registration and extraction, no GUI dependencies."""
import json,sys,time,logging
from pathlib import Path
import numpy as np
import tifffile
from tiff_frames import frames
import torch
import suite2p
from importlib.metadata import version

def run(job):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the selected Suite2p GPU environment. Select CPU backend or fix this environment.')
    device=torch.device('cuda'); torch.zeros(1,device=device)
    print(f'DEVICE: Suite2p {version("suite2p")} / CUDA / {torch.cuda.get_device_name(0)}',flush=True)
    logging.basicConfig(level=logging.INFO,stream=sys.stdout)
    cfg=job['cfg']; shape=tuple(job['shape']); started=time.perf_counter()
    if job['action']=='register':
        output=Path(job['binary'])
        with output.open('wb') as stream:
            for path in job['inputs']:
                for frame in frames(path):
                    (np.asarray(frame)//2).astype('int16').tofile(stream)
        expected=int(np.prod(shape))*2
        if output.stat().st_size!=expected: raise ValueError('Source frame count changed during conversion')
        movie=np.memmap(output,dtype='int16',mode='r+',shape=shape)
        settings=suite2p.default_settings()['registration']
        settings.update(nonrigid=True,block_size=[cfg['block_size']]*2,
                        maxregshift=cfg['maxregshift'],maxregshiftNR=cfg['maxregshiftNR'],
                        batch_size=100,nimg_init=min(300,shape[0]))
        result=suite2p.registration_wrapper(movie,settings=settings,device=device,save_path=str(output.parent))
        movie.flush(); result['gpu_suite2p_version']=version('suite2p')
        # Numeric NPZ avoids NumPy 2.x object-pickle incompatibility with the GUI's NumPy 1.x.
        np.savez(job['output'],**{k:np.asarray(v) for k,v in result.items() if v is not None})
    elif job['action']=='extract':
        from suite2p.extraction.extract import extract_traces
        movie=np.memmap(job['binary'],dtype='int16',mode='r',shape=shape)
        data=np.load(job['masks'],allow_pickle=True).item()
        F,Fneu=extract_traces(movie,data['cell'],data['neuropil'],batch_size=100,device=device)
        np.savez(job['output'],F=F,Fneu=Fneu,version=version('suite2p'))
    torch.cuda.synchronize()
    print(f'GPU stage elapsed: {time.perf_counter()-started:.2f}s; peak GPU memory: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB',flush=True)

if __name__=='__main__':
    run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
