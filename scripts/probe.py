import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))

def core():
    import importlib.metadata as md
    import numpy as np
    import scipy, tifffile, imagecodecs, h5py, numba
    import suite2p
    from suite2p.extraction import dcnv, masks, extract
    from suite2p.detection.detect import detect
    from PyQt5 import QtWidgets
    import studio, backend
    assert md.version('suite2p')=='0.10.3'
    a=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=studio.Studio()
    assert w.cfg['tau']==1.0
    assert w.fit_fov_button.text()==''
    w.close()
    F=np.ones((2,100),dtype=np.float32)
    spks=dcnv.oasis(F,batch_size=100,tau=1.,fs=23.2)
    assert spks.shape==F.shape
    print('GUI constructed; native Suite2p deconvolution ran; TIFF/codec/MAT/plotting imports OK.')

def models(cfg):
    import numpy as np
    import torch
    import suite2p
    from cellpose import models as cp
    from importlib.metadata import version
    assert version('suite2p')=='1.0.0.1'
    assert version('cellpose')=='4.0.8'
    device=torch.device(cfg['device'])
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Install/update the NVIDIA driver, or rerun install.py --device cpu.')
        print('GPU:',torch.cuda.get_device_name(0),'capability:',torch.cuda.get_device_capability(0),'Torch architectures:',torch.cuda.get_arch_list(),flush=True)
    # Execute a CUDA kernel; is_available alone cannot catch unsupported architectures.
    x=torch.ones((32,32),device=device); assert (x@x).sum().item()>0
    from suite2p.extraction.extract import extract_traces
    movie=np.ones((8,32,32),dtype=np.int16)
    F,Fn=extract_traces(movie,[(np.arange(4),np.ones(4,dtype=np.float32)/4)],
                       [np.arange(20,40)],batch_size=4,device=device)
    assert F.shape==(1,8) and Fn.shape==(1,8)
    print('Suite2p modern extraction executed.',flush=True)
    # Nonrigid registration API and CUDA kernels used by the application.
    import tempfile
    from scipy.ndimage import shift
    rng=np.random.default_rng(8)
    reference=(rng.random((64,64))*1000+500).astype(np.int16)
    frames=np.stack([shift(reference,(i%3-1,i%2),order=0) for i in range(60)])
    with tempfile.TemporaryDirectory(dir=ROOT/'.local') as folder:
        path=Path(folder)/'movie.bin'; frames.tofile(path)
        movie=np.memmap(path,mode='r+',dtype='int16',shape=frames.shape)
        settings=suite2p.default_settings()['registration']
        settings.update(nonrigid=True,block_size=[32,32],batch_size=10,nimg_init=60)
        result=suite2p.registration_wrapper(movie,settings=settings,device=device,save_path=folder)
        assert len(result['yoff'])==60
        movie.flush(); del movie
    print('Suite2p nonrigid registration executed.',flush=True)
    sys.path.insert(0,str(ROOT/'app/vendor/PROFILE'))
    from profile_inference.inference import ModelConfig,predict_vector_field
    config=json.loads((ROOT/'app/profile_configs/profile_gcamp8m_1p038um_m124_balanced.json').read_text())
    config['checkpoint']=str(ROOT/'app/vendor/PROFILE'/config['checkpoint'])
    rng=np.random.default_rng(4)
    image=(rng.random((64,64))*100).astype(np.float32)
    vf=predict_vector_field(image,ModelConfig(**config),device=str(device))
    assert np.isfinite(vf).all()
    print('PROFILE checkpoint and forward pass OK.',flush=True)
    sys.path.insert(0,str(ROOT/'app/vendor/SUPPORT'))
    from model_worker import support
    with tempfile.TemporaryDirectory(dir=ROOT/'.local') as folder:
        folder=Path(folder)
        data=(rng.random((2,64,64))*1000+100).astype(np.int16)
        binary=folder/'support.bin'; data.tofile(binary)
        np.save(folder/'ops.npy',dict(nframes=2,Ly=64,Lx=64,reg_file=str(binary)))
        support(dict(cfg=dict(support_root=str(ROOT/'app/vendor/SUPPORT'),
                              support_checkpoint=str(ROOT/'app/vendor/SUPPORT/src/GUI/trained_models/bs1.pth'),
                              device=str(device),support_bs=1),
                     ops=str(folder/'ops.npy'),output=str(folder/'denoised.npy')))
        output=np.load(folder/'denoised.npy')
        assert output.shape==data.shape and np.isfinite(output).all()
    print('SUPPORT checkpoint, patch stitching and movie output OK.',flush=True)
    model=cp.CellposeModel(gpu=device.type=='cuda',pretrained_model=str(ROOT/'.local/models/cellpose/cpsam'))
    result=model.eval(image,diameter=12,batch_size=1)
    assert result[0].shape==image.shape
    print('Cellpose-SAM checkpoint and inference OK.',flush=True)

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1]=='core': core()
    else: models(cfg)

