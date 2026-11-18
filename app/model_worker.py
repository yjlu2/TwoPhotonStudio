"""Runs only inside the selected model's own installed Python environment."""
import json
import sys
from pathlib import Path
import numpy as np

def support(job):
    import torch
    cfg = job['cfg']
    sys.path.insert(0, cfg['support_root'])
    from model.SUPPORT import SUPPORT
    from src.utils.dataset_pyqt import DatasetSupport_test_stitch
    ops = np.load(job['ops'], allow_pickle=True).item()
    n, h, w = int(ops['nframes']), int(ops['Ly']), int(ops['Lx'])
    movie = np.memmap(ops['reg_file'], mode='r', dtype='int16', shape=(n,h,w))
    out = np.lib.format.open_memmap(job['output'], mode='w+', dtype='float32', shape=movie.shape)
    device = 'cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu'
    print('DEVICE: SUPPORT / '+device+(' / '+torch.cuda.get_device_name(0) if device=='cuda' else ''),flush=True)
    model = SUPPORT(in_channels=61, mid_channels=[16,32,64,128,256], depth=5,
                    blind_conv_channels=64, one_by_one_channels=[32,16],
                    last_layer_channels=[64,32,16], bs_size=int(cfg['support_bs']))
    model.load_state_dict(torch.load(cfg['support_checkpoint'], map_location='cpu'))
    model.to(device).eval()
    intervals = job.get('intervals', [[0,n]])
    with torch.no_grad():
        for start, end in intervals:
            norm = torch.tensor(np.asarray(movie[np.clip(np.arange(start-30,start+31),start,end-1)], dtype='float32'))
            mean, std = norm.mean().item(), max(norm.std().item(), 1e-6)
            for t in range(start,end):
                ids = np.clip(np.arange(t-30,t+31), start, end-1)
                frames = torch.from_numpy(np.asarray(movie[ids], dtype='float32'))
                ph, pw = min(128,h), min(128,w)
                dataset = DatasetSupport_test_stitch(frames,
                    patch_size=[61,ph,pw], patch_interval=[1,max(1,ph//2),max(1,pw//2)],
                    mean_image=mean, std_image=std)
                loader = torch.utils.data.DataLoader(dataset, batch_size=4)
                frame = np.zeros((h,w), dtype='float32')
                for noisy, _, coordinates in loader:
                    prediction = model(noisy.to(device)).cpu().numpy()
                    for i in range(len(noisy)):
                        c = {k:int(v[i]) for k,v in coordinates.items()}
                        frame[c['stack_start_h']:c['stack_end_h'], c['stack_start_w']:c['stack_end_w']] = prediction[i].squeeze()[c['patch_start_h']:c['patch_end_h'],c['patch_start_w']:c['patch_end_w']]
                out[t] = frame*std+mean
                if t%10==0 or t==end-1:
                    print(f'SUPPORT {t+1}/{n} ({device})', flush=True)
    out.flush()

def cellpose(job):
    import torch
    from cellpose import models
    cfg=job['cfg']
    checkpoint=Path(cfg['cellpose_model'])
    if not checkpoint.is_file():
        raise ValueError('Select a downloaded Cellpose checkpoint.')
    model=models.CellposeModel(gpu=cfg['device']=='cuda' and torch.cuda.is_available(),
                              pretrained_model=str(checkpoint))
    print('DEVICE: Cellpose / '+str(model.device),flush=True)
    images=np.load(job['input'])
    masks=[]
    for i,img in enumerate(images):
        result=model.eval(img, diameter=cfg['diameter'], channels=[0,0])
        masks.append(result[0])
        print(f'Cellpose projection {i+1}/{len(images)}',flush=True)
    np.save(job['output'],np.asarray(masks,dtype='int32'))

def profile(job):
    cfg=job['cfg']
    from profile_config import resolve_profile_config
    c=resolve_profile_config(cfg)
    sys.path.insert(0,cfg['profile_root'])
    from profile_inference.inference import ModelConfig, predict_vector_field, vector_field_to_mask
    import torch
    print('DEVICE: PROFILE / '+cfg['device']+(' / '+torch.cuda.get_device_name(0) if cfg['device']=='cuda' and torch.cuda.is_available() else ''),flush=True)
    print(f"PROFILE config: {cfg['profile_config']}\nCheckpoint: {c['checkpoint']}",flush=True)
    model_cfg=ModelConfig(**c)
    result=[]
    for image in np.load(job['input']):
        vf=predict_vector_field(image,model_cfg,device=cfg['device'])
        mask,_=vector_field_to_mask(vf,image.shape,model_cfg)
        result.append(mask)
    np.save(job['output'],np.asarray(result,dtype='int32'))

if __name__=='__main__':
    job=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    {'support':support,'cellpose':cellpose,'profile':profile}[sys.argv[1]](job)
