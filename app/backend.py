"""TwoPhoton Studio: native Suite2p 0.10.3 backend, isolated model workers."""
from pathlib import Path
import copy
import colorsys
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
import time
from datetime import datetime, date, time as datetime_time, timedelta
import numpy as np
import tifffile
from tiff_frames import layout, frames, normalize_for_page_readers
from tiff_intensity import supported_dtype, prepare_inputs
from scipy.io import savemat, loadmat

HERE = Path(__file__).resolve().parent
from runtime_config import runtime_defaults, rebind_runtime, state_dir
STATE = state_dir()
os.environ['NUMBA_CACHE_DIR'] = str(STATE/'runtime_cache')
(STATE/'runtime_cache').mkdir(parents=True, exist_ok=True)
PROFILE = HERE/'vendor/PROFILE'

def roi_color(roi_id):
    """Stable high-saturation ROI color shared by the UI and every export."""
    hue=int(hashlib.sha256(str(roi_id).encode()).hexdigest()[:8],16)/2**32
    return colorsys.hsv_to_rgb(hue,.8,1.)

def defaults():
    cfg = dict(fs=23.2, tau=1.0, registration='suite2p', block_size=128, registered_tiff_mode='separate',
                suite2p_backend='gpu',suite2p_gpu_python='',
                maxregshift=0.1, maxregshiftNR=5, threshold_scaling=0.8,
                diameter=12, inner_neuropil_radius=2, min_neuropil_pixels=350,
                circular_neuropil=True,
                neucoeff=0.7, win_baseline=60.0, sig_baseline=10.0,
                support=False, support_root=str(HERE/'vendor/SUPPORT'),
                support_python='',
                support_checkpoint=str(HERE/'vendor/SUPPORT/src/GUI/trained_models/bs1.pth'),
                support_bs=1, cellpose_python='',
                cellpose_model='',
                profile_python='',
                profile_root=str(PROFILE), profile_config=str(HERE/'profile_configs/profile_gcamp8m_1p038um_m124_balanced.json') if (HERE/'vendor/PROFILE').is_dir() else '', profile_checkpoint='', continuous_files=True,
                sync_variable='data2026',sync_frame_column=1,sync_stim_column=2,sync_sample_rate=1000.0,sync_alignment='strict',
                trial_baseline_seconds=0.5,trial_post_seconds=10.0,
                matlab='',
                reg_code=str(HERE/'matlab'), device='cuda', merge_iou=0.5)

    cfg.update(runtime_defaults())
    return cfg

def relocate_config(cfg):
    return rebind_runtime(cfg)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def launch(args, log=print, cwd=None, log_path=None):
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.pop('QT_API', None)
    prefix=Path(args[0]).resolve().parent
    env['PATH']=os.pathsep.join(str(p) for p in (prefix,prefix/'Library/bin',prefix/'DLLs',prefix/'Scripts'))+os.pathsep+env.get('PATH','')
    p = subprocess.Popen([str(a) for a in args], cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding='utf-8', errors='replace',
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    from collections import deque
    tail=deque(maxlen=25)
    transcript=open(log_path,'w',encoding='utf-8') if log_path else None
    try:
        for line in p.stdout:
            tail.append(line.rstrip())
            if transcript:
                transcript.write(line); transcript.flush()
            log(line.rstrip())
        code=p.wait()
    finally:
        if transcript: transcript.close()
    if code:
        detail='\n'.join(tail)
        location=f'\nFull log: {log_path}' if log_path else ''
        raise RuntimeError(f'Model process exited with code {code}.\n{detail}{location}')

def natural(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(path))]

def intervals_from_meta(meta, continuous=True):
    if continuous:
        return [[0,meta['shape'][0]]]
    ids=np.asarray(meta['frame_map'])[:,1]
    starts=np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)]
    return [[int(a),int(b)] for a,b in zip(starts[:-1],starts[1:])]

def import_tiffs(paths, directory, log=print, preserve_order=False):
    """Index TIFF headers only. Pixel decoding starts at preparation, not import."""
    started=time.perf_counter()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    paths = [str(Path(p).resolve()) for p in paths]
    if not preserve_order: paths=sorted(paths,key=natural)
    shape = None
    frame_map = []
    records=[]
    for file_id, path in enumerate(paths, 1):
        stat=Path(path).stat()
        with tifffile.TiffFile(path) as tif:
            count, stack_shape, stack_dtype, contiguous = layout(tif)
            if contiguous:
                if not supported_dtype(stack_dtype):
                    raise ValueError('Expected 16-bit integer or 32-bit integer/float grayscale frames in ImageJ stack.')
                if shape is None: shape=stack_shape
                if shape != stack_shape: raise ValueError('All TIFF frames must have the same dimensions.')
                first_frame=len(frame_map)
                frame_map.extend([first_frame+i+1,file_id,i+1] for i in range(count))
            else:
                for page_id, page in enumerate(tif.pages, 1):
                    if len(page.shape) != 2 or not supported_dtype(page.dtype):
                        raise ValueError('Expected 2D 16-bit integer or 32-bit integer/float grayscale frames. Separate channels/planes before importing.')
                    if shape is None: shape=page.shape
                    if shape != page.shape: raise ValueError('All TIFF frames must have the same dimensions.')
                    frame_map.append([len(frame_map)+1,file_id,page_id])
            records.append(dict(path=path,frames=count,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
        log(f'Indexed {Path(path).name}; {count} frames (headers only, no copy)')
    if len(frame_map) < 2:
        raise ValueError('At least two frames are required.')
    meta = dict(files=paths, shape=[len(frame_map), *shape], frame_map=frame_map,
                records=records, storage='linked_sources', import_seconds=time.perf_counter()-started)
    write_json(directory/'source.json', meta)
    return meta

def native_ops(cfg, shape, directory):
    from suite2p import default_ops
    ops = default_ops()
    ops.update({k: cfg[k] for k in ('fs', 'tau', 'threshold_scaling', 'diameter',
               'inner_neuropil_radius', 'min_neuropil_pixels', 'neucoeff',
               'win_baseline', 'sig_baseline', 'maxregshift', 'maxregshiftNR')})
    ops.update(circular_neuropil=bool(cfg.get('circular_neuropil', True)),
               Ly=shape[1], Lx=shape[2], nframes=shape[0],
               block_size=[cfg['block_size']]*2, batch_size=100,
               nimg_init=min(300, shape[0]), nonrigid=True, preclassify=0,
               allow_overlap=False, lam_percentile=0, save_path=str(directory),
               ops_path=str(Path(directory)/'ops.npy'),
               yrange=[0, shape[1]], xrange=[0, shape[2]])
    return ops

def binary_movie(ops):
    return np.memmap(ops['reg_file'], dtype='int16', mode='r',
                     shape=(int(ops['nframes']), int(ops['Ly']), int(ops['Lx'])))

def export_registered_tiffs(movie, meta, log=print, *, mode='separate', run_dir):
    """Save each stack directly beside its source, or one combined analysis TIFF."""
    if mode=='none':
        log('Registered TIFF export disabled for this run.'); return []
    records=meta.get('records') or [{'path':p,'frames':int(np.count_nonzero(np.asarray(meta['frame_map'])[:,1]==i))}
                                    for i,p in enumerate(meta['files'],1)]
    first=0; groups=[]; run_dir=Path(run_dir)
    for i,record in enumerate(records,1):
        count=int(record['frames']); source=Path(record['path'])
        destination=source.with_name(source.stem+'_Registered.tif')
        groups.append((destination,first,first+count)); first+=count
    if first!=len(movie): raise ValueError('Source frame records do not match the registered movie.')
    if mode=='combined': groups=[(run_dir/'Registered'/'Registered.tif',0,first)]
    elif mode!='separate': raise ValueError('Unknown Registered TIFF export mode.')
    staged=[]; reserved=set()
    try:
      for proposed,start,end in groups:
        proposed.parent.mkdir(parents=True,exist_ok=True)
        destination=proposed; number=2
        timestamp=time.strftime('%Y%m%d-%H%M%S')
        while destination.exists() or str(destination).lower() in reserved:
            suffix='_'+timestamp+(f'_{number:02d}' if number>2 else '')
            destination=proposed.with_name(proposed.stem+suffix+proposed.suffix)
            number+=1
        reserved.add(str(destination).lower())
        temp=destination.with_name(destination.stem+'.'+uuid.uuid4().hex[:12]+'.partial.tif')
        if hasattr(log,'temporary'): log.temporary(temp)
        staged.append((temp,destination,end-start))
        frames=end-start
        calibration=meta.get('intensity_transform')
        estimated=frames*movie.shape[1]*movie.shape[2]*(4 if calibration else 2)
        with tifffile.TiffWriter(temp,bigtiff=estimated>=2**32-2**25) as writer:
            for written,index in enumerate(range(start,end),1):
                frame=np.asarray(movie[index],dtype='float32' if calibration else 'int16')
                if calibration: frame=frame*calibration['gain']+calibration['offset']
                writer.write(frame,photometric='minisblack',metadata=None,contiguous=True)
                if written%500==0: log(f'Registered TIFF: {written}/{frames} frames -> {destination}')
      for temp,destination,frames in staged:
            temp.rename(destination)
            log(f'Saved registered TIFF: {destination} ({frames} frames)')
      return [str(destination) for _,destination,_ in staged]
    finally:
        for temp,_,_ in staged: temp.unlink(missing_ok=True)

def imagej_auto_levels(image):
    """Approximate ImageJ/Fiji Auto brightness and contrast histogram limits."""
    values=np.asarray(image,dtype='float64'); values=values[np.isfinite(values)]
    if not len(values): return 0.,1.
    lo=float(values.min()); hi=float(values.max())
    if hi<=lo: return lo,lo+1.
    histogram,edges=np.histogram(values,bins=256,range=(lo,hi))
    limit=len(values)/10.; threshold=max(1.,len(values)/5000.)
    first=0
    while first<255:
        count=0 if histogram[first]>limit else histogram[first]
        if count>threshold: break
        first+=1
    last=255
    while last>0:
        count=0 if histogram[last]>limit else histogram[last]
        if count>threshold: break
        last-=1
    if last<=first: return lo,hi
    return float(edges[first]),float(edges[last+1])

def temporal_average_frame(movie, index, window=5):
    """Centered display-only temporal mean; registration data remain unchanged."""
    window=max(1,int(window)); index=int(index)
    if window==1: return np.asarray(movie[index],dtype='float32')
    half=window//2; start=max(0,index-half); end=min(len(movie),index+half+1)
    return np.asarray(movie[start:end],dtype='float32').mean(axis=0,dtype='float32')

def prepare(root, cfg, log=print):
    root = Path(root).resolve()
    meta = json.loads((root/'source.json').read_text(encoding='utf-8'))
    shape = meta['shape']
    # Old sessions retain their copied movie; new sessions link original files.
    inputs=[str(root/'input.tif')] if meta.get('storage')!='linked_sources' and (root/'input.tif').exists() else meta['files']
    for rec in meta.get('records',[]):
        st=Path(rec['path']).stat()
        if st.st_size!=rec['bytes'] or st.st_mtime_ns!=rec['mtime_ns']:
            raise ValueError('Source TIFF changed since import: '+rec['path'])
    folder = root/('prep_'+uuid.uuid4().hex[:8])
    folder.mkdir()
    inputs,intensity_transform=prepare_inputs(inputs,folder,log)
    # CPU Suite2p and MATLAB expect one TIFF page per frame.
    # GPU and already-registered paths stream the original contiguous stack.
    if cfg['registration']=='matlab' or (cfg['registration']=='suite2p' and cfg.get('suite2p_backend','cpu')=='cpu'):
        inputs=[normalize_for_page_readers(path,folder/f'input_multipage_{i}.tif',log)
                for i,path in enumerate(inputs,1)]
    ops = native_ops(cfg, shape, folder)
    if cfg['registration'] == 'suite2p':
        if cfg.get('suite2p_backend','cpu')=='gpu':
            job=dict(action='register',cfg=cfg,shape=shape,inputs=inputs,
                     binary=str(folder/'registered.bin'),output=str(folder/'gpu_registration.npz'))
            write_json(folder/'gpu_job.json',job)
            launch([cfg['suite2p_gpu_python'],HERE/'suite2p_gpu_worker.py',folder/'gpu_job.json'],log,log_path=folder/'gpu_registration.log')
            with np.load(job['output'],allow_pickle=False) as registered:
                ops.update({k:registered[k].item() if registered[k].ndim==0 else registered[k]
                            for k in registered.files})
            ops['reg_file']=job['binary']
        else:
            from suite2p import run_s2p
            ops.update(roidetect=False, do_registration=1, save_mat=False,
                       save_path0=str(folder), data_path=[str(root)], tiff_list=inputs)
            log('DEVICE: Suite2p registration / CPU (installed 0.10.3)')
            result = run_s2p(ops=ops, db=dict(data_path=[str(root)],
                             tiff_list=inputs, save_path0=str(folder)))
            ops = result[0] if isinstance(result, (list, np.ndarray)) else result
        ops['yrange'],ops['xrange']=np.asarray(automatic_valid_bounds(ops)).reshape(2,2).tolist()
        ops['valid_area_mode']='auto'
    elif cfg['registration'] == 'matlab':
        config = dict(inputs=inputs, output=str(folder/'registered.bin'),
                      metadata=str(folder/'registration.mat'), code=cfg['reg_code'])
        write_json(folder/'matlab.json', config)
        quote = lambda s: str(s).replace("'", "''").replace('\\', '/')
        expr = f"addpath('{quote(HERE)}'); studio_register('{quote(folder/'matlab.json')}')"
        launch([cfg['matlab'], '-batch', expr], log)
        r = loadmat(folder/'registration.mat', squeeze_me=True)
        ops.update(reg_file=str(folder/'registered.bin'), yoff=r['yoff'], xoff=r['xoff'],
                   corrXY=r['regError'], refImg=r['refImg'], registration_error=r['regError'])
        my = int(np.ceil(np.max(np.abs(r['yoff']))))+1
        mx = int(np.ceil(np.max(np.abs(r['xoff']))))+1
        ops['yrange'], ops['xrange'] = [my, shape[1]-my], [mx, shape[2]-mx]
    elif cfg['registration'] == 'already':
        ops['reg_file'] = str(folder/'registered.bin')
        with open(ops['reg_file'], 'wb') as f:
            for path in inputs:
                for frame in frames(path):
                    (np.asarray(frame)//2).astype('int16').tofile(f)
        ops.update(yoff=np.zeros(shape[0]), xoff=np.zeros(shape[0]))
    else:
        raise ValueError('Unknown registration method')
    if ops['yrange'][1] <= ops['yrange'][0] or ops['xrange'][1] <= ops['xrange'][0]:
        raise ValueError('Excessive registration displacement: no common valid area remains.')
    movie = binary_movie(ops)
    if movie.shape[0] != shape[0]:
        raise ValueError('Registration changed the frame count.')
    mean, std = summaries(movie)
    ops['meanImg'] = mean
    ops['registration_method'] = cfg['registration']
    ops['session_intervals'] = np.asarray(intervals_from_meta(meta,cfg.get('continuous_files',True)))
    ops['intensity_convention'] = 'Suite2p units: uint16 input divided by 2 before int16 binary'
    if intensity_transform:
        ops['source_intensity_transform']=intensity_transform
        ops['intensity_convention']=intensity_transform['convention']

    np.save(folder/'registered_ops.npy', ops)
    np.save(folder/'std.npy', std)
    detection_file = ops['reg_file']
    if cfg['support']:
        log('SUPPORT: detection only; all frames retained; boundary replication within 30 frames of each session edge.')
        job = dict(cfg=cfg, ops=str(folder/'registered_ops.npy'), output=str(folder/'support.npy'),
                   intervals=ops['session_intervals'].tolist())
        write_json(folder/'support_job.json', job)
        launch([cfg['support_python'], HERE/'model_worker.py', 'support', folder/'support_job.json'], log)
        den = np.load(folder/'support.npy', mmap_mode='r')
        if list(den.shape) != shape:
            raise ValueError('SUPPORT output/frame mapping mismatch')
        detection_file = str(folder/'detection.bin')
        with open(detection_file, 'wb') as f:
            for first in range(0, shape[0], 100):
                arr = den[first:first+100]
                if not np.isfinite(arr).all():
                    raise ValueError('SUPPORT produced nonfinite pixels')
                np.clip(np.rint(arr), -32768, 32767).astype('int16').tofile(f)
        denmean, _ = summaries(den)
        np.save(folder/'denoised_mean.npy', denmean)
    ops['registered_tiffs']=export_registered_tiffs(movie,{**meta,'intensity_transform':intensity_transform},
                                    log,mode=cfg.get('registered_tiff_mode','separate'),run_dir=folder)
    np.save(folder/'registered_ops.npy',ops)
    write_json(folder/'preparation.json', dict(cfg=cfg, detection_file=detection_file,
               boundary_context_frames=30 if cfg['support'] else 0))
    return str(folder/'registered_ops.npy')

def summaries(movie):
    a = np.zeros(movie.shape[1:], 'float64')
    b = np.zeros_like(a)
    for first in range(0, len(movie), 100):
        v = np.asarray(movie[first:first+100], dtype='float64')
        a += v.sum(0)
        b += (v*v).sum(0)
    mean = a/len(movie)
    return mean.astype('float32'), np.sqrt(np.maximum(0, b/len(movie)-mean**2)).astype('float32')

def automatic_valid_bounds(ops):
    """Conservative per-side bounds from same-frame composed Suite2p shifts."""
    h,w=int(ops['Ly']),int(ops['Lx'])
    if ops.get('registration_method')=='matlab':
        # DFT translation metadata uses a different sign convention.
        return [int(np.ceil(np.max(np.abs(ops.get('yoff',[0])))))+1,
                h-int(np.ceil(np.max(np.abs(ops.get('yoff',[0]))))) -1,
                int(np.ceil(np.max(np.abs(ops.get('xoff',[0])))))+1,
                w-int(np.ceil(np.max(np.abs(ops.get('xoff',[0]))))) -1]
    result=[]
    for key,extent in [('yoff',h),('xoff',w)]:
        rigid=np.asarray(ops.get(key,[0]),dtype='float64').reshape(-1)
        local=np.asarray(ops.get(key+'1',np.zeros((len(rigid),1))),dtype='float64')
        if local.size==0: local=np.zeros((len(rigid),1))
        local=local.reshape(len(rigid),-1)
        if not np.isfinite(rigid).all() or not np.isfinite(local).all():
            raise ValueError('Registration displacement contains nonfinite values.')
        total=rigid[:,None]+local
        # Both the rigid roll and subsequent nonrigid sampling must stay in bounds.
        low=min(0.,float(total.min()),float(local.min()))
        high=max(0.,float(total.max()),float(local.max()))
        top=int(np.ceil(-low))+(1 if low<0 else 0)
        bottom=extent-int(np.ceil(high))-(1 if high>0 else 0)
        if bottom<=top: raise ValueError('No common valid area remains after registration.')
        result.extend([top,bottom])
    return result

def set_valid_bounds(ops_file,bounds,mode='manual'):
    path=Path(ops_file); ops=np.load(path,allow_pickle=True).item()
    y0,y1,x0,x1=map(int,bounds)
    if not (0<=y0<y1<=int(ops['Ly']) and 0<=x0<x1<=int(ops['Lx'])):
        raise ValueError('Valid area must be a nonempty rectangle inside the FOV.')
    if 'original_valid_bounds' not in ops:
        ops['original_valid_bounds']=list(ops['yrange'])+list(ops['xrange'])
    ops.update(yrange=[y0,y1],xrange=[x0,x1],valid_area_mode=mode,
               automatic_valid_bounds=automatic_valid_bounds(ops))
    temporary=path.with_name(path.stem+'.area.partial.npy')
    np.save(temporary,ops); temporary.replace(path)
    return ops

def valid_mask(ops):
    mask = np.zeros((ops['Ly'], ops['Lx']), bool)
    mask[slice(*ops['yrange']), slice(*ops['xrange'])] = True
    return mask

def detect(ops_file, cfg, method, log=print):
    if method=='profile':
        from profile_config import resolve_profile_config
        resolve_profile_config(cfg)
    ops_file = Path(ops_file)
    ops = np.load(ops_file, allow_pickle=True).item()
    prep = json.loads((ops_file.parent/'preparation.json').read_text(encoding='utf-8'))
    ops.update(threshold_scaling=cfg['threshold_scaling'], diameter=cfg['diameter'],
               fs=cfg['fs'], tau=cfg['tau'], reg_file=prep['detection_file'])
    run = ops_file.parent/('detect_'+method+'_'+uuid.uuid4().hex[:8])
    run.mkdir()
    if method == 'suite2p':
        log('DEVICE: Suite2p functional detection / CPU')
        from suite2p.detection import detect as native_detect
        ops['save_path'] = str(run)
        _, stat = native_detect(ops)
    else:
        mov = binary_movie(ops)
        # Multiple time windows supplement the global morphological projection.
        projections = [ops['meanImg']]
        if cfg['support'] and (ops_file.parent/'denoised_mean.npy').exists():
            projections = [np.load(ops_file.parent/'denoised_mean.npy')]
        if method == 'cellpose':
            for ids in np.array_split(np.arange(len(mov)), min(4, len(mov))):
                projections.append(summaries(mov[ids[0]:ids[-1]+1])[0])
        np.save(run/'projections.npy', np.asarray(projections))
        write_json(run/'job.json', dict(cfg=cfg, input=str(run/'projections.npy'), output=str(run/'masks.npy')))
        launch([cfg[method+'_python'], HERE/'model_worker.py', method, run/'job.json'], log,
               log_path=run/'inference.log')
        masks = np.load(run/'masks.npy')
        stat = []
        for labels in masks:
            for label in np.unique(labels):
                if not label:
                    continue
                y, x = np.where(labels == label)
                if len(y) >= 3:
                    stat.append(dict(ypix=y, xpix=x, lam=np.ones(len(y), 'float32')))
    clean = []
    valid = valid_mask(ops)
    for s in stat:
        s = dict(s)
        if not valid[s['ypix'], s['xpix']].all():
            continue
        s.update(source=method, roi_id=uuid.uuid4().hex[:12], accepted=True,
                 reviewed=False, probability=np.nan)
        clean.append(s)
    result, ignored = merge_candidates([], clean, ops['Lx'], cfg['merge_iou'])
    log(f'{method}: {len(result)} valid candidates; {ignored} duplicate candidates')
    return result

def merge_candidates(existing, incoming, width, threshold=0.5):
    out = copy.deepcopy(list(existing))
    pixels = [set((s['ypix']*width+s['xpix']).tolist()) for s in out]
    ignored = 0
    for s in incoming:
        p = set((s['ypix']*width+s['xpix']).tolist())
        # IoU alone misses the common projection case where a second mask for the
        # same cell is substantially smaller and contained by the first.
        if any((len(p&q)/max(1, len(p|q)) >= threshold or
                len(p&q)/max(1, min(len(p),len(q))) >= max(.8,threshold)) for q in pixels):
            ignored += 1
            continue
        out.append(copy.deepcopy(s))
        pixels.append(p)
    return out, ignored

def sanitize_rois_for_extraction(stat, ops, valid, log=print):
    """Normalize masks and apply Suite2p-style overlap triage before roi_stats."""
    clean=[]; removed=[]
    Ly,Lx=int(ops['Ly']),int(ops['Lx'])
    for original in copy.deepcopy(list(stat)):
        s=dict(original)
        y=np.asarray(s.get('ypix',[])).reshape(-1); x=np.asarray(s.get('xpix',[])).reshape(-1)
        lam=np.asarray(s.get('lam',np.ones(len(y))),dtype='float32').reshape(-1)
        if len(y)!=len(x) or len(y)!=len(lam):
            removed.append(str(s.get('roi_id','unknown'))); continue
        finite=np.isfinite(y)&np.isfinite(x)&np.isfinite(lam)&(lam>0)
        y=y[finite].astype('int64'); x=x[finite].astype('int64'); lam=lam[finite]
        inside=(y>=0)&(y<Ly)&(x>=0)&(x<Lx); y,x,lam=y[inside],x[inside],lam[inside]
        if len(y):
            _,unique=np.unique(y*Lx+x,return_index=True); unique=np.sort(unique)
            y,x,lam=y[unique],x[unique],lam[unique]
        if len(y)<3 or not valid[y,x].all():
            removed.append(str(s.get('roi_id','unknown'))); continue
        s.update(ypix=y,xpix=x,lam=lam,med=[np.median(y),np.median(x)]); clean.append(s)
    if clean:
        overlap=np.zeros((Ly,Lx),'uint16')
        for s in clean: overlap[s['ypix'],s['xpix']]+=1
        keep=np.ones(len(clean),bool); max_overlap=float(ops.get('max_overlap',.75))
        # Suite2p checks in reverse order, subtracting rejected masks so that one
        # representative survives a stack of redundant masks.
        for i in range(len(clean)-1,-1,-1):
            s=clean[i]
            if np.mean(overlap[s['ypix'],s['xpix']]>1)>max_overlap:
                keep[i]=False; overlap[s['ypix'],s['xpix']]-=1
        removed.extend(str(s.get('roi_id','unknown')) for i,s in enumerate(clean) if not keep[i])
        clean=[s for i,s in enumerate(clean) if keep[i]]
    if removed:
        preview=', '.join(removed[:8])+(' ...' if len(removed)>8 else '')
        log(f'ROI cleanup: removed {len(removed)} invalid or redundantly overlapped candidate(s): {preview}')
    if not clean:
        raise ValueError('No extractable ROI remains after overlap cleanup.')
    return clean,removed

def extract(ops_file, stat, cfg, directory, log=print):
    """All candidates extracted; accepted is a label, never a response test."""
    cfg=copy.deepcopy(cfg)
    cfg.setdefault('circular_neuropil', True)
    from suite2p.detection.stats import roi_stats
    from suite2p.extraction import masks as native_masks
    from suite2p.extraction.extract import extract_traces_from_masks
    from suite2p.extraction import dcnv
    from suite2p.classification import classify
    import suite2p
    ops = np.load(ops_file, allow_pickle=True).item()
    ops.update({k: cfg[k] for k in ('fs','tau','neucoeff','inner_neuropil_radius',
                                  'min_neuropil_pixels','win_baseline','sig_baseline')})
    ops.update(allow_overlap=False, lam_percentile=0.0, baseline='maximin',
               circular_neuropil=bool(cfg['circular_neuropil']))
    if not stat:
        raise ValueError('Detect or add ROIs first.')
    valid = valid_mask(ops)
    stat,removed_roi_ids = sanitize_rois_for_extraction(stat,ops,valid,log)
    stat = roi_stats(np.array(stat, dtype=object), 10, 10, ops['Ly'], ops['Lx'],
                     max_overlap=None, do_crop=True)
    cell = [native_masks.create_cell_mask(s, ops['Ly'], ops['Lx'], False) for s in stat]
    if any(len(p)==0 or not np.isfinite(w).all() for p,w in cell):
        raise ValueError('ROI cleanup could not produce finite non-overlapping masks. See the processing log.')
    occupancy = (~valid).astype('float32')
    for s in stat:
        occupancy[s['ypix'], s['xpix']] = 1
    log('Neuropil mask shape: '+('circular expansion' if cfg['circular_neuropil'] else 'rectangular expansion'))
    neuropil = native_masks.create_neuropil_masks(
        [s['ypix'] for s in stat], [s['xpix'] for s in stat], occupancy,
        int(cfg['inner_neuropil_radius']), int(cfg['min_neuropil_pixels']),
        circular=bool(cfg['circular_neuropil']))
    counts = np.array([len(p) for p in neuropil])
    if np.any(counts < cfg['min_neuropil_pixels']):
        raise ValueError('Insufficient valid neuropil pixels. Check dense/duplicate ROIs or reduce the minimum.')
    if cfg.get('suite2p_backend','cpu')=='gpu':
        scratch=Path(ops_file).parent/('gpu_extract_'+uuid.uuid4().hex[:8]); scratch.mkdir()
        np.save(scratch/'masks.npy',dict(cell=cell,neuropil=neuropil))
        job=dict(action='extract',cfg=cfg,shape=[ops['nframes'],ops['Ly'],ops['Lx']],
                 binary=ops['reg_file'],masks=str(scratch/'masks.npy'),output=str(scratch/'traces.npz'))
        write_json(scratch/'job.json',job)
        launch([cfg['suite2p_gpu_python'],HERE/'suite2p_gpu_worker.py',scratch/'job.json'],log,log_path=scratch/'gpu_extraction.log')
        with np.load(job['output']) as arrays:
            F,Fneu=arrays['F'],arrays['Fneu']; ops['extraction_backend']='CUDA / Suite2p '+str(arrays['version'])
    else:
        log('DEVICE: Suite2p extraction / CPU')
        F, Fneu, _, _, ops = extract_traces_from_masks(ops, cell, neuropil)
        ops['extraction_backend']='CPU / Suite2p 0.10.3'
    log('DEVICE: baseline / OASIS / CPU (Suite2p 0.10.3)')
    calibration=ops.get('source_intensity_transform')
    if calibration:
        F=(F.astype('float64')*calibration['gain']+calibration['offset']).astype('float32')
        Fneu=(Fneu.astype('float64')*calibration['gain']+calibration['offset']).astype('float32')
        log('Restored raw F and neuropil Fneu to original TIFF intensity units before baseline, dF/F and OASIS.')
    Fc = F-cfg['neucoeff']*Fneu
    df=np.empty_like(Fc); spks=np.empty_like(Fc)
    for start,end in ops['session_intervals']:
        segment=dcnv.preprocess(F=Fc[:,start:end].copy(), baseline='maximin',
                        win_baseline=cfg['win_baseline'], sig_baseline=cfg['sig_baseline'], fs=cfg['fs'])
        df[:,start:end]=segment
        spks[:,start:end]=dcnv.oasis(F=segment.copy(), batch_size=100, tau=cfg['tau'], fs=cfg['fs'])
    baseline = Fc-df
    dff = np.divide(df, baseline, out=np.full_like(df, np.nan), where=baseline>1e-6)
    from scipy.stats import skew
    for i,s in enumerate(stat):
        s.update(skew=float(skew(Fc[i])), std=float(np.std(Fc[i])), neuropil_mask=neuropil[i])
    classifier = Path(suite2p.__file__).parent/'classifiers/classifier.npy'
    probs = classify(stat, str(classifier))[:,1]
    for i,s in enumerate(stat):
        s['probability'] = float(probs[i]) if s['source']=='suite2p' else np.nan
        if s['source']=='suite2p' and not s.get('reviewed', False):
            s['accepted'] = bool(probs[i] >= 0.5)
    iscell = np.array([[float(s['accepted']), probs[i] if s['source']=='suite2p' else np.nan]
                       for i,s in enumerate(stat)])
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    arrays = dict(F=F, Fneu=Fneu, Fc=Fc, F0=baseline, dff=dff, spks=spks, iscell=iscell)
    for name, value in arrays.items():
        np.save(directory/(name+'.npy'), value)
    np.save(directory/'stat.npy', stat)
    np.save(directory/'ops.npy', ops)
    np.save(directory/'neuropil_counts.npy', counts)
    write_json(directory/'removed_roi_ids.json',removed_roi_ids)
    write_json(directory/'settings.json', cfg)
    log(f'Computed {len(stat)} ROIs x {F.shape[1]} frames. Traces ready.')
    return str(directory)

def _matlab_export_value(value):
    """Encode Python timestamp metadata while preserving numeric trace arrays."""
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, np.datetime64):
        return '' if np.isnat(value) else np.datetime_as_string(value)
    if isinstance(value, dict):
        return {key: _matlab_export_value(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_matlab_export_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'M':
            return np.datetime_as_string(value)
        if value.dtype.kind == 'O':
            converted=np.empty(value.shape,dtype=object)
            for index in np.ndindex(value.shape):
                converted[index]=_matlab_export_value(value[index])
            return converted
    if value is None:
        return np.empty(0)
    return value

def export_mat(session, destination, replace=False, log=print):
    result = Path(session['result'])
    stat = np.load(result/'stat.npy', allow_pickle=True)
    current = session['stat']
    if [s['roi_id'] for s in stat] != [s['roi_id'] for s in current]:
        raise ValueError('ROI identities changed. Recompute traces.')
    arrays = {n:np.load(result/(n+'.npy')) for n in ('F','Fneu','Fc','F0','dff','spks','iscell')}
    for i,s in enumerate(stat):
        s['accepted'] = current[i]['accepted']
        s['reviewed'] = current[i]['reviewed']
        arrays['iscell'][i,0] = s['accepted']
    root = Path(session['root'])
    source = json.loads((root/'source.json').read_text(encoding='utf-8'))
    cfg = json.loads((result/'settings.json').read_text(encoding='utf-8'))
    ops = np.load(result/'ops.npy', allow_pickle=True).item()
    ops = {k:v for k,v in ops.items() if v is not None and not k.startswith('_')}
    nt = arrays['F'].shape[1]
    frame_times = np.arange(nt)/cfg['fs'] if session.get('frame_times') is None else session['frame_times']
    payload = dict(arrays, stat=stat, ops=ops, frameTimes=frame_times,
        frameMap=np.column_stack([source['frame_map'], frame_times]), filelist=source['files'],
        roi_ids=np.array([s['roi_id'] for s in stat], dtype=object),
        roi_source=np.array([s['source'] for s in stat], dtype=object),
        provenance=dict(app='TwoPhoton Studio 1.3.3', extraction=ops.get('extraction_backend','native suite2p 0.10.3'),
            trace_source='registered movie WITHOUT SUPPORT',
            coordinates='stat.xpix/ypix zero based; frameMap frame/file/page one based',
            frameMap_columns='frame,fileIndex,pageIndex,timeSeconds',
            time_source=('nominal fs anchored to first frame TTL' if (session.get('sync') or {}).get('alignment_mode')=='first_edge' else
                         'nominal fs' if session.get('frame_times') is None else 'imported acquisition timestamps'),
            spks='OASIS inferred fluorescence innovations, NOT measured spike counts',
            settings_json=json.dumps(cfg, ensure_ascii=False)))
    sync=session.get('sync')
    if sync:
        payload.update(stimFrames=np.asarray(sync['stim_frames'],dtype='int64').reshape(-1,1),
                       stimTimes=np.asarray(sync['stim_times'],dtype='float64').reshape(-1,1),
                       stimEndFrames=np.asarray(sync.get('stim_end_frames',[]),dtype='int64').reshape(-1,1),
                       stimEndTimes=np.asarray(sync.get('stim_end_times',[]),dtype='float64').reshape(-1,1),
                       syncFrameTTLSamples=np.asarray(sync['frame_ttl_samples'],dtype='int64').reshape(-1,1),
                       syncStimTTLSamples=np.asarray(sync['stim_ttl_samples'],dtype='int64').reshape(-1,1),
                       syncStimEndTTLSamples=np.asarray(sync.get('stim_end_ttl_samples',[]),dtype='int64').reshape(-1,1),
                       syncFile=sync['sync_file'], syncMode=sync['sync_mode'],
                       niSampleRate=float(sync.get('ni_sample_rate',1000.)))
        payload['provenance']['sync_mode']=sync['sync_mode']
        payload['provenance']['sync_file']=json.dumps(sync['sync_file'],ensure_ascii=False)
        payload['provenance']['stimulus_frames']=('1-based global frame indices mapped from stimulus TTL high intervals to nominal frame times anchored at first frame TTL' if sync.get('alignment_mode')=='first_edge' else '1-based global frame indices mapped from configured stimulus TTL high intervals to configured frame TTLs')
    destination = Path(destination)
    if destination.exists() and not replace:
        raise ValueError('Destination exists. Confirm Replace in the export window or choose another filename.')
    temp = destination.with_name(destination.stem+'.'+uuid.uuid4().hex[:12]+'.partial.mat')
    if hasattr(log,'temporary'): log.temporary(temp)
    try:
        savemat(temp, _matlab_export_value(payload), do_compression=True, long_field_names=True)
        # Replace only after the new MAT has been written successfully.
        if replace: temp.replace(destination)
        else: temp.rename(destination)
    finally:
        if temp.exists(): temp.unlink()
    return str(destination)

def export_selected_roi(session, roi_index, destination, trace_names, normalized=False,
                        show_stim=True, display_percentile=99.5):
    """Export one selected ROI as numeric MAT, trace PNG/PDF/FIG, and mask-FOV PNG/PDF."""
    result=Path(session['result']); ops=np.load(result/'ops.npy',allow_pickle=True).item()
    stat=list(np.load(result/'stat.npy',allow_pickle=True))
    roi_index=int(roi_index)
    if not 0<=roi_index<len(stat): raise ValueError('Select a valid ROI first.')
    s=stat[roi_index]; nframes=int(ops['nframes'])
    arrays={name:np.load(result/(name+'.npy'),mmap_mode='r') for name in ('F','Fneu','Fc','dff','spks')}
    times=session.get('frame_times')
    if times is None: times=np.arange(nframes)/float(ops.get('fs',23.2))
    times=np.asarray(times,dtype='float64').reshape(-1)
    if len(times)!=nframes: raise ValueError('Frame-time count does not match extracted traces.')
    destination=Path(destination).resolve(); base=destination.with_suffix('')
    paths=dict(mat=base.with_suffix('.mat'),trace_png=base.with_name(base.name+'_trace.png'),
               trace_pdf=base.with_name(base.name+'_trace.pdf'),trace_fig=base.with_name(base.name+'_trace.fig'),
               fov_png=base.with_name(base.name+'_mask_fov.png'),
               fov_pdf=base.with_name(base.name+'_mask_fov.pdf'))
    existing=[str(p) for p in paths.values() if p.exists()]
    if existing: raise ValueError('Export destination already exists: '+', '.join(existing))
    mask=np.zeros((int(ops['Ly']),int(ops['Lx'])),dtype='uint8'); mask[s['ypix'],s['xpix']]=1
    mean=np.asarray(ops['meanImg'],dtype='float32')
    sync=session.get('sync') or {}; stim_frames=np.asarray(sync.get('stim_frames',[]),dtype='int64')
    stim_times=np.asarray(sync.get('stim_times',[]),dtype='float64')
    stim_end_frames=np.asarray(sync.get('stim_end_frames',[]),dtype='int64')
    stim_end_times=np.asarray(sync.get('stim_end_times',[]),dtype='float64')
    roi_stat=dict(ypix=np.asarray(s['ypix'],dtype='int64'),xpix=np.asarray(s['xpix'],dtype='int64'),
                  lam=np.asarray(s['lam'],dtype='float32'),accepted=bool(s.get('accepted',True)),
                  reviewed=bool(s.get('reviewed',False)),probability=float(s.get('probability',np.nan)))
    payload={name:np.asarray(value[roi_index],dtype='float32').reshape(1,-1) for name,value in arrays.items()}
    payload.update(time=times.reshape(1,-1),stimFrames=stim_frames.reshape(-1,1),
                   stimTimes=stim_times.reshape(-1,1),stimEndFrames=stim_end_frames.reshape(-1,1),
                   stimEndTimes=stim_end_times.reshape(-1,1),roiMask=mask,registeredMean=mean,
                   roiStat=roi_stat,roiIndex=np.int64(roi_index+1),
                   roiId=str(s.get('roi_id','')),roiSource=str(s.get('source','')),
                   displayTraceNames=np.asarray(trace_names,dtype=object),
                   displayNormalized=bool(normalized),displayShowStim=bool(show_stim),
                   sourceFallResult=str(result))
    savemat(paths['mat'],payload,do_compression=True,long_field_names=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    colors={'F':'#65df9b','Fneu':'#ff9c6b','spks':'#d49aff','Fc':'#6dcfff','dff':'#d8d86a'}
    fig=Figure(figsize=(12,3.2),facecolor='none'); FigureCanvasAgg(fig); ax=fig.add_subplot(111); ax.set_facecolor('none')
    for name in trace_names:
        if name not in arrays: continue
        values=np.asarray(arrays[name][roi_index],dtype='float64')
        if normalized:
            finite=values[np.isfinite(values)]
            if len(finite): values=(values-finite.min())/max(float(np.ptp(finite)),1e-9)
        ax.plot(times,values,color=colors[name],linewidth=.75)
    if show_stim:
        for index,onset in enumerate(stim_times):
            ax.axvline(float(onset),color='#ff5a5f',linestyle='--',linewidth=.75)
            if index<len(stim_end_times): ax.axvspan(float(onset),float(stim_end_times[index]),color='#ff4850',alpha=.30)
    ax.set_xlabel('Time (s)',color='black'); ax.set_ylabel('Normalized display' if normalized else 'Native units / dF/F',color='black')
    ax.set_title(f'ROI {roi_index+1} | {s.get("source","")} | registered movie',color='black')
    ax.tick_params(colors='black'); [spine.set_color('black') for spine in ax.spines.values()]
    fig.tight_layout(); fig.savefig(paths['trace_png'],dpi=220,transparent=True); fig.savefig(paths['trace_pdf'],transparent=True)
    matlab=Path(session.get('cfg',{}).get('matlab') or defaults()['matlab'])
    if not matlab.is_file(): raise RuntimeError('MATLAB is required to create the editable trace FIG file.')
    quote=lambda value:str(value).replace("'","''")
    command=f"addpath('{quote(HERE)}'); studio_export_trace_fig('{quote(paths['mat'])}','{quote(paths['trace_fig'])}');"
    completed=subprocess.run([str(matlab),'-batch',command],cwd=str(HERE),capture_output=True,text=True)
    if completed.returncode or not paths['trace_fig'].is_file():
        raise RuntimeError('MATLAB could not create the trace FIG file.\n'+(completed.stderr or completed.stdout)[-2000:])
    fig=Figure(figsize=(7,7),facecolor='black'); FigureCanvasAgg(fig); ax=fig.add_subplot(111); ax.set_facecolor('black')
    lo,hi=np.percentile(mean,[1,float(display_percentile)]); ax.imshow(mean,cmap='gray',vmin=lo,vmax=max(hi,lo+1))
    from scipy.ndimage import binary_erosion
    edge=mask.astype(bool)&~binary_erosion(mask.astype(bool)); rgba=np.zeros((*mask.shape,4),dtype='float32')
    rgba[mask.astype(bool)]=[1.,.86,.25,.35]; rgba[edge]=[1.,.86,.25,1.]
    ax.imshow(rgba); ax.set_axis_off(); ax.set_title(f'ROI {roi_index+1} mask | {s.get("source","")}',color='white')
    fig.tight_layout(pad=.3); fig.savefig(paths['fov_png'],dpi=220,facecolor=fig.get_facecolor()); fig.savefig(paths['fov_pdf'],facecolor=fig.get_facecolor())
    return {k:str(v) for k,v in paths.items()}

def export_selected_rois(session, roi_indices, destination, trace_names, normalized=False,
                         show_stim=True, display_percentile=99.5, log=print):
    """Export one or more ROIs with shared UI colors and metric-specific line styles."""
    result=Path(session['result']); ops=np.load(result/'ops.npy',allow_pickle=True).item()
    stat=list(np.load(result/'stat.npy',allow_pickle=True)); indices=np.asarray(sorted(set(map(int,roi_indices))),dtype=int)
    if not len(indices) or np.any(indices<0) or np.any(indices>=len(stat)): raise ValueError('Select valid ROI rows first.')
    nframes=int(ops['nframes']); arrays={name:np.load(result/(name+'.npy'),mmap_mode='r') for name in ('F','Fneu','Fc','dff','spks')}
    times=session.get('frame_times')
    if times is None: times=np.arange(nframes)/float(ops.get('fs',23.2))
    times=np.asarray(times,dtype='float64').reshape(-1)
    if len(times)!=nframes: raise ValueError('Frame-time count does not match extracted traces.')
    destination=Path(destination).resolve(); base=destination.with_suffix('')
    paths=dict(mat=base.with_suffix('.mat'),trace_fig=base.with_name(base.name+'_traces.fig'),
               trace_png=base.with_name(base.name+'_traces.png'),trace_pdf=base.with_name(base.name+'_traces.pdf'),
               fov_png=base.with_name(base.name+'_roi_masks_registered_mean.png'),
               fov_pdf=base.with_name(base.name+'_roi_masks_registered_mean.pdf'))
    existing=[str(p) for p in paths.values() if p.exists()]
    if existing: raise ValueError('Export destination already exists: '+', '.join(existing))
    height,width=int(ops['Ly']),int(ops['Lx']); masks=np.zeros((len(indices),height,width),dtype='uint8')
    ids=[]; sources=[]
    for row,index in enumerate(indices):
        s=stat[index]; masks[row,s['ypix'],s['xpix']]=1
        ids.append(str(s.get('roi_id',''))); sources.append(str(s.get('source','')))
    colors=np.asarray([roi_color(value) for value in ids],dtype='float64')
    mean=np.asarray(ops['meanImg'],dtype='float32'); sync=session.get('sync') or {}
    payload={name:np.asarray(value[indices],dtype='float32') for name,value in arrays.items()}
    payload.update(time=times.reshape(1,-1),roiMasks=masks,registeredMean=mean,
                   roiIndices=(indices+1).astype('int64').reshape(-1,1),roiIds=np.asarray(ids,dtype=object).reshape(-1,1),
                   roiSources=np.asarray(sources,dtype=object).reshape(-1,1),roiColors=colors,
                   displayTraceNames=np.asarray(trace_names,dtype=object),displayNormalized=bool(normalized),
                   displayShowStim=bool(show_stim),stimTimes=np.asarray(sync.get('stim_times',[]),dtype='float64').reshape(-1,1),
                   stimEndTimes=np.asarray(sync.get('stim_end_times',[]),dtype='float64').reshape(-1,1),sourceFallResult=str(result))
    savemat(paths['mat'],payload,do_compression=True,long_field_names=True)

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    line_styles={'F':'-','Fneu':'--','Fc':'-.','dff':':','spks':(0,(5,2,1,2))}
    fig=Figure(figsize=(12,4.8),facecolor='none'); FigureCanvasAgg(fig); ax=fig.add_subplot(111); ax.set_facecolor('none')
    for row,index in enumerate(indices):
        for name in trace_names:
            values=np.asarray(arrays[name][index],dtype='float64')
            if normalized:
                finite=values[np.isfinite(values)]
                if len(finite): values=(values-finite.min())/max(float(np.ptp(finite)),1e-9)
            ax.plot(times,values,color=colors[row],linestyle=line_styles[name],linewidth=.9,
                    label=f'ROI {index+1} {name}')
    if show_stim:
        starts=np.asarray(sync.get('stim_times',[]),dtype=float); ends=np.asarray(sync.get('stim_end_times',[]),dtype=float)
        for k,onset in enumerate(starts):
            ax.axvline(float(onset),color='#ff5a5f',linestyle='--',linewidth=.65)
            if k<len(ends): ax.axvspan(float(onset),float(ends[k]),color='#ff4850',alpha=.30)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalized display' if normalized else 'Native units / dF/F')
    ax.set_title(f'{len(indices)} selected ROI trace(s) | color = ROI, line style = metric')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,ncol=max(1,int(np.ceil(len(indices)*len(trace_names)/24))))
    fig.tight_layout(); fig.savefig(paths['trace_png'],dpi=220,transparent=True); fig.savefig(paths['trace_pdf'],transparent=True)

    matlab=Path(session.get('cfg',{}).get('matlab') or defaults()['matlab'])
    if not matlab.is_file(): raise RuntimeError('MATLAB is required to create the editable trace FIG file.')
    quote=lambda value:str(value).replace("'","''")
    command=f"addpath('{quote(HERE)}'); studio_export_trace_fig('{quote(paths['mat'])}','{quote(paths['trace_fig'])}');"
    completed=subprocess.run([str(matlab),'-batch',command],cwd=str(HERE),capture_output=True,text=True)
    if completed.returncode or not paths['trace_fig'].is_file():
        raise RuntimeError('MATLAB could not create the trace FIG file.\n'+(completed.stderr or completed.stdout)[-2000:])

    fig=Figure(figsize=(7,7),facecolor='black'); FigureCanvasAgg(fig); ax=fig.add_subplot(111); ax.set_facecolor('black')
    lo,hi=np.percentile(mean,[1,float(display_percentile)]); ax.imshow(mean,cmap='gray',vmin=lo,vmax=max(hi,lo+1))
    from scipy.ndimage import binary_erosion
    rgba=np.zeros((*mean.shape,4),dtype='float32')
    for row,mask in enumerate(masks.astype(bool)):
        edge=mask&~binary_erosion(mask); rgba[mask,:3]=colors[row]; rgba[mask,3]=.35; rgba[edge,:3]=colors[row]; rgba[edge,3]=1.
        yy,xx=np.where(mask)
        if len(xx): ax.text(float(np.median(xx)),float(np.median(yy)),str(indices[row]+1),color=colors[row],ha='center',va='center',fontsize=7,fontweight='bold')
    ax.imshow(rgba); ax.set_axis_off(); ax.set_title('Selected ROI masks | Registered mean',color='white')
    fig.tight_layout(pad=.3); fig.savefig(paths['fov_png'],dpi=220,facecolor=fig.get_facecolor()); fig.savefig(paths['fov_pdf'],facecolor=fig.get_facecolor())
    return {key:str(value) for key,value in paths.items()}

def export_dff_heatmap_fig(values, times, destination, matlab_executable=None, log=print):
    """Create an editable MATLAB FIG for the no-sync dF/F trace heatmap."""
    destination=Path(destination).resolve().with_suffix('.fig')
    if destination.exists(): raise ValueError('Export destination already exists: '+str(destination))
    values=np.asarray(values,dtype='float32'); times=np.asarray(times,dtype='float64').reshape(1,-1)
    if values.ndim!=2 or values.shape[1]!=times.size: raise ValueError('Heatmap values and time axis do not match.')
    cache=STATE/'runtime_cache'; cache.mkdir(exist_ok=True)
    source=cache/f'dff_heatmap_{uuid.uuid4().hex}.mat'
    if hasattr(log,'temporary'): log.temporary(source)
    savemat(source,dict(heatmap=values,time=times),do_compression=True)
    matlab=Path(matlab_executable or defaults()['matlab'])
    if not matlab.is_file():
        source.unlink(missing_ok=True); raise RuntimeError('MATLAB is required to create the editable heatmap FIG file.')
    quote=lambda value:str(value).replace("'","''")
    command=f"addpath('{quote(HERE)}'); studio_export_dff_heatmap_fig('{quote(source)}','{quote(destination)}');"
    try:
        completed=subprocess.run([str(matlab),'-batch',command],cwd=str(HERE),capture_output=True,text=True)
        if completed.returncode or not destination.is_file():
            raise RuntimeError('MATLAB could not create the heatmap FIG file.\n'+(completed.stderr or completed.stdout)[-2000:])
    finally:
        source.unlink(missing_ok=True)
    return str(destination)

def analyze_trial_responses(session, cfg, log=print):
    """Align stimulus trials, baseline-z-score dF/F, sort ROIs, and export a heatmap."""
    if not session.get('sync'): raise ValueError('Import synchronization MAT before trial analysis.')
    result=Path(session['result']).resolve(); dff=np.load(result/'dff.npy',mmap_mode='r')
    stat=session['stat']; accepted=np.asarray([bool(s.get('accepted',True)) for s in stat])
    cell_indices=np.flatnonzero(accepted)
    if not len(cell_indices): raise ValueError('No ROIs are marked as cells.')
    if dff.shape[0]!=len(stat): raise ValueError('ROI identities changed. Recompute traces before trial analysis.')
    ops=np.load(result/'ops.npy',allow_pickle=True).item(); nframes=dff.shape[1]
    times=session.get('frame_times')
    if times is None: times=np.arange(nframes,dtype='float64')/float(ops.get('fs',cfg['fs']))
    times=np.asarray(times,dtype='float64').reshape(-1)
    if len(times)!=nframes: raise ValueError('Frame-time count does not match dF/F traces.')
    sync=session['sync']; starts=np.asarray(sync.get('stim_times',[]),dtype='float64').reshape(-1)
    ends=np.asarray(sync.get('stim_end_times',[]),dtype='float64').reshape(-1)
    if not len(starts): raise ValueError('No stimulus intervals were found in synchronization data.')
    baseline=float(cfg.get('trial_baseline_seconds',.5)); post=float(cfg.get('trial_post_seconds',10.))
    if baseline<=0 or post<=0: raise ValueError('Trial baseline and post-stimulus windows must be positive.')
    dt=1./float(cfg['fs']); relative=np.arange(-baseline,post+dt*.25,dt,dtype='float64')
    valid=np.flatnonzero((starts-baseline>=times[0])&(starts+post<=times[-1]))
    if not len(valid): raise ValueError('No complete trials fit inside the imaging time range.')
    if len(valid)<len(starts): log(f'Trial analysis: omitted {len(starts)-len(valid)} incomplete edge trials.')
    starts_used=starts[valid]; ends_used=ends[valid] if len(ends)==len(starts) else np.full(len(valid),np.nan)
    trial_z=np.empty((len(cell_indices),len(valid),len(relative)),dtype='float32')
    baseline_mask=relative<0
    for trial,onset in enumerate(starts_used):
        sample_times=onset+relative
        values=np.vstack([np.interp(sample_times,times,np.asarray(dff[index],dtype='float64')) for index in cell_indices])
        base=values[:,baseline_mask]; mean=base.mean(1,keepdims=True); std=base.std(1,ddof=1,keepdims=True)
        std[~np.isfinite(std)|(std<1e-6)]=np.nan
        trial_z[:,trial,:]=((values-mean)/std).astype('float32')
    mean_z=np.nanmean(trial_z,axis=1)
    durations=ends_used-starts_used; finite_durations=durations[np.isfinite(durations)&(durations>0)]
    stim_duration=float(np.median(finite_durations)) if len(finite_durations) else min(1.,post)
    response_mask=(relative>=0)&(relative<=min(stim_duration,post))
    response_score=np.nanmean(mean_z[:,response_mask],axis=1)
    order=np.argsort(np.nan_to_num(response_score,nan=-np.inf))[::-1]
    sorted_z=mean_z[order]; sorted_indices=cell_indices[order]
    output=result/'trial_analysis'; output.mkdir(exist_ok=True)
    mat_path=output/'trial_response_heatmap.mat'; png_path=output/'trial_response_heatmap.png'; pdf_path=output/'trial_response_heatmap.pdf'
    fig_path=output/'trial_response_heatmap.fig'
    payload=dict(trialZ=trial_z,meanZ=mean_z,sortedMeanZ=sorted_z,time=relative.reshape(1,-1),
                 responseScore=response_score.reshape(-1,1),sortOrder=(order+1).reshape(-1,1),
                 roiIndices=(cell_indices+1).reshape(-1,1),sortedRoiIndices=(sorted_indices+1).reshape(-1,1),
                 roiIds=np.asarray([stat[i]['roi_id'] for i in cell_indices],dtype=object),
                 stimulusOnsets=starts_used.reshape(-1,1),stimulusOffsets=ends_used.reshape(-1,1),
                 baselineSeconds=baseline,postSeconds=post,stimulusDuration=stim_duration,
                 zScoreDefinition='per ROI and trial: (dff - pre-stimulus mean) / pre-stimulus sample SD')
    savemat(mat_path,payload,do_compression=True,long_field_names=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(11,8),facecolor='white'); FigureCanvasAgg(fig)
    grid=fig.add_gridspec(3,1,height_ratios=[.55,1.1,5.2],hspace=.08)
    stim_ax=fig.add_subplot(grid[0]); trace_ax=fig.add_subplot(grid[1],sharex=stim_ax); heat_ax=fig.add_subplot(grid[2],sharex=stim_ax)
    stim_stop=min(stim_duration,post); stim_ax.fill_between([0,stim_stop],[0,0],[1,1],color='#ef3b46',alpha=.45)
    stim_ax.plot([relative[0],0,0,stim_stop,stim_stop,relative[-1]],[0,0,1,1,0,0],color='#e31a1c',linewidth=1.2)
    stim_ax.set_ylim(-.05,1.05); stim_ax.set_yticks([]); stim_ax.set_ylabel('Stimulus',rotation=0,labelpad=35,va='center')
    population=np.nanmean(mean_z,axis=0); trace_ax.plot(relative,population,color='black',linewidth=1.2)
    trace_ax.axvline(0,color='#e31a1c',linewidth=.8); trace_ax.set_ylabel('Mean z-score')
    finite=sorted_z[np.isfinite(sorted_z)]; limit=float(np.percentile(np.abs(finite),98)) if len(finite) else 1.; limit=max(limit,.5)
    image=heat_ax.imshow(sorted_z,aspect='auto',interpolation='nearest',origin='upper',
                         extent=[relative[0],relative[-1],len(sorted_z)+.5,.5],cmap='RdBu_r',vmin=-limit,vmax=limit)
    heat_ax.axvline(0,color='black',linewidth=.7,alpha=.7); heat_ax.set_ylabel('Cell ROIs (response-sorted)'); heat_ax.set_xlabel('Time from stimulus onset (s)')
    colorbar=fig.colorbar(image,ax=heat_ax,pad=.015); colorbar.set_label('dF/F z-score')
    fig.suptitle(f'{len(cell_indices)} cells | {len(valid)} trials | baseline −{baseline:g} to 0 s | sorted by mean stimulus response',fontsize=11)
    stim_ax.tick_params(labelbottom=False); trace_ax.tick_params(labelbottom=False); fig.subplots_adjust(left=.10,right=.92,top=.94,bottom=.08)
    fig.savefig(png_path,dpi=220); fig.savefig(pdf_path)
    matlab=Path(cfg.get('matlab') or defaults()['matlab'])
    if not matlab.is_file(): raise RuntimeError('MATLAB is required to create the editable heatmap FIG file.')
    quote=lambda value:str(value).replace("'","''")
    command=f"addpath('{quote(HERE)}'); studio_export_heatmap_fig('{quote(mat_path)}','{quote(fig_path)}');"
    completed=subprocess.run([str(matlab),'-batch',command],cwd=str(HERE),capture_output=True,text=True)
    if completed.returncode or not fig_path.is_file():
        raise RuntimeError('MATLAB could not create the heatmap FIG file.\n'+(completed.stderr or completed.stdout)[-2000:])
    log(f'Trial heatmap: {len(cell_indices)} cells, {len(valid)} trials, baseline {baseline:g} s; saved to {output}')
    return dict(mat=str(mat_path),png=str(png_path),pdf=str(pdf_path),fig=str(fig_path),cells=int(len(cell_indices)),trials=int(len(valid)))

def validate_times(times, meta, cfg):
    times=np.asarray(times, dtype='float64').reshape(-1)
    if len(times)!=meta['shape'][0] or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Expected one finite, strictly increasing timestamp per frame.')
    for start,end in intervals_from_meta(meta,cfg.get('continuous_files',True)):
        dt=np.diff(times[start:end])
        if len(dt) and np.max(np.abs(dt*cfg['fs']-1))>.1:
            raise ValueError('Within-session timestamps disagree with fs or contain frame gaps. Resolve this before deconvolution.')
    return times

def _ttl_edges(signal, debounce_samples=0):
    signal=np.asarray(signal,dtype='float64').reshape(-1)
    if signal.size==0 or not np.isfinite(signal).all():
        raise ValueError('Synchronization channel is empty or contains nonfinite values.')
    threshold=(float(signal.min())+float(signal.max()))/2.0
    if float(signal.max())<=float(signal.min()): raise ValueError('Synchronization TTL channel is constant.')
    padded=np.r_[False,signal>threshold,False].astype('int8'); changes=np.diff(padded)
    rising=np.flatnonzero(changes==1).astype('int64')
    falling=np.flatnonzero(changes==-1).astype('int64')
    if debounce_samples>0 and len(rising)>1:
        merged_rising=[int(rising[0])]; merged_falling=[int(falling[0])]
        for start,end in zip(rising[1:],falling[1:]):
            if int(start)-merged_falling[-1]<int(debounce_samples): merged_falling[-1]=int(end)
            else: merged_rising.append(int(start)); merged_falling.append(int(end))
        rising=np.asarray(merged_rising,dtype='int64'); falling=np.asarray(merged_falling,dtype='int64')
    return rising,falling,float(threshold)

def read_sync_mat(path, expected_frames, cfg):
    """Read configurable frame and stimulus TTL channels, including pulse ends."""
    path=str(Path(path).resolve())
    variable=str(cfg.get('sync_variable','data2026')).strip()
    frame_col=int(cfg.get('sync_frame_column',1)); stim_col=int(cfg.get('sync_stim_column',2))
    ni_rate=float(cfg.get('sync_sample_rate',1000.)); frame_rate=float(cfg.get('fs',23.2))
    try:
        loaded=loadmat(path,variable_names=[variable],squeeze_me=False)
        if variable not in loaded: raise KeyError(variable)
        data=loaded[variable]
    except Exception as exc:
        raise ValueError(f'Cannot read MATLAB variable {variable!r} from synchronization MAT: {path}\n{exc}') from exc
    data=np.asarray(data)
    required=max(frame_col,stim_col)
    if data.ndim!=2 or data.shape[1]<required or min(frame_col,stim_col)<1:
        raise ValueError(f'{variable} must be a 2-D matrix with at least {required} columns; got {data.shape}. Columns are 1-based.')
    debounce=max(1,int(round(ni_rate*.010)))
    frame_samples,_,frame_threshold=_ttl_edges(data[:,frame_col-1],debounce)
    stim_samples,stim_end_samples,stim_threshold=_ttl_edges(data[:,stim_col-1],debounce)
    alignment=str(cfg.get('sync_alignment','strict'))
    if alignment not in ('strict','first_edge'):
        raise ValueError(f'Unknown synchronization alignment mode: {alignment}')
    if expected_frames<1 or ni_rate<=0 or frame_rate<=0:
        raise ValueError('Synchronization requires positive TIFF frame count, NI sample rate and imaging frame rate.')
    if not len(frame_samples):
        raise ValueError(f'No frame TTL rising edge detected in {Path(path).name}: {variable} column {frame_col}.')
    trimmed_terminal_frame=False
    if alignment=='strict':
        if len(frame_samples)==int(expected_frames)+1:
            frame_samples=frame_samples[:-1]; trimmed_terminal_frame=True
        if len(frame_samples)!=int(expected_frames):
            raise ValueError(f'Synchronization mismatch for {Path(path).name}: {variable} column {frame_col} has {len(frame_samples)} rising edges after 10 ms debounce, expected {expected_frames} TIFF frames. Select first-rise alignment in Settings only if frame 1 really coincides with the first TTL.')
        mapping_samples=frame_samples
        frame_times=(frame_samples-frame_samples[0])/float(ni_rate)
    else:
        # Anchor the TIFF movie to the first observed TTL. Later frames use nominal fs;
        # missing/noisy TTLs must never silently shift subsequent frame indices.
        frame_times=np.arange(int(expected_frames),dtype='float64')/frame_rate
        mapping_samples=frame_samples[0]+frame_times*ni_rate
    # Latest acquired imaging frame at each stimulus onset, matching TPM MATLAB.
    stim_frames=np.searchsorted(mapping_samples,stim_samples,side='right')
    stim_frames=np.clip(stim_frames,1,int(expected_frames)).astype('int64')
    stim_end_frames=np.searchsorted(mapping_samples,stim_end_samples,side='right')
    stim_end_frames=np.clip(stim_end_frames,1,int(expected_frames)).astype('int64')
    stim_times=(stim_samples-frame_samples[0])/float(ni_rate)
    stim_end_times=(stim_end_samples-frame_samples[0])/float(ni_rate)
    return dict(frame_times=frame_times.astype('float64'),stim_frames=stim_frames,
                stim_end_frames=stim_end_frames,stim_times=stim_times.astype('float64'),
                stim_end_times=stim_end_times.astype('float64'),frame_ttl_samples=frame_samples,
                stim_ttl_samples=stim_samples,stim_end_ttl_samples=stim_end_samples,sync_file=path,
                sync_variable=variable,frame_column=frame_col,stim_column=stim_col,
                frame_threshold=frame_threshold,stim_threshold=stim_threshold,
                debounce_samples=debounce,trimmed_terminal_frame=trimmed_terminal_frame,
                alignment_mode=alignment,expected_frames=int(expected_frames),
                sync_mode=(f'NI {variable}: first TIFF frame anchored to first rising edge of column {frame_col}; later frames at nominal {frame_rate:g} Hz; column {stim_col} stimulus intervals'
                           if alignment=='first_edge' else
                           f'NI {variable}: column {frame_col} frame TTL / column {stim_col} stimulus high intervals'),
                ni_sample_rate=float(ni_rate),frame_rate=float(frame_rate))

def read_sync_files(paths, meta, cfg):
    """Read one combined MAT or one MAT per imported TIFF segment."""
    paths=[str(Path(p).resolve()) for p in paths]
    records=meta.get('records',[]); total=int(meta['shape'][0])
    if len(paths)==1:
        return read_sync_mat(paths[0],total,cfg)
    if len(paths)!=len(records):
        raise ValueError('Select one combined synchronization MAT, or exactly one synchronization MAT for each imported TIFF stack.')
    frame_times=[]; stim_frames=[]; stim_end_frames=[]; stim_times=[]; stim_end_times=[]
    frame_samples=[]; stim_samples=[]; stim_end_samples=[]; offset_frames=0; offset_time=0.; items=[]
    for path,record in zip(paths,records):
        item=read_sync_mat(path,int(record['frames']),cfg); items.append(item)
        frame_times.append(item['frame_times']+offset_time)
        stim_frames.append(item['stim_frames']+offset_frames)
        stim_end_frames.append(item['stim_end_frames']+offset_frames)
        stim_times.append(item['stim_times']+offset_time)
        stim_end_times.append(item['stim_end_times']+offset_time)
        frame_samples.append(item['frame_ttl_samples']); stim_samples.append(item['stim_ttl_samples']); stim_end_samples.append(item['stim_end_ttl_samples'])
        offset_frames+=int(record['frames']); offset_time=float(frame_times[-1][-1])+1/float(cfg['fs'])
    return dict(frame_times=np.concatenate(frame_times),stim_frames=np.concatenate(stim_frames),
                stim_end_frames=np.concatenate(stim_end_frames),stim_times=np.concatenate(stim_times),
                stim_end_times=np.concatenate(stim_end_times),frame_ttl_samples=np.concatenate(frame_samples),
                stim_ttl_samples=np.concatenate(stim_samples),stim_end_ttl_samples=np.concatenate(stim_end_samples),
                sync_file=paths,sync_variable=items[0]['sync_variable'],frame_column=items[0]['frame_column'],
                stim_column=items[0]['stim_column'],frame_threshold=items[0]['frame_threshold'],
                stim_threshold=items[0]['stim_threshold'],debounce_samples=items[0]['debounce_samples'],
                trimmed_terminal_frame=any(item['trimmed_terminal_frame'] for item in items),
                alignment_mode=items[0]['alignment_mode'],expected_frames=total,
                sync_mode=items[0]['sync_mode']+' (per TIFF)',
                ni_sample_rate=float(cfg.get('sync_sample_rate',1000.)),frame_rate=float(cfg['fs']))
