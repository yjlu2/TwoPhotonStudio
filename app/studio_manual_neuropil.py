"""TwoPhotonStudio manual neuropil re-extraction; no automatic mask expansion."""
from pathlib import Path
import copy
import json
import shutil
import subprocess
import numpy as np
import h5py
from scipy.io import savemat
from scipy.stats import skew

def reextract_manual_neuropil(result, fall2, destination, matlab, log=print):
    from suite2p.extraction.extract import extract_traces_from_masks
    from suite2p.extraction.masks import create_cell_mask
    from suite2p.extraction import dcnv
    result, fall2, destination = map(Path, (result, fall2, destination))
    if destination.exists():
        raise ValueError('Choose a new result directory.')
    ops = np.load(result/'ops.npy', allow_pickle=True).item()
    stat = copy.deepcopy(np.load(result/'stat.npy', allow_pickle=True))
    arrays = {n: np.load(result/(n+'.npy')).copy() for n in ('F','Fneu','Fc','F0','dff','spks','iscell')}
    def readtext(f, obj):
        return ''.join(chr(int(c)) for c in np.asarray(obj).ravel())
    with h5py.File(fall2, 'r') as f:
        s = f['manualNeuropil']
        rows = np.asarray(s['roiRows']).astype(int).ravel()-1
        masks = np.asarray(s['neuropilMasks']).transpose(2,1,0).astype(bool)
        # MATLAB string arrays use opaque MCOS storage; roi_ids is an ordinary cell array.
        refs=np.asarray(f['roi_ids']).ravel()
        ids = [readtext(f, f[ref]) for ref in refs]
        if len(refs)!=len(stat) or np.any(rows<0) or np.any(rows>=len(stat)):
            raise ValueError('Manual ROI rows do not match this session.')
        if len(rows)!=len(set(rows)) or masks.shape!=(int(ops['Ly']),int(ops['Lx']),len(rows)):
            raise ValueError('Invalid manual mask dimensions or ROI rows.')
        if any(str(s['roi_id'])!=rid for s,rid in zip(stat,ids)):
            raise ValueError('Manual ROI identities do not match this session.')
        if not np.array_equal(arrays['F'], np.asarray(f['F']).T):
            raise ValueError('Fall-2.mat F does not match the current extraction result.')
        # Fneu can legitimately differ after drawing/exporting a new mask set.
        # Fall-2 is the baseline being patched; keep every unselected row as saved there.
        if not np.array_equal(arrays['Fneu'], np.asarray(f['Fneu']).T):
            log('Fall-2.mat contains another neuropil extraction revision; matching ROI IDs and F verified. Re-extracting the new masks.')
        for name in arrays:
            value=np.asarray(f[name]).T
            if value.shape!=arrays[name].shape:
                raise ValueError('Fall-2.mat has incompatible '+name+' dimensions.')
            arrays[name]=value.copy()
        statrefs=np.asarray(f['stat']).ravel()
        for i,ref in enumerate(statrefs):
            saved=f[ref]
            stat[i]['neuropil_mask']=np.asarray(saved['neuropil_mask']).ravel().astype('int64')
            for name in ('std','skew'):
                if name in saved: stat[i][name]=float(np.asarray(saved[name]).ravel()[0])
            stat[i].pop('neuropil_source',None)
    neuropil = [np.flatnonzero(masks[:,:,i].ravel()).astype('int64') for i in range(len(rows))]
    if any(len(p)==0 for p in neuropil):
        raise ValueError('A manual neuropil mask is empty.')
    # Keep the exact saved union, including its cell/border exclusions.
    extract_ops = copy.deepcopy(ops)
    if not Path(extract_ops['reg_file']).exists():
        candidate = result.parent/Path(str(extract_ops['reg_file'])).parent.name/'registered.bin'
        if not candidate.exists(): raise FileNotFoundError(extract_ops['reg_file'])
        extract_ops['reg_file']=str(candidate)
    extract_ops['batch_size']=min(int(ops.get('batch_size',100)),100)
    cell=[create_cell_mask(stat[r],int(ops['Ly']),int(ops['Lx']),False) for r in rows]
    log('TwoPhotonStudio / native Suite2p: extracting %d manual neuropil masks from registered movie.' % len(rows))
    _, new_neu, _, _, _=extract_traces_from_masks(extract_ops,cell,neuropil)
    fc=arrays['F'][rows]-float(ops['neucoeff'])*new_neu
    df=np.empty_like(fc); spks=np.empty_like(fc)
    for start,end in np.asarray(ops['session_intervals'],dtype=int):
        part=dcnv.preprocess(F=fc[:,start:end].copy(),baseline='maximin',win_baseline=float(ops['win_baseline']),sig_baseline=float(ops['sig_baseline']),fs=float(ops['fs']))
        df[:,start:end]=part
        spks[:,start:end]=dcnv.oasis(F=part.copy(),batch_size=100,tau=float(ops['tau']),fs=float(ops['fs']))
    baseline=fc-df
    changes=dict(Fneu=new_neu,Fc=fc,F0=baseline,dff=np.divide(df,baseline,out=np.full_like(df,np.nan),where=baseline>1e-6),spks=spks)
    for name,value in changes.items(): arrays[name][rows]=value
    for i,r in enumerate(rows):
        stat[r].update(neuropil_mask=neuropil[i],neuropil_source='manual',std=float(np.std(fc[i])),skew=float(skew(fc[i])))
    ops.update(manual_neuropil_rows=rows+1,manual_neuropil_source=str(fall2),manual_neuropil_extraction='TwoPhotonStudio / Suite2p 0.10.3 CPU; exact hand-drawn union masks')
    destination.mkdir()
    for p in result.iterdir():
        if p.is_file() and p.suffix in ('.json','.npy'): shutil.copy2(p,destination/p.name)
    for name,value in arrays.items(): np.save(destination/(name+'.npy'),value)
    np.save(destination/'stat.npy',stat); np.save(destination/'ops.npy',ops)
    np.save(destination/'neuropil_counts.npy',np.array([len(s['neuropil_mask']) for s in stat]))
    mask_cells=np.empty((len(rows),1),dtype=object)
    for i,p in enumerate(neuropil): mask_cells[i,0]=p.reshape(1,-1)
    savemat(destination/'manual_Fneu_patch.mat',dict(changes,roiRows=(rows+1).reshape(-1,1),neuropilIndices=mask_cells,roiStd=np.std(fc,axis=1),roiSkew=skew(fc,axis=1),nativeResult=str(destination)))
    helper=Path(__file__).with_name('apply_studio_manual_Fneu.m')
    quote=lambda p: str(p).replace("'","''")
    command="addpath('%s'); apply_studio_manual_Fneu('%s','%s');" % (quote(helper.parent),quote(fall2),quote(destination/'manual_Fneu_patch.mat'))
    # MATLAB preserves all unrelated fields and the v7.3 structure.
    run=subprocess.run([str(matlab),'-batch',command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
    log(run.stdout)
    if run.returncode: raise RuntimeError('MATLAB update failed. Original Fall-2.mat is retained; new native results remain in '+str(destination))
    log('Updated Fall-2.mat and native result: '+str(destination))
    return str(destination)
