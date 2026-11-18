"""Meaningful numerical/integration checks. Synthetic inputs only."""
import json
import sys
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import shift
from scipy.io import loadmat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import backend as B

def main(base):
    base=Path(base); base.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(47); n,h,w=240,96,112
    yy,xx=np.mgrid[:h,:w]
    reference=500+100*rng.random((h,w))
    cells=[(27,28),(66,72),(40,86)]
    movie=[]
    for t in range(n):
        frame=reference.copy()
        for i,(y,x) in enumerate(cells):
            activity=600*np.exp(-((t-(40+i*30))%100)/10)
            frame+=(1400+activity)*np.exp(-((yy-y)**2+(xx-x)**2)/18)
        frame+=rng.normal(0,15,(h,w))
        movie.append(np.clip(frame,0,65535).astype('uint16'))
    movie=np.array(movie)
    for name,frames in [('stack2.tif',movie[:120]),('stack10.tif',movie[120:])]:
        tifffile.imwrite(base/name,frames,photometric='minisblack')
    root=base/'analysis'
    meta=B.import_tiffs([base/'stack10.tif',base/'stack2.tif'],root)
    assert Path(meta['files'][0]).name=='stack2.tif'
    assert meta['frame_map'][120]==[121,2,1]
    cfg=B.defaults(); cfg.update(registration='already',suite2p_backend='cpu',registered_tiff_mode='none',fs=10.,tau=1.,min_neuropil_pixels=100,continuous_files=False)
    opath=B.prepare(root,cfg); ops=np.load(opath,allow_pickle=True).item()
    assert np.array_equal(B.binary_movie(ops),movie//2)
    stat=[]
    for i,(y,x) in enumerate(cells[:2]):
        mask=(yy-y)**2+(xx-x)**2<=16; ys,xs=np.where(mask)
        stat.append(dict(ypix=ys,xpix=xs,lam=np.ones(len(ys),'float32'),source='manual',
                         roi_id=f'test{i}',accepted=True,reviewed=True,probability=np.nan))
    result=B.extract(opath,stat,cfg,base/'traces')
    F=np.load(Path(result)/'F.npy'); Fneu=np.load(Path(result)/'Fneu.npy')
    saved=np.load(Path(result)/'stat.npy',allow_pickle=True)
    for i,s in enumerate(saved):
        expected=(movie//2)[:,s['ypix'],s['xpix']].mean(1)
        np.testing.assert_allclose(F[i],expected,rtol=2e-6)
        np.testing.assert_allclose(Fneu[i],(movie//2).reshape(n,-1)[:,s['neuropil_mask']].mean(1),rtol=2e-6)
    from suite2p.extraction import dcnv
    Fc=F-cfg['neucoeff']*Fneu
    expected_df=dcnv.preprocess(F=Fc[:,120:].copy(),baseline='maximin',win_baseline=60.,sig_baseline=10.,fs=10.)
    expected_spks=dcnv.oasis(F=expected_df.copy(),batch_size=100,tau=1.,fs=10.)
    np.testing.assert_allclose(np.load(Path(result)/'spks.npy')[:,120:],expected_spks)
    session=dict(root=str(root),ops_file=opath,stat=list(saved),result=result,frame_times=None)
    B.export_mat(session,base/'Fall.mat'); mat=loadmat(base/'Fall.mat')
    assert mat['F'].shape==(2,n) and mat['frameMap'][120,1]==2
    np.testing.assert_array_equal(mat['F'],F)
    # Added ROI changes shared neuropil; all three rows must be freshly extracted.
    y,x=cells[2]; ys,xs=np.where((yy-y)**2+(xx-x)**2<=16)
    added=dict(ypix=ys,xpix=xs,lam=np.ones(len(ys),'float32'),source='manual',roi_id='test2',accepted=True,reviewed=True)
    merged,rejected=B.merge_candidates(stat,[stat[0],added],w)
    assert len(merged)==3 and rejected==1
    result2=B.extract(opath,merged,cfg,base/'traces_added')
    assert np.load(Path(result2)/'F.npy').shape==(3,n)
    # Real native functional detector.
    found=B.detect(opath,cfg,'suite2p')
    assert len(found)>0
    B.write_json(base/'checks.json',dict(multi_tiff_order=True,frame_map=True,
        native_extraction_numerical=True,deconvolution_session_boundaries=True,
        matlab_export_roundtrip=True,manual_add_recompute=True,native_detection_count=len(found),
        ops_file=opath,result=result2))
    print('CORE CHECKS PASSED',flush=True)

if __name__=='__main__': main(sys.argv[1])

