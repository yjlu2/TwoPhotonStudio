"""Fixed session-wide calibration for Suite2p's int16 working movie."""
from pathlib import Path
import numpy as np
import tifffile
from tiff_frames import frames

def supported_dtype(dtype):
    dtype=np.dtype(dtype)
    return (dtype.kind in 'ui' and dtype.itemsize in (2,4)) or (dtype.kind=='f' and dtype.itemsize==4)

def prepare_inputs(paths,folder,log=print):
    types=[]
    for path in paths:
        with tifffile.TiffFile(path) as tif: types.append(tif.pages[0].dtype)
    if all(dtype==np.dtype('uint16') for dtype in types): return paths,None
    low=0.; high=0.
    log('Scanning 16/32-bit TIFF intensities for one shared, fixed calibration...')
    for path in paths:
        for i,frame in enumerate(frames(path),1):
            if not np.isfinite(frame).all():
                raise ValueError(f'TIFF contains NaN or Inf: {path}, frame {i}. Clean nonfinite pixels before analysis.')
            low=min(low,float(np.min(frame))); high=max(high,float(np.max(frame)))
        log(f'Intensity scan complete: {Path(path).name}; session range [{low:g}, {high:g}]')
    if high<=low: raise ValueError('TIFF session contains no intensity variation.')
    gain=(high-low)/32760.
    converted=[]
    for index,path in enumerate(paths,1):
        output=Path(folder)/f'input_calibrated_{index}.tif'
        with tifffile.TiffWriter(output,bigtiff=True) as writer:
            for i,frame in enumerate(frames(path),1):
                work=np.clip(np.rint((np.asarray(frame,dtype='float64')-low)/gain),0,32760)
                writer.write((work.astype('uint16')*2),photometric='minisblack',metadata=None)
                if i%500==0: log(f'Calibrated TIFF: {Path(path).name}, {i} frames')
        converted.append(str(output))
    transform=dict(gain=float(gain),offset=float(low),input_dtypes=[str(d) for d in types],
                   convention='original intensity = working int16 intensity * gain + offset',
                   maximum_quantization_error=float(gain/2))
    log(f'Fixed intensity calibration: original = working * {gain:g} + {low:g}; traces and registered TIFF will be restored to original units.')
    return converted,transform
