"""Read ordinary multi-page TIFFs and ImageJ contiguous single-IFD stacks."""
from pathlib import Path
import numpy as np
import tifffile


def layout(tif):
    """Return (number of frames, height/width, dtype, is_contiguous_stack)."""
    if tif.is_imagej and len(tif.pages) == 1 and len(tif.series) == 1:
        series = tif.series[0]
        if len(series.shape) == 3 and series.axes.endswith('YX'):
            return int(series.shape[0]), tuple(series.shape[-2:]), series.dtype, True
    return len(tif.pages), None, None, False


def frames(path):
    """Yield frames without materializing a potentially multi-gigabyte stack."""
    with tifffile.TiffFile(path) as tif:
        count, _, _, contiguous = layout(tif)
        if contiguous:
            try:
                movie = tifffile.memmap(path)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'ImageJ contiguous TIFF cannot be memory-mapped: {path}') from exc
            if movie.ndim != 3 or movie.shape[0] != count:
                raise ValueError(f'ImageJ TIFF metadata and pixel layout disagree: {path}')
            for frame in movie:
                yield frame
        else:
            for page in tif.pages:
                yield page.asarray()


def normalize_for_page_readers(path, destination, log=print):
    """Create a conventional multi-page TIFF only for backends requiring pages."""
    destination = Path(destination)
    with tifffile.TiffFile(path) as tif:
        count, _, _, contiguous = layout(tif)
    if not contiguous:
        return str(path)
    log(f'Converting ImageJ single-IFD stack to multi-page TIFF for registration: {Path(path).name} ({count} frames). This requires temporary disk space.')
    with tifffile.TiffWriter(destination, bigtiff=True) as writer:
        for index, frame in enumerate(frames(path), 1):
            writer.write(np.asarray(frame), photometric='minisblack')
            if index % 500 == 0:
                log(f'Converted {index}/{count} frames: {Path(path).name}')
    return str(destination)
