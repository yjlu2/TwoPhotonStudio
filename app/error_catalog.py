"""User-facing diagnoses. Technical traceback text belongs only in log files."""
from pathlib import Path
import tempfile
import uuid


def open_diagnostic_log(path=None):
    candidates=[Path(path)] if path is not None else []
    candidates.append(Path(tempfile.gettempdir())/'TwoPhotonStudio_logs'/(uuid.uuid4().hex+'.log'))
    last=None
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True,exist_ok=True)
            return candidate,candidate.open('w',encoding='utf-8',buffering=1)
        except OSError as exc: last=exc
    raise last


def write_diagnostic_log(detail,path=None):
    actual,handle=open_diagnostic_log(path)
    with handle: handle.write(detail)
    return actual

# First matching rule wins. Specific cases precede broad exception categories.
# code, level, title, signatures, cause, next action
RULES=[
 ('OUT-01','Warning','Output location cannot be used',
  ('output relocation requires','unfinished output copy','output relocation cannot','output-copy checksum','structured object-array metadata'),
  'The selected output cannot be used safely for this session copy.',
  'Choose a new folder outside the current session. The original session is retained; check the log for copy details.'),
 ('DROP-01','Warning','Unsupported dropped item',
  ('dropped npy is not','dropped folder contains no','drop only one session','drop one saved session'),
  'The dropped item is not one saved session or a supported TIFF import.',
  'Drop a session.npy file by itself, a session folder, or single-FOV TIFF files/folders.'),
 ('GPU-01','Error','GPU memory exhausted',
  ('cuda out of memory','cuda_error_out_of_memory','cublas_status_alloc_failed','torch.outofmemoryerror'),
  'The GPU could not allocate memory for this step.',
  'Close other GPU jobs. Retry with smaller batches or use CPU.'),
 ('GPU-02','Error','GPU architecture unsupported',
  ('no kernel image','not compatible with the current pytorch','unsupported gpu architecture'),
  'The installed PyTorch build does not support this GPU.',
  'Use Suite2p CPU or install a matching PyTorch/CUDA runtime.'),
 ('GPU-03','Error','GPU unavailable',
  ('cuda is unavailable','no nvidia driver','cuda driver version is insufficient','torch not compiled with cuda','cuda initialization error'),
  'CUDA or the NVIDIA driver is unavailable in the selected Python environment.',
  'Check the driver and Python runtime in Settings, or choose CPU.'),
 ('IO-01','Error','Disk space exhausted',
  ('no space left','disk full','not enough space','disk quota','errno 28','winerror 112','winerror 39','磁盘空间不足','磁盘已满'),
  'An output, temporary or cache disk has insufficient free space.',
  'Free space on the affected disk or choose a different output folder.'),
 ('MEM-01','Error','Memory allocation failed',
  ('memoryerror','bad allocation','unable to allocate','out of memory','paging file is too small','winerror 1455'),
  'The system or a model could not allocate memory; this message does not identify a specific device.',
  'Close memory-heavy programs. Check RAM, GPU memory and Windows paging-file space before retrying.'),
 ('IO-02','Error','File is in use',
  ('winerror 32','sharing violation','being used by another process'),
  'Another process is locking a required file.',
  'Close the file in MATLAB, ImageJ or other programs, then retry.'),
 ('IO-03','Error','Access denied',
  ('permissionerror','permission denied','access is denied','winerror 5','拒绝访问'),
  'The file or folder is not writable, or an application is locking it.',
  'Check folder permissions and close open output files; choose a writable local folder if needed.'),
 ('IO-04','Error','Path unavailable',
  ('filenotfounderror','winerror 2','winerror 3','winerror 53','winerror 67','no such file','network path was not found','cannot find the path'),
  'An input file, output folder, mapped drive or runtime cannot be found.',
  'Reconnect the drive or correct the path in Sources or Settings. See the log for the exact path.'),
 ('IO-05','Warning','Output already exists',
  ('destination exists','destination already exists','fileexistserror','export destination already exists','already exists:'),
  'The selected output filename already exists.',
  'Choose another name or use Replace where available.'),
 ('TIFF-01','Error','Unsupported TIFF layout',
  ('expected 2d','expected uint16','expected 16-bit','separate channels/planes','not a tiff'),
  'The file is not a supported single-channel 2D frame stack.',
  'Import a 16-bit integer or 32-bit integer/float grayscale TIFF; separate channels and planes first.'),
 ('TIFF-02','Error','Stack sizes differ',
  ('same dimensions','fov dimensions differ','same fov'),
  'The imported stacks do not share the same image dimensions or FOV.',
  'Select stacks from the same FOV, plane and channel, with matching dimensions.'),
 ('TIFF-03','Warning','Too few imaging frames',
  ('at least 50','at least two frames','need at least one array to stack','no binned frames'),
  'There are too few frames, or too few usable frames after temporal binning.',
  'Check the imported frame count. Import the complete movie; use Already registered for a registered movie.'),
 ('TIFF-04','Error','Source changed',
  ('source tiff changed','source frame count changed','registration changed the frame count'),
  'A source or registered movie no longer matches its recorded frame mapping.',
  'Restore the original file or create a new import. Do not reuse old traces with a changed movie.'),
 ('REG-01','Error','No common registration area',
  ('no common valid area','excessive registration displacement'),
  'Motion leaves no valid area shared by the registered frames.',
  'Inspect motion and registration settings; exclude corrupted input frames before rerunning.'),
 ('ROI-01','Warning','ROI outside analysis area',
  ('roi crosses','outside the valid','outside the common valid'),
  'The ROI crosses the current valid registration boundary.',
  'Move the ROI inside the yellow border or review Edit area before changing the boundary.'),
 ('ROI-02','Error','Overlapping or invalid ROI masks',
  ('fully overlapped','finite non-overlapping','roi cleanup','duplicate candidates','strong overlap'),
  'The masks contain duplicates, full overlap or invalid weights.',
  'Remove duplicate or empty ROIs, then recompute traces.'),
 ('ROI-03','Error','Insufficient neuropil pixels',
  ('insufficient valid neuropil','neuropil pixels'),
  'The current masks and valid area leave too few pixels for neuropil extraction.',
  'Inspect dense ROIs and valid area, or reduce the minimum neuropil pixels in Settings.'),
 ('SYNC-01','Error','Frame TTL count mismatch',
  ('synchronization mismatch','frame-time count does not match','timestamp per frame'),
  'The synchronization frame count does not match the imaging movie.',
  'Check the MAT variable, frame column, NI sample rate and TIFF selection. Use first-rise alignment only if its assumptions hold.'),
 ('SYNC-02','Error','Frame timing mismatch',
  ('timestamps disagree','frame gaps','strictly increasing'),
  'Frame times do not agree with the imaging frame rate or are not increasing.',
  'Correct fs and synchronization settings, then import the MAT file again.'),
 ('SYNC-03','Error','Invalid synchronization channels',
  ('cannot read matlab variable','must be a 2-d matrix','ttl channel is constant','no frame ttl','synchronization channel is empty','synchronization requires positive'),
  'The selected MAT matrix or TTL column is missing, constant or invalid.',
  'Check matrix name, 1-based column numbers and NI sample rate in Settings.'),
 ('SYNC-04','Warning','Synchronization not ready',
  ('import synchronization mat','no stimulus intervals','no complete trials','select one combined synchronization'),
  'The trial analysis has no complete stimulus intervals aligned to imaging.',
  'Import the matching synchronization MAT; check trial windows and stimulus columns.'),
 ('MODEL-01','Error','PROFILE configuration unavailable',
  ('profile model is not configured','profile config does not exist','cannot read profile json','profile 2d model config','profile repository is missing','profile config has no checkpoint','profile checkpoint does not exist','profile python executable does not exist'),
  'A PROFILE config, matching checkpoint or inference environment is missing or incompatible.',
  'Choose the matching 2D JSON and checkpoint and check the PROFILE paths in Settings.'),
 ('MODEL-02','Error','Model weights incompatible',
  ('size mismatch for','missing key(s) in state_dict','unexpected key(s)','select a downloaded cellpose checkpoint','unpicklingerror'),
  'The checkpoint is unavailable or does not match the selected model.',
  'Select matching official weights and model settings; check the checkpoint path in the log.'),
 ('ENV-01','Error','Python dependency unavailable',
  ('modulenotfounderror','importerror','dll load failed','winerror 126','numpy.dtype size changed'),
  'The selected environment is missing a library or contains incompatible packages.',
  'Check the Python environment paths in Settings; repair the affected environment.'),
 ('FIG-01','Error','MATLAB export unavailable',
  ('matlab is required','matlab could not create','matlab executable'),
  'The editable FIG exporter could not find or start its MATLAB dependency.',
  'Check the MATLAB path in Settings, or export MAT/PNG/PDF instead.'),
 ('SRC-01','Warning','No new TIFF stacks selected',
  ('already in this session','select at least one new tiff','keep existing source stacks'),
  'The selection contains no new stacks or would change the existing source order.',
  'Choose new stacks from the same FOV and append them after the existing files.'),
 ('MAT-01','Error','Unsupported MAT metadata',
  ('could not convert','cannot convert','field names are restricted'),
  'A result metadata field cannot be written to this MAT format.',
  'Save the session and retain the diagnostic log for an exporter compatibility fix.'),
 ('DATA-01','Error','Invalid pixel or signal values',
  ('nan or inf','nonfinite','no intensity variation','heatmap values and time axis'),
  'The source contains nonfinite values, no intensity variation or inconsistent signal dimensions.',
  'Inspect the source movie and signal arrays. Correct invalid input before analysis.'),
 ('FLOW-01','Warning','Traces need recomputing',
  ('compute traces','recompute traces','roi identities changed','no rois are marked as cells','detect or add rois first','select valid roi','select a valid roi','select one or more rois','select an roi'),
  'The selected ROI set or trace results are not ready for this action.',
  'Select valid cell ROIs and run Compute traces for the current ROI set.'),
 ('FLOW-02','Warning','Registration not ready',
  ('run step 1','import tiff files first','import images first','prepared session first','preparation settings changed','wait for processing','finish manual add','finish add','use finish add','registration or denoising settings changed'),
  'A required earlier step is incomplete, or its settings have changed.',
  'Finish or stop the active operation, then rerun the required pipeline step.'),
]


def diagnose_error(detail):
    text=str(detail).lower()
    for code,level,title,signatures,cause,action in RULES:
        if any(signature in text for signature in signatures):
            return dict(code=code,level=level,title=title,cause=cause,action=action)
    return dict(code='OTHER-01',level='Error',title='Step failed',
                cause='The step failed for a reason not yet covered by the error catalogue.',
                action='Save the session and use Open log to inspect or share the diagnostic log.')


def error_summary(detail):
    if str(detail).startswith(('Error [','Warning [')): return str(detail)
    info=diagnose_error(detail)
    return f"{info['level']} [{info['code']}] {info['title']}\n\nReason: {info['cause']}\n\nNext: {info['action']}"


def catalogue_text():
    rows=['TwoPhoton Studio - common error reference','',
          'Warning: fix inputs/settings and retry. Error: the current step failed.',
          'Full tracebacks are saved in diagnostic logs, not displayed in popups.','']
    for code,level,title,_,cause,action in RULES:
        rows.extend([f'{level} [{code}] {title}','Reason: '+cause,'Next: '+action,''])
    rows.extend(['Error [OTHER-01] Step failed','Use Open log for unclassified failures.'])
    return '\n'.join(rows)+'\n'
