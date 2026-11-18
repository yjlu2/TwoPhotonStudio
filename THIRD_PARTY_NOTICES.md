# Third-party notices

TwoPhoton Studio's original application code is released under GPL-3.0-only
(see LICENSE), as selected by the project owner. Third-party works keep their existing terms.

- **Suite2p**: GPL-3.0, MouseLand / Howard Hughes Medical Institute.
  Installed from PyPI, not copied into this source tree.
  https://github.com/MouseLand/suite2p
- **Cellpose / Cellpose-SAM**: retain the upstream BSD license and author notices
  from the installed Cellpose package. The cpsam weights are obtained from the official
  model repository, not committed to this repository.
  https://github.com/MouseLand/cellpose
  https://huggingface.co/mouseland/cellpose-sam
- **PROFILE**: NICALab, MIT; original license and README retained in app/vendor/PROFILE.
  Includes inference source and m124/m110 weights from the accompanying release data package;
  local inference presets are supplied separately in app/profile_configs.
  https://github.com/NICALab/PROFILE
- **SUPPORT**: NICALab, GPL-3.0; original license/README retained in app/vendor/SUPPORT.
  Includes model/inference utilities and the bs1 pretrained checkpoint.
  https://github.com/NICALab/SUPPORT
- **dftregistration.m**: copyright 2016 Manuel Guizar Sicairos, James R. Fienup,
  University of Rochester. The original BSD-style license is retained inside the file.
  Cite Guizar-Sicairos, Thurman and Fienup, Optics Letters 33, 156–158 (2008).
- **PyQt5**: GPL/commercial dual licensing; this distribution uses the GPL-compatible path.
- **PyTorch, NumPy, SciPy, matplotlib, scikit-image, scikit-learn, tifffile, imagecodecs,
  h5py, pyqtgraph, numba** and other dependencies retain their own installed licenses.
- **MATLAB** is a separately licensed user installation and is not redistributed.

The included icon is the artwork supplied by the TwoPhoton Studio project owner.
No affiliation with or endorsement by upstream projects is implied.
The source tree contains the selected inference components, not the full upstream training
repositories; upstream URLs above provide the original projects.

