# Dependency and model inventory

| Component | Pinned release / source | Where installed |
| --- | --- | --- |
| GUI / legacy Suite2p | Python 3.10; Suite2p 0.10.3; NumPy 1.24.4; PyQt5 5.15.9 | gui environment |
| GUI Torch dependency | PyTorch 2.4.1+cpu | gui environment |
| Modern inference | Python 3.11; Suite2p 1.0.0.1; Cellpose 4.0.8 | models environment |
| GPU Torch | PyTorch 2.7.0+cu128; TorchVision 0.22.0+cu128 | models environment |
| CPU inference alternative | PyTorch 2.7.0+cpu; TorchVision 0.22.0+cpu | models environment |
| PROFILE | Vendored inference snapshot, exact SHA256 manifest | app/vendor/PROFILE |
| PROFILE default weights | mouse_target_scale_m124 + matching custom presets; m110 alternative | app/vendor/PROFILE/models |
| SUPPORT | Vendored inference snapshot + bs1.pth | app/vendor/SUPPORT |
| Cellpose-SAM weights | cpsam, pinned Hugging Face revision and SHA256 | .local/models/cellpose |
| DFT registration | dftregistration.m (2016), retained BSD notice | app/matlab |
| MATLAB | User installation, detected/configured executable | external |

Direct dependency pins are in requirements/core.txt and requirements/models.txt.
Those files load core-constraints.txt / models-constraints.txt, which constrain the
transitive Python package versions from the Windows release validation environments.
Torch local build suffixes are selected separately by the CPU/CUDA installer option.
Complete installed inventories are also written to .local/core-installed.txt and
.local/models-installed.txt. These are version constraints, not hash-locked wheel archives;
Python patch releases and Conda runtime packages are resolved by conda-forge.

PROFILE's bundled m124/m110 files use the names in the original release data package.
The presets reference those exact files; m124 is not a cell diameter. Other official JSONs
in vendor/PROFILE/configs may refer to weights not included in this distribution.
Select the bundled profile_configs presets, or separately obtain each additional official
checkpoint and its matching JSON. The default installation does not require Google Drive access.

Sources:
- [Suite2p](https://github.com/MouseLand/suite2p)
- [Cellpose](https://github.com/MouseLand/cellpose)
- [Cellpose-SAM checkpoint](https://huggingface.co/mouseland/cellpose-sam)
- [PROFILE and model package](https://github.com/NICALab/PROFILE)
- [SUPPORT](https://github.com/NICALab/SUPPORT)
- [Official PyTorch version-specific commands](https://pytorch.org/get-started/previous-versions/)

The vendored sources are a snapshot from the existing application. No unverified upstream
commit identifier is claimed. bundled-assets.json records exact SHA256 hashes.
GPU support is checked by running a kernel and model inference; torch.cuda.is_available()
alone is not treated as sufficient. CPU mode is supported by explicit installation choice.

