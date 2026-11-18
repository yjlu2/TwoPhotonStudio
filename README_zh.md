# TwoPhoton Studio：Windows / Conda 安装教程

<img src="app/assets/twophoton-studio.png" width="96" alt="TwoPhoton Studio 图标">

这是可放到 GitHub 的源码发布版 **0.0.1**，基于当前 TwoPhoton Studio 1.3.4 功能。
界面全部为英文。软件许可证为 GPL-3.0；第三方组件保留各自许可证。

## 1. 安装前准备

- Windows 10/11，64 位。当前版本没有宣称支持 Linux/macOS。
- 已安装 Miniforge、Miniconda 或 Anaconda。新用户可使用
  [Miniforge 官方安装包](https://github.com/conda-forge/miniforge#miniforge3)。
- GPU 模式需要 NVIDIA 显卡及兼容驱动。没有 NVIDIA GPU 时使用 CPU 安装命令。
- 建议为 CUDA 安装及缓存预留至少 20 GB；实际占用取决于下载和 Conda 包。
- MATLAB 按用户已安装处理，不需要另装 MATLAB Runtime。
- 首次安装需要访问 conda-forge、PyPI、PyTorch 和 Hugging Face。

## 2. 下载整个源码文件夹

在 GitHub 仓库中选择 **Code → Download ZIP**，解压到例如：

```text
D:\TwoPhotonStudio
```

这个文件夹内应直接包含 `install.py`、`Launch.cmd`、`app`、`requirements`。
不要只下载 studio.py。可以选择 F 盘等其他位置，后面的命令替换成真实路径。
不要在 ZIP 内运行，也不要放到只读目录。安装后不要直接搬动 Conda 环境。

如果使用 Git，在希望存放源码的目录打开 Anaconda Prompt，然后运行：

```bat
git clone https://github.com/yjlu2/TwoPhotonStudio.git
cd TwoPhotonStudio
```

源码仓库：[yjlu2/TwoPhotonStudio](https://github.com/yjlu2/TwoPhotonStudio)。
接着按第 3 节安装依赖。

## 3. 安装所有 Python 依赖及默认模型

打开 **Anaconda Prompt** 或 **Miniforge Prompt**。以下是 CMD/Anaconda Prompt 命令，
不是 MATLAB 命令：

```bat
cd /d D:\TwoPhotonStudio
python install.py --device cuda
```

没有 NVIDIA GPU / 希望使用 CPU：

```bat
cd /d D:\TwoPhotonStudio
python install.py --device cpu
```

安装脚本会完成：

1. 创建独立的 `.local\envs\gui` 与 `.local\envs\models` 环境。
2. 安装固定版本的 Suite2p、PyTorch、Cellpose-SAM、PyQt、NumPy、SciPy、
   TIFF 编解码、MAT 读写、PROFILE/SUPPORT 所需库及传递依赖。
3. 检查随源码提供的 PROFILE / SUPPORT 文件及默认模型的完整性。
4. 下载约 **1.23 GB** Cellpose-SAM 权重并核对官方固定版本的 SHA256。
5. 检查依赖冲突，启动无界面显示的 GUI 测试，运行 GPU 运算和三个模型的小图测试。
6. 写入当前机器的环境路径、启动脚本和安装报告。

请等到出现 **INSTALLATION COMPLETE**，不要只看到某一步 pip 成功就关闭窗口。
第一次安装耗时主要由网络下载决定；模型测试也需要时间。
脚本不会修改你已有的 suite2p、cellpose 或 base 环境。

如果想把环境放到其他盘：

```bat
python install.py --device cuda --env-root "E:\TwoPhotonEnvs"
```

若已下载完全相同的官方 cpsam 权重，可避免再次联网下载：

```bat
python install.py --device cuda --cellpose-weights "E:\Models\cpsam"
```

该参数仍会校验文件，不接受同名但内容不匹配的权重。

## 4. MATLAB 路径

脚本自动从 PATH、Windows 注册表和默认安装目录查找 MATLAB。
非标准路径可以直接指定：

```bat
python install.py --device cuda --matlab "D:\Applications\MATLAB_2025b\bin\matlab.exe"
```

也可在软件 **Settings → Models and environments → MATLAB executable** 中修改。
没有自动找到 MATLAB 不影响 Python 环境的安装，但使用 FIG 导出 / MATLAB 配准前
需要填写正确路径。软件不自动安装 MATLAB 或 MATLAB Runtime。

## 5. 启动

推荐直接双击仓库根目录的 **Launch.cmd**。

也可以像启动 Suite2p 一样先激活环境，再输入命令：

```bat
conda activate D:\TwoPhotonStudio\.local\envs\gui
twophoton-studio
```

如果用了自定义 `--env-root`，把上面的激活路径改成该目录下的 `gui`。

安装完成后处理数据不需要持续联网，默认模型都已存到本地。
目前未发布到 PyPI，所以不要使用 `pip install twophoton-studio` 代替本教程。

## 6. 检查安装 / 故障排查

在源码目录执行：

```bat
python install.py --check
```

报告保存在 `.local\doctor-report.json`，环境实际安装清单保存在
`.local\core-installed.txt` 和 `.local\models-installed.txt`。

| 提示 | 处理方式 |
| --- | --- |
| Conda was not found | 使用 Anaconda Prompt / Miniforge Prompt；检查 `conda --version` |
| 网络超时 / 下载失败 | 确认官方站点可访问，修复网络后重跑原安装命令 |
| SSL certificate verify failed | 更新 Conda/证书，或按单位网络要求配置可信 CA；不要关闭证书验证 |
| pip dependency conflict | 保存完整输出，避免手动将不同环境混装；重跑安装脚本 |
| CUDA unavailable / no kernel image | 更新显卡驱动，检查报告中的显卡能力；或重跑 `--device cpu` |
| CUDA out of memory | 关闭占用 GPU 的程序，减少分析批量，必要时用 CPU |
| WinError 112 / no space left | 更换 `--env-root` 或清理磁盘；模型和缓存仍位于源码目录的 .local |
| MATLAB not found automatically | 在安装参数或 Settings 中填写 matlab.exe 完整路径 |
| Refusing to modify an unmanaged environment | 选择新的环境目录，脚本不会覆盖你原有的 Conda 环境 |

安装失败可以重跑相同命令，已经创建且属于本仓库的环境会继续使用。
中断的模型下载会重新开始；SHA256 不一致不会被当作安装成功。
运行期间的分析错误仍在 GUI 中简要显示，并提供完整日志路径。
旧版 Suite2p 导入时可能提示 `cellpose did not import`：它指的是 GUI 环境内的可选模块。
本软件的 Cellpose-SAM 在独立 models 环境运行，安装检查会实际加载该模型验证。
安装过程完整输出另存为 `.local\install.log`，方便排查安装失败。

## 7. 日常分析及文件位置

- 默认帧率 23.2 Hz、tau 1 s；根据自己的实验修改。
- 一次导入同一 FOV 的多个单通道、单平面 TIFF stacks。
- 注册 / 可选去噪 → ROI 检测 / 手动复核 → Compute traces → 同步 MAT → 分析 / 导出。
- CPU / GPU 注册和提取由 Settings 控制；功能性检测和 OASIS 等仍可能使用 CPU。
- 改亮度、对比度仅影响显示。
- 分析输出仍按照导入及输出目录设置保存，不写进 Conda 环境。
- `.local` 存放本机环境、默认下载权重、配置、安装缓存和日志；
  **它不是实验结果文件夹**，安装后不要随意删除。
- GitHub 不提交 `.local`、个人设置、日志、TIFF、Fall.mat 或 session.npy。

安装成功并关闭安装程序后，如果需要节省磁盘，可以删除
`.local\pip-cache`、`.local\conda-pkgs` 和 `.local\tmp` 缓存/安装临时目录；
不要删除 envs、models、runtime.json 或 launch.cmd。

## 8. 更新 / 分享

本仓库已经包含默认 PROFILE / SUPPORT 权重。不要另从自己的分析目录复制整个环境。
发布者上传本源码目录；使用者按上述命令独立安装。
源码更新后，在原目录重跑安装命令。若要迁移到新路径，重新安装，不直接移动 Conda 目录。

科学结果的可靠性需要用自己的标注/实验验证；安装测试只检验软件和模型能运行。
详见 [验证记录](VALIDATION.md) 和 [依赖说明](docs/DEPENDENCIES.md)。

