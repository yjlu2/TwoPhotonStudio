# GitHub 仓库与后续发布

源码仓库：[yjlu2/TwoPhotonStudio](https://github.com/yjlu2/TwoPhotonStudio)。
发行版页面：[Releases](https://github.com/yjlu2/TwoPhotonStudio/releases)。

使用者按根目录 README_zh.md 安装，不需要下载发布者的 Conda 环境。
维护者通过 Git 或 GitHub Desktop 更新源码；默认模型超过网页单文件上传限制，
不要用网页逐个拖入文件。

## 提交更新

在已经克隆的仓库目录内执行：

```bat
git status
git add .
git diff --cached --stat
git commit -m "Describe the software update"
git push origin main
```

提交前检查文件列表。不要提交 `.local`、实验 TIFF、Fall.mat、session.npy、
个人设置、日志或 Conda runtimes；`.gitignore` 已排除这些内容。
不要使用 `git add -f .local`。

`.gitattributes` 保留 vendor 文件的原始字节，以保证 bundled-assets.json 的
SHA256 校验不受 Git 换行转换影响。修改随包第三方文件或模型时，需要相应更新清单。

## 制作发行版

1. 更新源码发行号、说明和测试记录；现有发行号为 0.0.1，GUI 功能版本为 1.3.4。
2. 运行安装自检与对应的回归验证，查看 GitHub Actions 结果。
3. 运行 `python scripts/make_source_zip.py` 生成源码 ZIP 和 SHA256；升级版本时同步修改该脚本中的文件名。
4. 在 GitHub Releases 创建对应版本的标签，上传 ZIP 和 SHA256 文件。
5. 保留 GPL-3.0 LICENSE 和各第三方许可证，不覆盖已经发布的版本标签。

本仓库包含 PROFILE / SUPPORT 默认权重，最大单文件约 32.4 MB。
约 1.23 GB 的 Cellpose-SAM 权重不提交，安装时从官方固定版本地址下载并校验。
尚未发布到 PyPI，不要把安装方式写成 `pip install twophoton-studio`。

